"""Timestamp association and position-only trajectory evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrajectoryMetrics:
    matched_poses: int
    alignment: str
    scale: float
    yaw_offset_deg: float
    rmse_m: float
    mean_m: float
    median_m: float
    std_m: float
    min_m: float
    max_m: float


def load_camera_timestamps(path: str | Path) -> np.ndarray:
    """Load the real capture times from a sauvc_traj_recorder camera index CSV."""
    frame_index = pd.read_csv(path)
    if "t" not in frame_index.columns:
        raise ValueError(f"{path} needs a 't' timestamp column")
    timestamps = frame_index["t"].to_numpy(dtype=float)
    if len(timestamps) == 0 or not np.all(np.isfinite(timestamps)):
        raise ValueError(f"{path} has no usable timestamps")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{path} timestamps must be strictly increasing")
    return timestamps


def load_ground_truth_positions(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load recorder odometry positions; supports t or timestamp time columns."""
    ground_truth = pd.read_csv(path).rename(columns={"t": "timestamp"})
    required = {"timestamp", "x", "y", "z"}
    missing = required - set(ground_truth.columns)
    if missing:
        raise ValueError(f"{path} missing ground-truth columns: {sorted(missing)}")
    ground_truth = ground_truth.sort_values("timestamp")
    times = ground_truth["timestamp"].to_numpy(dtype=float)
    positions = ground_truth[["x", "y", "z"]].to_numpy(dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path} must contain at least two increasing timestamps")
    return times, positions


def interpolate_positions(
    source_times: np.ndarray, source_positions: np.ndarray, query_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate position and return a mask for timestamps in range."""
    valid = (query_times >= source_times[0]) & (query_times <= source_times[-1])
    times = query_times[valid]
    positions = np.column_stack([
        np.interp(times, source_times, source_positions[:, axis])
        for axis in range(3)
    ])
    return positions, valid


def align_planar(
    estimate_xy: np.ndarray, ground_truth_xy: np.ndarray, allow_scale: bool = False
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit a proper 2-D rigid (or similarity) transform from estimate to ground truth."""
    if len(estimate_xy) < 3:
        raise ValueError("Need at least three matched poses for planar alignment")
    estimate_center = estimate_xy.mean(axis=0)
    ground_truth_center = ground_truth_xy.mean(axis=0)
    estimate_zero = estimate_xy - estimate_center
    ground_truth_zero = ground_truth_xy - ground_truth_center
    covariance = ground_truth_zero.T @ estimate_zero / len(estimate_xy)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    estimate_variance = float(np.sum(estimate_zero**2) / len(estimate_xy))
    if estimate_variance <= np.finfo(float).eps:
        raise ValueError("Estimated path has no planar motion; cannot align it")
    scale = (
        float(np.trace(np.diag(singular_values) @ correction) / estimate_variance)
        if allow_scale else 1.0
    )
    translation = ground_truth_center - scale * rotation @ estimate_center
    aligned = (scale * (rotation @ estimate_xy.T)).T + translation
    return aligned, scale, rotation, translation


def evaluate_planar_trajectory(
    timestamps: np.ndarray,
    estimated_positions: np.ndarray,
    gt_times: np.ndarray,
    gt_positions: np.ndarray,
    allow_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TrajectoryMetrics]:
    """Associate, align in XY, and calculate position error for each matched frame."""
    ground_truth, valid = interpolate_positions(gt_times, gt_positions, timestamps)
    estimate = estimated_positions[valid]
    aligned_xy, scale, rotation, _ = align_planar(
        estimate[:, :2], ground_truth[:, :2], allow_scale=allow_scale
    )
    aligned = estimate.copy()
    aligned[:, :2] = aligned_xy
    # The current filter's depth measurement is intentionally not used for this
    # benchmark: recorder fluid_pressure units need calibration first.
    errors = np.linalg.norm(aligned[:, :2] - ground_truth[:, :2], axis=1)
    yaw_offset_deg = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    metrics = TrajectoryMetrics(
        matched_poses=int(len(errors)),
        alignment="Sim(2)" if allow_scale else "SE(2)",
        scale=scale,
        yaw_offset_deg=yaw_offset_deg,
        rmse_m=float(np.sqrt(np.mean(errors**2))),
        mean_m=float(np.mean(errors)),
        median_m=float(np.median(errors)),
        std_m=float(np.std(errors)),
        min_m=float(np.min(errors)),
        max_m=float(np.max(errors)),
    )
    return aligned, ground_truth, errors, metrics


def write_evaluation(
    out_dir: str | Path,
    timestamps: np.ndarray,
    estimate: np.ndarray,
    aligned: np.ndarray,
    ground_truth: np.ndarray,
    errors: np.ndarray,
    metrics: TrajectoryMetrics,
) -> None:
    """Write machine-readable metrics and matched per-frame position errors."""
    out_dir = Path(out_dir)
    with (out_dir / "trajectory_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(asdict(metrics), stream, indent=2)
        stream.write("\n")
    with (out_dir / "trajectory_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "timestamp_s", "estimate_x_m", "estimate_y_m", "estimate_z_m",
            "aligned_x_m", "aligned_y_m", "aligned_z_m",
            "ground_truth_x_m", "ground_truth_y_m", "ground_truth_z_m",
            "planar_error_m",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for timestamp, raw, fitted, truth, error in zip(
            timestamps, estimate, aligned, ground_truth, errors, strict=True
        ):
            writer.writerow({
                "timestamp_s": f"{timestamp:.9f}",
                "estimate_x_m": raw[0], "estimate_y_m": raw[1], "estimate_z_m": raw[2],
                "aligned_x_m": fitted[0], "aligned_y_m": fitted[1], "aligned_z_m": fitted[2],
                "ground_truth_x_m": truth[0], "ground_truth_y_m": truth[1],
                "ground_truth_z_m": truth[2], "planar_error_m": error,
            })
