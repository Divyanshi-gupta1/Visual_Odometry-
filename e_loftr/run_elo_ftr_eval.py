#!/usr/bin/env python3
"""Run EfficientLoFTR + EKF on the down-camera recording and write error plots.

This keeps SuperPoint/LightGlue untouched.  It uses the same timestamps, IMU
heading, RANSAC quality gate, and EKF as the existing runner, then writes:

  trajectory.png
  trajectory_comparison.png
  error_analysis.png          (ATE vs time, accumulated error, histogram)
  trajectory_comparison.csv
  trajectory_metrics.json
  frame_log.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fusion.sensors import load_imu_csv, load_pressure_csv
from src.pipeline.elo_ftr_vo_pipeline import ELoFTRVOPipeline
from src.utils.io import camera_from_config, ensure_dir, load_config
from src.utils.trajectory_eval import (
    evaluate_planar_trajectory,
    interpolate_positions,
    load_camera_timestamps,
    load_ground_truth_positions,
    write_evaluation,
)
from src.utils.visualization import (
    plot_error_analysis,
    plot_trajectory,
    plot_trajectory_comparison,
)


def _initial_pose(gt_csv: Path, t0: float) -> tuple[float, float, float, float]:
    gt = pd.read_csv(gt_csv).rename(columns={"t": "timestamp"})
    if "timestamp" not in gt.columns:
        raise ValueError(f"{gt_csv} needs a t/timestamp column")
    idx = (gt["timestamp"] - t0).abs().idxmin()
    row = gt.iloc[idx]
    yaw = 0.0
    if {"qx", "qy", "qz", "qw"}.issubset(row.index):
        qw, qx, qy, qz = float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])
        yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    return float(row["x"]), float(row["y"]), float(row["z"]), yaw


def _write_frame_log(result, out_path: Path) -> None:
    fields = [
        "source_frame", "timestamp_s", "visual_accepted", "visual_reason",
        "matches", "inliers", "inlier_ratio", "reprojection_error_px",
        "parallax_px", "rotation_deg", "x_m", "y_m", "z_m", "yaw_rad",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame in result.frames:
            visual = frame.visual
            metrics = visual.quality.metrics if visual and visual.quality else None
            writer.writerow({
                "source_frame": frame.frame_idx,
                "timestamp_s": f"{frame.timestamp:.6f}",
                "visual_accepted": bool(visual and visual.accepted),
                "visual_reason": visual.reason if visual else "initial_frame",
                "matches": metrics.num_matches if metrics else 0,
                "inliers": metrics.num_inliers if metrics else 0,
                "inlier_ratio": metrics.inlier_ratio if metrics else 0,
                "reprojection_error_px": metrics.mean_reproj_error if metrics else 0,
                "parallax_px": metrics.median_parallax if metrics else 0,
                "rotation_deg": metrics.rotation_deg if metrics else 0,
                "x_m": frame.position[0], "y_m": frame.position[1],
                "z_m": frame.position[2], "yaw_rad": frame.yaw,
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="E-LoFTR + EKF trajectory evaluation")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--video",
        default="/Users/macbook/Documents/VO/data/videos/camera_down.mp4",
    )
    parser.add_argument(
        "--timestamps",
        default="/Users/macbook/Documents/VO/data/camera_down_index.csv",
    )
    parser.add_argument("--heading-imu", default="/Users/macbook/Documents/VO/data/imu.csv")
    parser.add_argument("--pressure", default="/Users/macbook/Documents/VO/data/pressure.csv")
    parser.add_argument("--ground-truth", default="/Users/macbook/Documents/VO/data/odometry.csv")
    parser.add_argument("--output", default="/Users/macbook/Documents/VO/output/trajectory_05_down_elo_ftr")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tile-spacing-m", type=float, default=0.30)
    parser.add_argument("--tile-spacing-px", type=float, default=34.0)
    parser.add_argument("--min-parallax-px", type=float, default=18.0)
    parser.add_argument("--imu-heading-yaw-sign", type=float, default=-1.0)
    parser.add_argument("--max-error-target-m", type=float, default=1.0)
    parser.add_argument("--allow-scale-alignment", action="store_true")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    cfg.setdefault("tiles", {})["spacing_m"] = args.tile_spacing_m
    cfg.setdefault("tiles", {})["spacing_px"] = args.tile_spacing_px
    cfg.setdefault("visual", {})["down_camera_axes"] = True
    cfg.setdefault("imu", {})["heading_yaw_sign"] = args.imu_heading_yaw_sign
    cfg.setdefault("quality_gate", {})["min_parallax_px"] = args.min_parallax_px
    cfg.setdefault("elo_ftr", {}).setdefault("max_step_m", 0.80)

    camera = camera_from_config(cfg)
    out_dir = ensure_dir(Path(args.output).expanduser())
    video_out = ensure_dir(out_dir / Path(args.video).stem)

    timestamps = load_camera_timestamps(args.timestamps)
    heading = load_imu_csv(args.heading_imu) if args.heading_imu else None
    pressure = load_pressure_csv(args.pressure) if Path(args.pressure).is_file() else None
    pose0 = _initial_pose(Path(args.ground_truth), float(timestamps[0]))

    pipeline = ELoFTRVOPipeline(cfg, camera)
    print(f"E-LoFTR backend: {pipeline.matcher.backend} on {pipeline.matcher.device}")
    result = pipeline.process_video(
        args.video,
        heading_samples=heading,
        pressure_samples=pressure,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        frame_timestamps=timestamps,
        initial_pose=pose0,
    )

    plot_trajectory(result.trajectory, video_out / "trajectory.png")
    _write_frame_log(result, video_out / "frame_log.csv")

    result_times = np.asarray([frame.timestamp for frame in result.frames])
    gt_times, gt_positions = load_ground_truth_positions(args.ground_truth)
    aligned, ground_truth, errors, metrics = evaluate_planar_trajectory(
        result_times,
        result.trajectory,
        gt_times,
        gt_positions,
        allow_scale=args.allow_scale_alignment,
    )
    _, in_range = interpolate_positions(gt_times, gt_positions, result_times)
    write_evaluation(
        video_out,
        result_times[in_range],
        result.trajectory[in_range],
        aligned,
        ground_truth,
        errors,
        metrics,
    )
    plot_trajectory_comparison(
        ground_truth, aligned, metrics, video_out / "trajectory_comparison.png"
    )
    plot_error_analysis(
        result_times[in_range],
        aligned,
        ground_truth,
        errors,
        metrics,
        video_out / "error_analysis.png",
        max_error_target_m=args.max_error_target_m,
    )

    accepted = sum(1 for f in result.frames if f.visual is not None and f.visual.accepted)
    pairs = max(0, len(result.frames) - 1)
    print("=" * 64)
    print("E-LoFTR + EKF REPORT")
    print("=" * 64)
    print(f"backend: {pipeline.matcher.backend}")
    print(f"frames: {len(result.frames)}  accepted: {accepted}/{pairs}")
    print(f"{metrics.alignment} RMSE: {metrics.rmse_m:.3f} m  mean: {metrics.mean_m:.3f} m")
    print(f"max accumulated ATE: {metrics.max_m:.3f} m  target: {args.max_error_target_m:.3f} m")
    print(f"plots: {video_out}")
    if metrics.max_m > args.max_error_target_m:
        print(
            f"WARNING: peak planar error {metrics.max_m:.3f} m exceeds "
            f"{args.max_error_target_m:.3f} m"
        )
    print("=" * 64)


if __name__ == "__main__":
    main()
