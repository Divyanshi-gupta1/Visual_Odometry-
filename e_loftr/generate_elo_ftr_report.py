#!/usr/bin/env python3
"""Generate E-LoFTR + EKF trajectory / error report for the down-camera dataset.

Creates a fresh output folder (does not overwrite SuperPoint results) with:

  trajectory.png
  trajectory_comparison.png
  error_analysis.png
  trajectory_comparison.csv
  trajectory_metrics.json
  frame_log.csv

Tuned defaults target peak planar ATE <= 1.0 m on the supplied recording.
"""

from __future__ import annotations

import argparse
import csv
import json
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


DATA = Path("/Users/macbook/Documents/VO/data")
DEFAULT_OUT = Path("/Users/macbook/Documents/VO/output/trajectory_05_down_elo_ftr")


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
        "parallax_px", "rotation_deg", "tx_m", "ty_m", "dyaw_rad",
        "x_m", "y_m", "z_m", "yaw_rad",
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
                "tx_m": visual.tx if visual else 0.0,
                "ty_m": visual.ty if visual else 0.0,
                "dyaw_rad": visual.dyaw if visual else 0.0,
                "x_m": frame.position[0],
                "y_m": frame.position[1],
                "z_m": frame.position[2],
                "yaw_rad": frame.yaw,
            })


def _apply_low_error_tuning(cfg: dict, args: argparse.Namespace) -> dict:
    """Tighten gates so bad tile matches cannot push ATE above the 1 m target."""
    cfg = dict(cfg)
    cfg.setdefault("tiles", {})
    cfg.setdefault("quality_gate", {})
    cfg.setdefault("elo_ftr", {})
    cfg.setdefault("visual", {})
    cfg.setdefault("imu", {})
    cfg.setdefault("ekf", {})

    cfg["tiles"]["spacing_m"] = args.tile_spacing_m
    cfg["tiles"]["spacing_px"] = args.tile_spacing_px
    cfg["visual"]["down_camera_axes"] = True
    cfg["visual"]["prefer_forward"] = True
    cfg["visual"]["max_step_m"] = args.max_step_m
    cfg["imu"]["heading_yaw_sign"] = args.imu_heading_yaw_sign

    # Quality gate tuned for dense E-LoFTR matches on a repetitive floor.
    cfg["quality_gate"].update({
        "min_matches": args.min_matches,
        "min_inlier_ratio": args.min_inlier_ratio,
        "max_reproj_error": args.max_reproj_error,
        "min_parallax_px": args.min_parallax_px,
        "max_rotation_deg": args.max_rotation_deg,
        "ransac_reproj_threshold": args.ransac_reproj_threshold,
    })
    cfg["elo_ftr"].update({
        "confidence_threshold": args.confidence_threshold,
        "max_side": args.max_side,
        "max_step_m": args.max_step_m,
        "device": args.device,
    })
    # Visual increments dominate XY; keep EKF process noise modest.
    cfg["ekf"].update({
        "sigma_visual_xy": args.sigma_visual_xy,
        "sigma_visual_yaw": args.sigma_visual_yaw,
        "sigma_accel": 0.05,
        "sigma_gyro": 0.02,
    })
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="E-LoFTR + EKF report generator")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--video", default=str(DATA / "videos/camera_down.mp4"))
    parser.add_argument("--timestamps", default=str(DATA / "camera_down_index.csv"))
    parser.add_argument("--heading-imu", default=str(DATA / "imu.csv"))
    parser.add_argument("--pressure", default=str(DATA / "pressure.csv"))
    parser.add_argument("--ground-truth", default=str(DATA / "odometry.csv"))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Skip source frames before this index (motion starts ~508)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tile-spacing-m", type=float, default=0.30)
    parser.add_argument("--tile-spacing-px", type=float, default=34.0)
    parser.add_argument("--min-parallax-px", type=float, default=0.5)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.28)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--max-reproj-error", type=float, default=3.0)
    parser.add_argument("--max-rotation-deg", type=float, default=20.0)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=3.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--max-step-m", type=float, default=0.18)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--sigma-visual-xy", type=float, default=0.03)
    parser.add_argument("--sigma-visual-yaw", type=float, default=0.02)
    parser.add_argument("--imu-heading-yaw-sign", type=float, default=-1.0)
    parser.add_argument("--device", default=None, help="cuda / mps / cpu")
    parser.add_argument("--max-error-target-m", type=float, default=1.0)
    parser.add_argument(
        "--allow-scale-alignment",
        action="store_true",
        help="Use Sim(2) alignment (diagnostic only)",
    )
    parser.add_argument(
        "--force-se2",
        action="store_true",
        help="Force SE(2) even if scale looks wrong (default)",
    )
    args = parser.parse_args()

    cfg = _apply_low_error_tuning(load_config(ROOT / args.config), args)
    camera = camera_from_config(cfg)
    out_dir = ensure_dir(Path(args.output).expanduser())
    video_out = ensure_dir(out_dir / Path(args.video).stem)

    timestamps = load_camera_timestamps(args.timestamps)
    heading = load_imu_csv(args.heading_imu) if args.heading_imu else None
    pressure = load_pressure_csv(args.pressure) if Path(args.pressure).is_file() else None
    start_ts = float(timestamps[min(args.start_frame, len(timestamps) - 1)])
    pose0 = _initial_pose(Path(args.ground_truth), start_ts)

    pipeline = ELoFTRVOPipeline(cfg, camera)
    print(f"E-LoFTR backend: {pipeline.matcher.backend} on {pipeline.matcher.device}")
    print(
        f"start_frame={args.start_frame} stride={args.frame_stride} "
        f"tile={args.tile_spacing_m}m/{args.tile_spacing_px}px "
        f"max_step={args.max_step_m}m min_parallax={args.min_parallax_px}px"
    )

    result = pipeline.process_video(
        args.video,
        heading_samples=heading,
        pressure_samples=pressure,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        start_frame=args.start_frame,
        frame_timestamps=timestamps,
        initial_pose=pose0,
        max_visual_reference_s=0.0,
    )

    plot_trajectory(result.trajectory, video_out / "trajectory.png")
    _write_frame_log(result, video_out / "frame_log.csv")

    result_times = np.asarray([frame.timestamp for frame in result.frames], dtype=float)
    gt_times, gt_positions = load_ground_truth_positions(args.ground_truth)
    allow_scale = bool(args.allow_scale_alignment) and not args.force_se2
    aligned, ground_truth, errors, metrics = evaluate_planar_trajectory(
        result_times,
        result.trajectory,
        gt_times,
        gt_positions,
        allow_scale=allow_scale,
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
        ground_truth,
        aligned,
        metrics,
        video_out / "trajectory_comparison.png",
        raw_estimate=result.trajectory[in_range],
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
    # Also score the unaligned world-frame path (seeded at GT pose).
    unaligned_err = np.linalg.norm(
        result.trajectory[in_range][:, :2] - ground_truth[:, :2], axis=1
    )

    accepted = sum(1 for f in result.frames if f.visual is not None and f.visual.accepted)
    pairs = max(0, len(result.frames) - 1)
    report = {
        "backend": pipeline.matcher.backend,
        "device": pipeline.matcher.device,
        "frames": len(result.frames),
        "accepted_updates": accepted,
        "candidate_pairs": pairs,
        "acceptance_rate": accepted / pairs if pairs else 0.0,
        "alignment": metrics.alignment,
        "rmse_m": metrics.rmse_m,
        "mean_m": metrics.mean_m,
        "median_m": metrics.median_m,
        "max_m": metrics.max_m,
        "unaligned_rmse_m": float(np.sqrt(np.mean(unaligned_err**2))),
        "unaligned_max_m": float(np.max(unaligned_err)),
        "predicted_path_length_m": float(
            np.sum(np.linalg.norm(np.diff(result.trajectory[:, :2], axis=0), axis=1))
        ) if len(result.trajectory) > 1 else 0.0,
        "max_error_target_m": args.max_error_target_m,
        "meets_1m_target": bool(metrics.max_m <= args.max_error_target_m),
        "output_dir": str(video_out),
    }
    with (video_out / "elo_ftr_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    print("=" * 64)
    print("E-LoFTR + EKF REPORT")
    print("=" * 64)
    for key, value in report.items():
        print(f"{key}: {value}")
    if metrics.max_m > args.max_error_target_m:
        print(
            f"WARNING: peak planar error {metrics.max_m:.3f} m exceeds "
            f"{args.max_error_target_m:.3f} m — try raising --min-parallax-px "
            "or lowering --max-step-m / --tile-spacing-px"
        )
    print("=" * 64)


if __name__ == "__main__":
    main()
