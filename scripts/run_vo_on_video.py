#!/usr/bin/env python3
"""Run visual odometry + EKF on a tile-floor video."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fusion.sensors import load_imu_csv, load_pressure_csv
from src.pipeline.vo_pipeline import VOPipeline
from src.utils.trajectory_eval import (
    evaluate_planar_trajectory,
    interpolate_positions,
    load_camera_timestamps,
    load_ground_truth_positions,
    write_evaluation,
)
from src.utils.io import camera_from_config, ensure_dir, load_config
from src.utils.visualization import plot_trajectory, plot_trajectory_comparison


def _video_paths(args: argparse.Namespace, cfg: dict) -> list[Path]:
    """Resolve explicit videos, or all MP4s in a supplied/configured directory."""
    if args.video:
        paths = [Path(path).expanduser() for path in args.video]
    else:
        directory = Path(args.video_dir or cfg["paths"]["video_dir"]).expanduser()
        if not directory.is_absolute():
            directory = ROOT / directory
        paths = sorted(directory.glob("*.mp4"))
    if not paths:
        raise FileNotFoundError("No MP4 files found. Use --video or --video-dir.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Video files not found:\n" + "\n".join(missing))
    return paths


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


def _run_one(video_path: Path, args: argparse.Namespace, cfg: dict, camera, imu, heading_imu, pressure, out_dir: Path, pipeline_class) -> dict:
    video_out = ensure_dir(out_dir / video_path.stem)
    frame_timestamps = load_camera_timestamps(args.timestamps) if args.timestamps else None
    frame_stride = args.frame_stride if args.frame_stride is not None else (1 if frame_timestamps is not None else 2)
    pipeline = pipeline_class(cfg, camera)
    result = pipeline.process_video(
        video_path, imu_samples=imu, heading_samples=heading_imu,
        pressure_samples=pressure,
        max_frames=args.max_frames, frame_stride=frame_stride,
        frame_timestamps=frame_timestamps,
        max_visual_reference_s=args.max_visual_reference_s,
    )
    plot_trajectory(result.trajectory, video_out / "trajectory.png")
    _write_frame_log(result, video_out / "frame_log.csv")

    accepted = sum(1 for f in result.frames if f.visual is not None and f.visual.accepted)
    pairs = max(0, len(result.frames) - 1)
    displacement = 0.0
    if result.trajectory.size:
        displacement = float(np.linalg.norm(result.trajectory[-1, :2] - result.trajectory[0, :2]))
    row = {
        "video": video_path.name, "processed_frames": len(result.frames),
        "accepted_updates": accepted, "candidate_pairs": pairs,
        "acceptance_rate": accepted / pairs if pairs else 0.0,
        "planar_displacement_m": displacement,
        "output_dir": str(video_out),
    }
    if args.ground_truth:
        gt_times, gt_positions = load_ground_truth_positions(args.ground_truth)
        result_times = np.asarray([frame.timestamp for frame in result.frames])
        aligned, ground_truth, errors, metrics = evaluate_planar_trajectory(
            result_times, result.trajectory, gt_times, gt_positions,
            allow_scale=args.allow_scale_alignment,
        )
        _, in_range = interpolate_positions(gt_times, gt_positions, result_times)
        write_evaluation(
            video_out, result_times[in_range], result.trajectory[in_range], aligned,
            ground_truth, errors, metrics,
        )
        plot_trajectory_comparison(ground_truth, aligned, metrics, video_out / "trajectory_comparison.png")
        row.update({"ate_rmse_m": metrics.rmse_m, "ate_mean_m": metrics.mean_m,
                    "alignment": metrics.alignment, "alignment_scale": metrics.scale})
        print(f"  {metrics.alignment} ATE RMSE: {metrics.rmse_m:.3f} m "
              f"(mean {metrics.mean_m:.3f} m, scale {metrics.scale:.4f})")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="SAUVC VO + EKF pipeline")
    parser.add_argument("--video", action="append", help="Video path; may be repeated")
    parser.add_argument("--video-dir", help="Process every .mp4 in this directory")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--matcher", choices=("superpoint", "elo_ftr"), default="superpoint",
                        help="Matching backend: existing SuperPoint/LightGlue or EfficientLoFTR")
    parser.add_argument("--imu", default=None, help="Optional IMU CSV")
    parser.add_argument("--heading-imu", default=None,
                        help="IMU CSV used only for quaternion heading; does not integrate acceleration")
    parser.add_argument("--pressure", default=None, help="Optional pressure CSV")
    parser.add_argument("--timestamps", help="Camera index CSV with real per-frame 't' times")
    parser.add_argument("--ground-truth", help="Odometry CSV with t/x/y/z for trajectory scoring")
    parser.add_argument("--allow-scale-alignment", action="store_true",
                        help="Use Sim(2) only for scale-ambiguous monocular evaluation")
    parser.add_argument("--tile-spacing-m", type=float,
                        help="Known physical tile period in metres; overrides the config")
    parser.add_argument("--tile-spacing-px", type=float,
                        help="Measured tile period in pixels; avoids unreliable automatic scale inference")
    parser.add_argument("--down-camera", action="store_true",
                        help="Map down-camera image axes to AUV forward/right body axes")
    parser.add_argument("--imu-heading-yaw-sign", type=float, default=1.0,
                        help="Sign converting IMU quaternion yaw to navigation yaw (trajectory_05 uses -1)")
    parser.add_argument("--min-parallax-px", type=float,
                        help="Override visual minimum parallax; raise it to reject weak tile matches")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None,
                        help="Process every Nth source frame (default: 1 with --timestamps, else 2)")
    parser.add_argument("--max-visual-reference-s", type=float, default=0.0,
                        help="Maximum time to retain a failed visual reference (default: 0, strict frame-to-frame)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: paths.output_dir from the config)",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    if args.tile_spacing_m is not None:
        if args.tile_spacing_m <= 0:
            parser.error("--tile-spacing-m must be positive")
        cfg.setdefault("tiles", {})["spacing_m"] = args.tile_spacing_m
    if args.tile_spacing_px is not None:
        if args.tile_spacing_px <= 0:
            parser.error("--tile-spacing-px must be positive")
        cfg.setdefault("tiles", {})["spacing_px"] = args.tile_spacing_px
    if args.down_camera:
        cfg.setdefault("visual", {})["down_camera_axes"] = True
    if args.heading_imu:
        cfg.setdefault("imu", {})["heading_yaw_sign"] = args.imu_heading_yaw_sign
    if args.min_parallax_px is not None:
        if args.min_parallax_px <= 0:
            parser.error("--min-parallax-px must be positive")
        cfg.setdefault("quality_gate", {})["min_parallax_px"] = args.min_parallax_px
    camera = camera_from_config(cfg)
    pipeline_class = VOPipeline
    if args.matcher == "elo_ftr":
        from src.pipeline.elo_ftr_vo_pipeline import ELoFTRVOPipeline

        pipeline_class = ELoFTRVOPipeline
    out_dir = Path(args.output or cfg["paths"]["output_dir"]).expanduser()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir = ensure_dir(out_dir)

    imu = load_imu_csv(args.imu) if args.imu else None
    heading_imu = load_imu_csv(args.heading_imu) if args.heading_imu else None
    pressure = load_pressure_csv(args.pressure) if args.pressure else None

    rows = []
    for video_path in _video_paths(args, cfg):
        print(f"Processing: {video_path}")
        row = _run_one(video_path, args, cfg, camera, imu, heading_imu, pressure, out_dir, pipeline_class)
        rows.append(row)
        print(
            f"  accepted {row['accepted_updates']}/{row['candidate_pairs']} visual updates; "
            f"outputs: {row['output_dir']}"
        )
    with (out_dir / "batch_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Batch summary: {out_dir / 'batch_summary.csv'}")


if __name__ == "__main__":
    main()
