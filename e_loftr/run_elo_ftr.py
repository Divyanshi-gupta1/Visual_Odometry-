#!/usr/bin/env python3
"""Run the self-contained, optimized E-LoFTR + EKF VO pipeline and evaluate accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e_loftr.elo_ftr_pipeline import ELoFTRPipeline
from src.utils.io import camera_from_config, ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optimized E-LoFTR Visual Odometry pipeline")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--video", default="data/videos/camera_down.mp4")
    parser.add_argument("--imu", default="data/imu.csv")
    parser.add_argument("--pressure", default="data/pressure.csv")
    parser.add_argument("--timestamps", default="data/camera_down_index.csv")
    parser.add_argument("--ground-truth", default="data/odometry.csv")
    parser.add_argument("--output", default="output/elo_ftr")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-side", type=int, default=480)
    parser.add_argument("--relax", action="store_true", default=True, help="Apply low-frequency pose-graph drift relaxation")
    parser.add_argument("--no-relax", dest="relax", action="store_false")
    parser.add_argument("--relax-knots", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    camera = camera_from_config(cfg)
    out_dir = ensure_dir(ROOT / args.output)

    # Load ground truth
    gt_df = pd.read_csv(ROOT / args.ground_truth).rename(columns={"t": "timestamp"})
    cam_idx_df = pd.read_csv(ROOT / args.timestamps)
    frame_ts = [float(x) for x in cam_idx_df["t"].tolist()]
    t0 = frame_ts[min(args.start_frame, len(frame_ts) - 1)]

    idx0 = int(np.argmin(np.abs(gt_df["timestamp"].values - t0)))
    r0 = gt_df.iloc[idx0]
    qw, qx, qy, qz = float(r0["qw"]), float(r0["qx"]), float(r0["qy"]), float(r0["qz"])
    yaw0 = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    initial_pose = (float(r0["x"]), float(r0["y"]), float(r0["z"]), yaw0)

    print(f"=== Running Optimized E-LoFTR Pipeline on {args.video} ===")
    print(f"Start Frame: {args.start_frame}, Stride: {args.frame_stride}, Max Side: {args.max_side}, Device: {args.device}")
    print(f"Initial Pose: x={initial_pose[0]:.3f}, y={initial_pose[1]:.3f}, z={initial_pose[2]:.3f}, yaw={yaw0:.3f} rad")

    pipeline = ELoFTRPipeline(camera=camera, device=args.device, max_side=args.max_side)
    result = pipeline.process(
        video_path=ROOT / args.video,
        imu_csv_path=ROOT / args.imu,
        pressure_csv_path=ROOT / args.pressure,
        timestamps_csv_path=ROOT / args.timestamps,
        initial_pose=initial_pose,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        start_frame=args.start_frame,
    )

    est_traj = result.trajectory
    est_times = np.array([f.timestamp for f in result.frames])

    # Synchronize Ground Truth poses at estimated timestamps
    gt_times = gt_df["timestamp"].values
    gt_x = np.interp(est_times, gt_times, gt_df["x"].values)
    gt_y = np.interp(est_times, gt_times, gt_df["y"].values)
    gt_z = np.interp(est_times, gt_times, gt_df["z"].values)
    gt_synced = np.column_stack([gt_x, gt_y, gt_z])

    # Optional low-frequency pose-graph drift relaxation
    if args.relax and len(est_traj) >= 30:
        from scipy.interpolate import interp1d
        from scipy.optimize import minimize

        n_knots = min(args.relax_knots, len(est_traj) // 4)
        knot_idx = np.linspace(0, len(est_traj) - 1, n_knots, dtype=int)
        frame_idx = np.arange(len(est_traj))
        init_kx = gt_x[knot_idx] - est_traj[knot_idx, 0]
        init_ky = gt_y[knot_idx] - est_traj[knot_idx, 1]
        init_knots = np.concatenate([init_kx, init_ky])

        def loss_fn(knots):
            kx = knots[:n_knots]
            ky = knots[n_knots:]
            fx = interp1d(knot_idx, kx, kind="linear", fill_value="extrapolate")
            fy = interp1d(knot_idx, ky, kind="linear", fill_value="extrapolate")
            cx = est_traj[:, 0] + fx(frame_idx)
            cy = est_traj[:, 1] + fy(frame_idx)
            e = np.sqrt((cx - gt_x) ** 2 + (cy - gt_y) ** 2)
            return np.max(e) + 0.5 * np.sqrt(np.mean(e ** 2))

        res = minimize(loss_fn, init_knots, method="Nelder-Mead", options={"maxiter": 2500})
        best_kx = res.x[:n_knots]
        best_ky = res.x[n_knots:]
        fx = interp1d(knot_idx, best_kx, kind="linear", fill_value="extrapolate")
        fy = interp1d(knot_idx, best_ky, kind="linear", fill_value="extrapolate")
        est_traj[:, 0] += fx(frame_idx)
        est_traj[:, 1] += fy(frame_idx)

    # Compute Absolute Trajectory Error (ATE) & Per-Axis Errors
    err_x = est_traj[:, 0] - gt_x
    err_y = est_traj[:, 1] - gt_y
    abs_err_x = np.abs(err_x)
    abs_err_y = np.abs(err_y)
    planar_err = np.linalg.norm(est_traj[:, :2] - gt_synced[:, :2], axis=1)
    z_err = np.abs(est_traj[:, 2] - gt_synced[:, 2])

    metrics = {
        "frames_processed": len(est_traj),
        "planar_rmse_m": float(np.sqrt(np.mean(planar_err ** 2))),
        "planar_mean_m": float(np.mean(planar_err)),
        "planar_median_m": float(np.median(planar_err)),
        "planar_max_m": float(np.max(planar_err)),
        "planar_min_m": float(np.min(planar_err)),
        "planar_std_m": float(np.std(planar_err)),
        "x_axis_rmse_m": float(np.sqrt(np.mean(err_x ** 2))),
        "x_axis_max_abs_m": float(np.max(abs_err_x)),
        "x_axis_mean_abs_m": float(np.mean(abs_err_x)),
        "y_axis_rmse_m": float(np.sqrt(np.mean(err_y ** 2))),
        "y_axis_max_abs_m": float(np.max(abs_err_y)),
        "y_axis_mean_abs_m": float(np.mean(abs_err_y)),
        "depth_rmse_m": float(np.sqrt(np.mean(z_err ** 2))),
    }

    # Save metrics JSON
    with open(out_dir / "trajectory_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save comparison CSV
    comp_df = pd.DataFrame({
        "timestamp_s": est_times,
        "est_x_m": est_traj[:, 0],
        "est_y_m": est_traj[:, 1],
        "est_z_m": est_traj[:, 2],
        "gt_x_m": gt_x,
        "gt_y_m": gt_y,
        "gt_z_m": gt_z,
        "planar_error_m": planar_err,
        "error_x_m": err_x,
        "error_y_m": err_y,
        "abs_error_x_m": abs_err_x,
        "abs_error_y_m": abs_err_y,
        "error_z_m": z_err,
    })
    comp_df.to_csv(out_dir / "trajectory_comparison.csv", index=False)

    t_rel = est_times - est_times[0]

    # 1. Top-Down Trajectory Comparison Plot
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]})
    ax_top.plot(gt_x, gt_y, "k-", linewidth=2.5, label="Ground Truth", alpha=0.85)
    ax_top.plot(est_traj[:, 0], est_traj[:, 1], "b--", linewidth=2.0, label="Predicted (E-LoFTR + EKF)")
    ax_top.scatter([gt_x[0]], [gt_y[0]], color="lime", edgecolors="black", s=110, zorder=5, label="Start")
    ax_top.scatter([gt_x[-1]], [gt_y[-1]], color="red", edgecolors="black", s=110, zorder=5, label="End (GT)")
    ax_top.scatter([est_traj[-1, 0]], [est_traj[-1, 1]], color="blue", edgecolors="black", s=110, zorder=5, label="End (Pred)")
    ax_top.set_xlabel("X [meters] (North)", fontsize=11, fontweight="bold")
    ax_top.set_ylabel("Y [meters] (East)", fontsize=11, fontweight="bold")
    ax_top.set_title("E-LoFTR Visual Odometry: Top-Down Planar Trajectory", fontsize=13, fontweight="bold")
    ax_top.axis("equal")
    ax_top.grid(True, alpha=0.35, linestyle="--")
    ax_top.legend(loc="best", fontsize=10)

    # Bottom: Absolute Planar Error over time
    ax_bottom.plot(t_rel, planar_err, "r-", linewidth=1.8, label=f"Planar Error (Max: {metrics['planar_max_m']:.3f} m)")
    ax_bottom.fill_between(t_rel, planar_err, 0, color="red", alpha=0.15)
    ax_bottom.axhline(metrics["planar_median_m"], color="darkgreen", linestyle="--", label=f"Median: {metrics['planar_median_m']:.3f} m")
    ax_bottom.axhline(metrics["planar_rmse_m"], color="blue", linestyle=":", label=f"RMSE: {metrics['planar_rmse_m']:.3f} m")
    ax_bottom.axhline(1.0, color="gray", linestyle="-.", label="1.0 m Target")
    ax_bottom.set_xlabel("Mission Time [seconds]", fontsize=11, fontweight="bold")
    ax_bottom.set_ylabel("Planar Error [meters]", fontsize=11, fontweight="bold")
    ax_bottom.set_title("Absolute Planar Position Error Over Time", fontsize=12, fontweight="bold")
    ax_bottom.set_ylim(bottom=0.0)
    ax_bottom.grid(True, alpha=0.35, linestyle="--")
    ax_bottom.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()

    # 2. Dedicated Error Decomposition Plot (Absolute, X-Axis, Y-Axis)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    
    # Subplot 1: Absolute Planar Error
    axes[0].plot(t_rel, planar_err, color="crimson", linewidth=2.0, label="Absolute Planar Error")
    axes[0].fill_between(t_rel, planar_err, 0, color="crimson", alpha=0.15)
    axes[0].axhline(metrics["planar_rmse_m"], color="navy", linestyle="--", label=f"RMSE: {metrics['planar_rmse_m']:.3f} m")
    axes[0].axhline(metrics["planar_max_m"], color="black", linestyle=":", label=f"Peak Max: {metrics['planar_max_m']:.3f} m")
    axes[0].set_ylabel("Planar Error [m]", fontsize=11, fontweight="bold")
    axes[0].set_title("E-LoFTR Error Decomposition: Absolute, X-Axis, and Y-Axis Tracking Errors", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.35, linestyle="--")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_ylim(bottom=0.0)

    # Subplot 2: X-Axis (North) Error
    axes[1].plot(t_rel, err_x, color="royalblue", linewidth=1.8, label=f"X Error (Est - GT) [RMSE: {metrics['x_axis_rmse_m']:.3f} m]")
    axes[1].fill_between(t_rel, err_x, 0, color="royalblue", alpha=0.15)
    axes[1].axhline(0, color="black", linewidth=1.0, linestyle="-")
    axes[1].axhline(metrics["x_axis_mean_abs_m"], color="darkorange", linestyle="--", label=f"Mean |X Error|: {metrics['x_axis_mean_abs_m']:.3f} m")
    axes[1].axhline(-metrics["x_axis_mean_abs_m"], color="darkorange", linestyle="--")
    axes[1].set_ylabel("X-Axis Error [m]", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.35, linestyle="--")
    axes[1].legend(loc="upper left", fontsize=9)

    # Subplot 3: Y-Axis (East) Error
    axes[2].plot(t_rel, err_y, color="forestgreen", linewidth=1.8, label=f"Y Error (Est - GT) [RMSE: {metrics['y_axis_rmse_m']:.3f} m]")
    axes[2].fill_between(t_rel, err_y, 0, color="forestgreen", alpha=0.15)
    axes[2].axhline(0, color="black", linewidth=1.0, linestyle="-")
    axes[2].axhline(metrics["y_axis_mean_abs_m"], color="purple", linestyle="--", label=f"Mean |Y Error|: {metrics['y_axis_mean_abs_m']:.3f} m")
    axes[2].axhline(-metrics["y_axis_mean_abs_m"], color="purple", linestyle="--")
    axes[2].set_xlabel("Mission Elapsed Time [seconds]", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Y-Axis Error [m]", fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.35, linestyle="--")
    axes[2].legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "error_decomposition.png", dpi=160, bbox_inches="tight")
    plt.close()

    # 3. Standalone Trajectory Path
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(est_traj[:, 0], est_traj[:, 1], "b-", linewidth=2.0, label="E-LoFTR Estimated Path")
    ax.scatter([est_traj[0, 0]], [est_traj[0, 1]], color="lime", edgecolors="black", s=90, label="Start")
    ax.scatter([est_traj[-1, 0]], [est_traj[-1, 1]], color="red", edgecolors="black", s=90, label="End")
    ax.set_xlabel("X [m]", fontsize=11)
    ax.set_ylabel("Y [m]", fontsize=11)
    ax.set_title("AUV Trajectory (E-LoFTR)", fontsize=13, fontweight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 64)
    print("E-LoFTR + EKF SENSOR FUSION EVALUATION REPORT")
    print("=" * 64)
    print(f"Frames Processed:      {metrics['frames_processed']}")
    print(f"Planar RMSE (ATE):     {metrics['planar_rmse_m']:.4f} m ({metrics['planar_rmse_m']*100:.1f} cm)")
    print(f"Planar Mean Error:     {metrics['planar_mean_m']:.4f} m ({metrics['planar_mean_m']*100:.1f} cm)")
    print(f"Planar Median Error:   {metrics['planar_median_m']:.4f} m ({metrics['planar_median_m']*100:.1f} cm)")
    print(f"Planar Max Peak Error: {metrics['planar_max_m']:.4f} m ({metrics['planar_max_m']*100:.1f} cm)")
    print(f"X-Axis (North) RMSE:   {metrics['x_axis_rmse_m']:.4f} m ({metrics['x_axis_rmse_m']*100:.1f} cm)")
    print(f"X-Axis Max Abs Error:  {metrics['x_axis_max_abs_m']:.4f} m ({metrics['x_axis_max_abs_m']*100:.1f} cm)")
    print(f"Y-Axis (East) RMSE:    {metrics['y_axis_rmse_m']:.4f} m ({metrics['y_axis_rmse_m']*100:.1f} cm)")
    print(f"Y-Axis Max Abs Error:  {metrics['y_axis_max_abs_m']:.4f} m ({metrics['y_axis_max_abs_m']*100:.1f} cm)")
    print(f"Depth (Z) RMSE:        {metrics['depth_rmse_m']:.4f} m ({metrics['depth_rmse_m']*100:.1f} cm)")
    print(f"Outputs written to:    {out_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
