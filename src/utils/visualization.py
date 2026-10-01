from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    max_draw: int = 80,
) -> np.ndarray:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0 if img0.ndim == 3 else cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    canvas[:h1, w0 : w0 + w1] = img1 if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    n = min(len(kpts0), max_draw)
    for i in range(n):
        p0 = tuple(np.round(kpts0[i]).astype(int))
        p1 = tuple(np.round(kpts1[i]).astype(int) + np.array([w0, 0]))
        if inlier_mask is not None and len(inlier_mask) > i:
            color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
        else:
            color = (0, 255, 255)
        cv2.circle(canvas, p0, 3, color, -1)
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.line(canvas, p0, p1, color, 1)
    return canvas


def _axis_limits(xs: np.ndarray, ys: np.ndarray, pad_frac: float = 0.12, min_span: float = 1.0):
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    span_x = max(x1 - x0, min_span)
    span_y = max(y1 - y0, min_span)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * max(span_x, span_y) * (1.0 + pad_frac)
    return cx - half, cx + half, cy - half, cy + half


def plot_trajectory(traj: np.ndarray, out_path: str | Path) -> None:
    if traj.size == 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(traj[:, 0], traj[:, 1], "-b", linewidth=2.4, label="Predicted path")
    ax.scatter(traj[0, 0], traj[0, 1], c="lime", edgecolors="k", s=90, zorder=5, label="start")
    ax.scatter(traj[-1, 0], traj[-1, 1], c="red", edgecolors="k", s=90, zorder=5, label="end")
    xmin, xmax, ymin, ymax = _axis_limits(traj[:, 0], traj[:, 1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("AUV trajectory (top-down, world frame)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectory_comparison(
    ground_truth: np.ndarray,
    aligned_estimate: np.ndarray,
    metrics: object,
    out_path: str | Path,
    raw_estimate: np.ndarray | None = None,
) -> None:
    """Plot ground truth vs predicted path with a readable axis span."""
    import matplotlib

    matplotlib.use("Agg")
    fig, axes = plt.subplots(1, 2 if raw_estimate is not None else 1, figsize=(14 if raw_estimate is not None else 8, 6))
    if raw_estimate is None:
        axes = [axes]

    ax = axes[0]
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], "-k", linewidth=2.4, label="Ground truth")
    ax.plot(aligned_estimate[:, 0], aligned_estimate[:, 1], color="tab:blue", linewidth=2.0,
            label="Predicted (SE(2) aligned)")
    ax.scatter(ground_truth[0, 0], ground_truth[0, 1], c="lime", edgecolors="k", s=80, zorder=5, label="start")
    ax.scatter(ground_truth[-1, 0], ground_truth[-1, 1], c="red", edgecolors="k", s=80, zorder=5, label="end")
    xs = np.concatenate([ground_truth[:, 0], aligned_estimate[:, 0]])
    ys = np.concatenate([ground_truth[:, 1], aligned_estimate[:, 1]])
    xmin, xmax, ymin, ymax = _axis_limits(xs, ys)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Aligned comparison — {metrics.alignment}, RMSE {metrics.rmse_m:.3f} m"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if raw_estimate is not None:
        ax = axes[1]
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], "-k", linewidth=2.4, label="Ground truth")
        ax.plot(raw_estimate[:, 0], raw_estimate[:, 1], color="tab:orange", linewidth=2.0,
                label="Predicted (world frame)")
        ax.scatter(ground_truth[0, 0], ground_truth[0, 1], c="lime", edgecolors="k", s=80, zorder=5)
        ax.scatter(raw_estimate[-1, 0], raw_estimate[-1, 1], c="tab:orange", edgecolors="k", s=70, zorder=5)
        xs = np.concatenate([ground_truth[:, 0], raw_estimate[:, 0]])
        ys = np.concatenate([ground_truth[:, 1], raw_estimate[:, 1]])
        xmin, xmax, ymin, ymax = _axis_limits(xs, ys)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Unaligned world-frame overlay")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_analysis(
    timestamps: np.ndarray,
    estimate: np.ndarray,
    ground_truth: np.ndarray,
    errors: np.ndarray,
    metrics: object,
    out_path: str | Path,
    max_error_target_m: float = 1.0,
) -> None:
    """Trajectory, error vs time, accumulated error, and error histogram."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_axis = timestamps - timestamps[0]
    est_path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(estimate[:, :2], axis=0), axis=1))])
    gt_path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ground_truth[:, :2], axis=0), axis=1))])
    accumulated = np.maximum.accumulate(errors)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], "k-", linewidth=2.6, label="Ground truth")
    ax.plot(estimate[:, 0], estimate[:, 1], color="tab:blue", linewidth=2.2, label="E-LoFTR + EKF")
    ax.scatter(ground_truth[0, 0], ground_truth[0, 1], c="lime", edgecolors="k", s=80, zorder=5, label="Start")
    ax.scatter(ground_truth[-1, 0], ground_truth[-1, 1], c="red", edgecolors="k", s=80, zorder=5, label="End")
    ax.scatter(estimate[-1, 0], estimate[-1, 1], c="tab:blue", edgecolors="k", s=70, zorder=5)
    xs = np.concatenate([ground_truth[:, 0], estimate[:, 0]])
    ys = np.concatenate([ground_truth[:, 1], estimate[:, 1]])
    xmin, xmax, ymin, ymax = _axis_limits(xs, ys)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Trajectory (top-down)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.plot(t_axis, errors, color="crimson", linewidth=1.6, label="Planar ATE")
    ax.fill_between(t_axis, errors, color="crimson", alpha=0.12)
    ax.axhline(getattr(metrics, "rmse_m", float(np.sqrt(np.mean(errors**2)))),
               color="navy", linestyle=":", label=f"RMSE {getattr(metrics, 'rmse_m', 0):.3f} m")
    ax.axhline(max_error_target_m, color="gray", linestyle="-.", label=f"{max_error_target_m:.1f} m target")
    ax.set_title("Error vs time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Planar error [m]")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(t_axis, accumulated, color="darkorange", linewidth=2.0, label="Peak accumulated ATE")
    ax.plot(t_axis, est_path, color="tab:blue", linestyle="--", label="Estimated path length")
    ax.plot(t_axis, gt_path, color="black", linestyle=":", label="GT path length")
    ax.axhline(max_error_target_m, color="gray", linestyle="-.", label=f"{max_error_target_m:.1f} m target")
    ax.set_title("Accumulated error and path length")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Metres")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t_axis, ground_truth[:, 0], "k-", linewidth=1.8, label="GT X")
    ax.plot(t_axis, estimate[:, 0], color="tab:blue", linewidth=1.6, label="Pred X")
    ax.plot(t_axis, ground_truth[:, 1], "k--", linewidth=1.8, label="GT Y")
    ax.plot(t_axis, estimate[:, 1], color="tab:orange", linewidth=1.6, label="Pred Y")
    ax.set_title("Position vs time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def annotate_frame(frame: np.ndarray, text_lines: list[str]) -> np.ndarray:
    out = frame.copy()
    y = 24
    for line in text_lines:
        cv2.putText(
            out, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
        )
        y += 22
    return out
