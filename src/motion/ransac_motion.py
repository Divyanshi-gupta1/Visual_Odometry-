from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from src.features.quality_gate import QualityDecision, evaluate_visual_update
from src.utils.io import CameraIntrinsics


@dataclass
class VisualMotion:
    accepted: bool
    reason: str
    tx: float  # meters, body/world x
    ty: float  # meters, body/world y
    dyaw: float  # radians
    inlier_mask: np.ndarray
    homography: np.ndarray | None
    quality: QualityDecision | None


def estimate_planar_motion(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    camera: CameraIntrinsics,
    tile_spacing_m: float,
    quality_cfg: dict,
    tile_spacing_px: float = 0.0,
    dt: float | None = None,
    max_step_m: float = 0.0,
    prior_tx: float = 0.0,
    prior_ty: float = 0.0,
) -> VisualMotion:
    """Estimate 2D motion over flat pool floor using homography + tile scale."""
    if len(kpts0) < quality_cfg.get("min_matches", 12):
        return VisualMotion(
            False, "too_few_matches", 0, 0, 0,
            np.array([]), None, None,
        )

    H, mask = cv2.findHomography(
        kpts0,
        kpts1,
        cv2.RANSAC,
        quality_cfg.get("ransac_reproj_threshold", 3.0),
    )

    qcfg = quality_cfg if "min_inlier_ratio" in quality_cfg else quality_cfg
    gate_cfg = {
        "min_inlier_ratio": qcfg.get("min_inlier_ratio", 0.35),
        "max_reproj_error": qcfg.get("max_reproj_error", 2.5),
        "min_parallax_px": qcfg.get("min_parallax_px", 2.0),
        "max_rotation_deg": qcfg.get("max_rotation_deg", 25.0),
    }

    decision = evaluate_visual_update(
        kpts0, kpts1, mask if mask is not None else np.zeros((len(kpts0), 1)),
        H, **gate_cfg,
    )

    if H is None:
        return VisualMotion(
            False, decision.reason, 0, 0, 0,
            mask.ravel() if mask is not None else np.array([]),
            H, decision,
        )

    if tile_spacing_px > 0:
        scale = tile_spacing_m / tile_spacing_px
        spacing_m = tile_spacing_m
    else:
        scale = _estimate_metric_scale(
            kpts0, kpts1, mask if mask is not None else np.ones((len(kpts0), 1)),
            tile_spacing_m,
        )
        spacing_m = tile_spacing_m

    # Median inlier flow is more stable than H[0,2] on a repetitive grid.
    tx_px, ty_px = _median_flow_px(kpts0, kpts1, mask)
    _, _, dyaw = _decompose_homography(H, camera.K)
    tx = tx_px * scale
    ty = ty_px * scale

    # Dense matchers often lock onto a neighbouring identical tile. Fold that
    # integer-period error back into (-s/2, s/2) so the AUV still advances.
    tx, ty = _resolve_tile_period(tx, ty, spacing_m, prior_tx, prior_ty)
    # Always fold into ±half a tile so a 1-period mismatch cannot stall the path.
    tx = _wrap_half_period(tx, spacing_m)
    ty = _wrap_half_period(ty, spacing_m)

    step = float(np.hypot(tx, ty))
    if max_step_m > 0:
        # max_step_m is metres per accepted pair, not metres per second.
        limit = float(max_step_m)
        if step > limit:
            return VisualMotion(
                False, "excessive_step", tx, ty, dyaw,
                mask.ravel() if mask is not None else np.array([]),
                H, decision,
            )

    # Stationary frames (tiny flow) are valid zero-motion updates, not failures.
    # Only reject the match geometry itself (inliers / reprojection / rotation).
    if decision.reason in ("low_inlier_ratio", "high_reproj_error",
                           "excessive_rotation", "homography_failed", "no_matches"):
        return VisualMotion(
            False, decision.reason, tx, ty, dyaw,
            mask.ravel() if mask is not None else np.array([]),
            H, decision,
        )

    return VisualMotion(
        True, "ok", tx, ty, dyaw,
        mask.ravel() if mask is not None else np.array([]),
        H, decision,
    )


def _wrap_half_period(value: float, spacing: float) -> float:
    if spacing <= 1e-6:
        return value
    return float((value + 0.5 * spacing) % spacing - 0.5 * spacing)


def _median_flow_px(
    kpts0: np.ndarray, kpts1: np.ndarray, mask: np.ndarray | None
) -> tuple[float, float]:
    if mask is None or len(kpts0) == 0:
        return 0.0, 0.0
    inliers = mask.ravel().astype(bool)
    if int(inliers.sum()) < 4:
        inliers = np.ones(len(kpts0), dtype=bool)
    flow = kpts1[inliers] - kpts0[inliers]
    return float(np.median(flow[:, 0])), float(np.median(flow[:, 1]))


def _resolve_tile_period(
    tx: float,
    ty: float,
    spacing_m: float,
    prior_tx: float = 0.0,
    prior_ty: float = 0.0,
    search_radius: int = 2,
) -> tuple[float, float]:
    """Pick the tile-period hypothesis closest to the current motion prior."""
    if spacing_m <= 1e-6:
        return tx, ty
    candidates: list[tuple[float, float]] = []
    for ix in range(-search_radius, search_radius + 1):
        for iy in range(-search_radius, search_radius + 1):
            candidates.append((tx + ix * spacing_m, ty + iy * spacing_m))

    prior_norm = float(np.hypot(prior_tx, prior_ty))
    if prior_norm < 0.015:
        return min(candidates, key=lambda c: c[0] ** 2 + c[1] ** 2)

    def score(c: tuple[float, float]) -> float:
        cx, cy = c
        dist = (cx - prior_tx) ** 2 + (cy - prior_ty) ** 2
        mag = max(float(np.hypot(cx, cy)), 1e-6)
        alignment = (cx * prior_tx + cy * prior_ty) / (mag * prior_norm)
        return dist - 0.15 * alignment

    return min(candidates, key=score)


def _decompose_homography(H: np.ndarray, K: np.ndarray) -> tuple[float, float, float]:
    """
    Planar motion for downward-facing camera.
    Translation is read directly from H (pixel space); yaw from upper 2×2 block.
    """
    tx_px = float(H[0, 2])
    ty_px = float(H[1, 2])

    # Normalize rotation sub-block for yaw estimate
    r00, r10 = float(H[0, 0]), float(H[1, 0])
    norm = max(np.hypot(r00, r10), 1e-6)
    dyaw = float(np.arctan2(r10 / norm, r00 / norm))
    return tx_px, ty_px, dyaw


def _estimate_metric_scale(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    inlier_mask: np.ndarray,
    tile_spacing_m: float,
) -> float:
    """
    Estimate meters-per-pixel from regular grid spacing among inlier keypoints.
    Uses distance histogram peak (tile period) rather than naive NN.
    """
    inliers = inlier_mask.ravel().astype(bool)
    pts = kpts0[inliers]
    if len(pts) < 6:
        return tile_spacing_m / 80.0

    # Sample pairwise distances (subset for speed)
    dists: list[float] = []
    n = min(len(pts), 80)
    subset = pts[:n]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(subset[i] - subset[j]))
            if 15.0 < d < 400.0:
                dists.append(d)

    if not dists:
        return tile_spacing_m / 80.0

    # First histogram peak ≈ tile spacing in pixels
    hist, edges = np.histogram(dists, bins=40)
    peak_idx = int(np.argmax(hist))
    spacing_px = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])

    # Grid can also produce 2x, sqrt(2)x multiples — pick smallest strong peak
    sorted_bins = np.argsort(hist)[::-1]
    for idx in sorted_bins[:5]:
        if hist[idx] < max(3, 0.25 * hist[peak_idx]):
            continue
        candidate = 0.5 * (edges[idx] + edges[idx + 1])
        if candidate >= 15.0:
            spacing_px = min(spacing_px, candidate)

    spacing_px = max(spacing_px, 15.0)
    return tile_spacing_m / spacing_px
