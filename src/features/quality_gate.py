from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class QualityMetrics:
    inlier_ratio: float
    mean_reproj_error: float
    median_parallax: float
    num_inliers: int
    num_matches: int
    rotation_deg: float


@dataclass
class QualityDecision:
    accepted: bool
    reason: str
    metrics: QualityMetrics


def evaluate_visual_update(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    inlier_mask: np.ndarray,
    homography: np.ndarray | None,
    min_inlier_ratio: float,
    max_reproj_error: float,
    min_parallax_px: float,
    max_rotation_deg: float,
) -> QualityDecision:
    n = len(kpts0)
    if n == 0:
        return QualityDecision(
            accepted=False,
            reason="no_matches",
            metrics=QualityMetrics(0, 0, 0, 0, 0, 0),
        )

    inliers = inlier_mask.ravel().astype(bool)
    num_inliers = int(inliers.sum())
    inlier_ratio = num_inliers / n

    if homography is None or num_inliers < 4:
        return QualityDecision(
            accepted=False,
            reason="homography_failed",
            metrics=QualityMetrics(inlier_ratio, 999, 0, num_inliers, n, 0),
        )

    reproj = _reprojection_errors(kpts0[inliers], kpts1[inliers], homography)
    mean_reproj = float(np.mean(reproj)) if reproj.size else 999.0

    parallax = np.linalg.norm(kpts1 - kpts0, axis=1)
    median_parallax = float(np.median(parallax[inliers])) if num_inliers else 0.0

    rot_deg = _homography_rotation_deg(homography)

    metrics = QualityMetrics(
        inlier_ratio=inlier_ratio,
        mean_reproj_error=mean_reproj,
        median_parallax=median_parallax,
        num_inliers=num_inliers,
        num_matches=n,
        rotation_deg=rot_deg,
    )

    if inlier_ratio < min_inlier_ratio:
        return QualityDecision(False, "low_inlier_ratio", metrics)
    if mean_reproj > max_reproj_error:
        return QualityDecision(False, "high_reproj_error", metrics)
    if median_parallax < min_parallax_px:
        return QualityDecision(False, "insufficient_parallax", metrics)
    if abs(rot_deg) > max_rotation_deg:
        return QualityDecision(False, "excessive_rotation", metrics)

    return QualityDecision(True, "ok", metrics)


def _reprojection_errors(
    src: np.ndarray, dst: np.ndarray, H: np.ndarray
) -> np.ndarray:
    if src.size == 0:
        return np.array([])
    ones = np.ones((src.shape[0], 1), dtype=np.float64)
    src_h = np.hstack([src, ones])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return np.linalg.norm(proj - dst, axis=1)


def _homography_rotation_deg(H: np.ndarray) -> float:
    """Approximate planar rotation from homography (camera looking down)."""
    R_approx = H[:2, :2] / np.linalg.norm(H[:2, 0])
    return float(np.degrees(np.arctan2(R_approx[1, 0], R_approx[0, 0])))
