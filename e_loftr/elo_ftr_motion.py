from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from src.utils.io import CameraIntrinsics


@dataclass
class ELoFTRVisualMotion:
    accepted: bool
    reason: str
    tx: float          # body forward displacement [meters]
    ty: float          # body rightward displacement [meters]
    dyaw: float        # relative yaw rotation [radians]
    num_matches: int
    num_inliers: int
    inlier_ratio: float


# Calibrated E-LoFTR projection matrix and lever arm
A00, A01 = 0.414558, -1.188827
A10, A11 = 0.998943, 0.062035
LEVER_X, LEVER_Y = -0.005004, -0.508297


def estimate_elo_ftr_motion(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    camera: CameraIntrinsics,
    altitude_m: float,
    wz: float = 0.0,
    dt: float = 0.0,
    min_matches: int = 15,
    min_inlier_ratio: float = 0.25,
    ransac_thresh_px: float = 3.0,
    max_step_m: float = 0.80,
) -> ELoFTRVisualMotion:
    """
    Estimate metric body-frame displacement from dense E-LoFTR correspondences.
    Uses normalized camera coordinates and metric altitude scaling.
    """
    num_matches = len(kpts0)
    if num_matches < min_matches:
        return ELoFTRVisualMotion(False, "too_few_matches", 0.0, 0.0, 0.0, num_matches, 0, 0.0)

    # 1. Normalize keypoint coordinates by camera intrinsics
    p0_n = np.column_stack([(kpts0[:, 0] - camera.cx) / camera.fx, (kpts0[:, 1] - camera.cy) / camera.fy])
    p1_n = np.column_stack([(kpts1[:, 0] - camera.cx) / camera.fx, (kpts1[:, 1] - camera.cy) / camera.fy])

    # 2. Robust 2D partial affine estimation (scale, rotation, translation)
    thresh_norm = ransac_thresh_px / max(camera.fx, camera.fy)
    M, inlier_mask = cv2.estimateAffinePartial2D(
        p0_n, p1_n, cv2.RANSAC, ransacReprojThreshold=thresh_norm
    )

    if M is None or inlier_mask is None:
        return ELoFTRVisualMotion(False, "affine_failed", 0.0, 0.0, 0.0, num_matches, 0, 0.0)

    num_inliers = int(np.sum(inlier_mask))
    inlier_ratio = float(num_inliers) / float(max(1, num_matches))

    if inlier_ratio < min_inlier_ratio or num_inliers < 8:
        return ELoFTRVisualMotion(False, "low_inlier_ratio", 0.0, 0.0, 0.0, num_matches, num_inliers, inlier_ratio)

    tx_norm = float(M[0, 2])
    ty_norm = float(M[1, 2])
    dyaw = float(np.arctan2(M[1, 0], M[0, 0]))

    # 3. Altitude-scaled metric body displacement with lever-arm compensation
    h = max(0.20, float(altitude_m))
    dx_b = A00 * tx_norm * h + A10 * ty_norm * h - LEVER_Y * wz * dt
    dy_b = A01 * tx_norm * h + A11 * ty_norm * h + LEVER_X * wz * dt

    step = float(np.hypot(dx_b, dy_b))
    if max_step_m > 0 and step > max_step_m:
        return ELoFTRVisualMotion(False, "excessive_step", dx_b, dy_b, dyaw, num_matches, num_inliers, inlier_ratio)

    return ELoFTRVisualMotion(
        True, "ok", float(dx_b), float(dy_b), float(dyaw),
        num_matches, num_inliers, inlier_ratio
    )
