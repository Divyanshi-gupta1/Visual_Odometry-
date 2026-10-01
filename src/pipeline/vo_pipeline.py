from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.features.superpoint_lightglue import FeatureMatcher
from src.fusion.ekf import AUVExtendedKalmanFilter
from src.fusion.sensors import (
    IMUSample,
    PressureSample,
    interpolate_depth,
    interpolate_imu,
)
from src.motion.ransac_motion import VisualMotion, estimate_planar_motion
from src.utils.io import CameraIntrinsics


@dataclass
class FrameResult:
    frame_idx: int
    timestamp: float
    visual: VisualMotion | None
    position: np.ndarray
    velocity: np.ndarray
    yaw: float
    depth: float | None = None


@dataclass
class VOPipelineResult:
    frames: list[FrameResult] = field(default_factory=list)
    trajectory: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


class VOPipeline:
    """End-to-end: feature match → RANSAC → quality gate → EKF fusion."""

    def __init__(self, config: dict, camera: CameraIntrinsics) -> None:
        self.config = config
        self.camera = camera
        feat = config.get("features", {})
        self.matcher = FeatureMatcher(
            backend=feat.get("backend", "auto"),
            max_keypoints=int(feat.get("max_keypoints", 512)),
            keypoint_threshold=float(feat.get("keypoint_threshold", 0.005)),
            match_threshold=float(feat.get("match_threshold", 0.2)),
        )
        self.ekf = AUVExtendedKalmanFilter(config)
        qg = config.get("quality_gate", {})
        self.quality_cfg = {**feat, **qg}
        self.tile_spacing = float(config.get("tiles", {}).get("spacing_m", 0.30))
        self.tile_spacing_px = float(config.get("tiles", {}).get("spacing_px", 0))
        self.max_step_m = float(config.get("elo_ftr", {}).get("max_step_m", 0.0))
        if self.max_step_m <= 0:
            self.max_step_m = float(config.get("visual", {}).get("max_step_m", 0.80))
        self.fallback_matcher = FeatureMatcher(backend="tile_corners")

    def process_video(
        self,
        video_path: str | Path,
        imu_samples: list[IMUSample] | None = None,
        heading_samples: list[IMUSample] | None = None,
        pressure_samples: list[PressureSample] | None = None,
        max_frames: int | None = None,
        frame_stride: int = 1,
        start_frame: int = 0,
        frame_timestamps: Sequence[float] | None = None,
        max_visual_reference_s: float = 0.0,
        initial_pose: tuple[float, float, float, float] | None = None,
    ) -> VOPipelineResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if start_frame < 0:
            raise ValueError("start_frame must be zero or positive")
        if max_visual_reference_s < 0:
            raise ValueError("max_visual_reference_s must be zero or positive")

        timestamps_arr: np.ndarray | None = None
        if frame_timestamps is not None:
            timestamps_arr = np.asarray(frame_timestamps, dtype=float)
            if len(timestamps_arr) == 0:
                raise ValueError("frame_timestamps must not be empty")
            if not np.all(np.isfinite(timestamps_arr)):
                raise ValueError("frame_timestamps must contain only finite values")
            if np.any(np.diff(timestamps_arr) <= 0):
                raise ValueError("frame_timestamps must be strictly increasing")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dt_default = 1.0 / fps

        if initial_pose is not None:
            self.ekf.set_initial_state(
                x=initial_pose[0],
                y=initial_pose[1],
                z=initial_pose[2],
                yaw=initial_pose[3],
            )

        # Hold the last frame that produced an accepted visual displacement.
        # If a short-baseline match is rejected, replacing the reference would
        # permanently discard the vehicle motion during that interval.
        visual_reference_frame: np.ndarray | None = None
        visual_reference_time = 0.0
        prev_time = 0.0
        initial_heading_yaw: float | None = None
        use_heading_imu = bool(heading_samples)
        results: list[FrameResult] = []
        traj: list[np.ndarray] = []
        prior_tx = 0.0
        prior_ty = 0.0
        yaw0 = float(self.ekf.yaw)

        source_idx = 0
        processed_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if source_idx < start_frame:
                source_idx += 1
                continue
            if (source_idx - start_frame) % frame_stride != 0:
                source_idx += 1
                continue
            if max_frames is not None and processed_idx >= max_frames:
                break

            if timestamps_arr is None:
                timestamp = source_idx * dt_default
                dt = dt_default if processed_idx == 0 else timestamp - prev_time
            else:
                if source_idx >= len(timestamps_arr):
                    raise ValueError(
                        "Video has more frames than the supplied timestamp index "
                        f"({source_idx + 1} > {len(timestamps_arr)})."
                    )
                timestamp = float(timestamps_arr[source_idx])
                dt = 0.0 if processed_idx == 0 else timestamp - prev_time

            imu = interpolate_imu(imu_samples or [], timestamp)
            if imu is not None:
                self.ekf.predict(
                    dt,
                    {
                        "ax": imu.ax,
                        "ay": imu.ay,
                        "az": imu.az,
                        "gx": imu.gx,
                        "gy": imu.gy,
                        "gz": imu.gz,
                        # Heading IMU is authoritative; do not double-integrate yaw
                        # from gyro when absolute heading is available separately.
                        "gravity_inclusive": True,
                    },
                )

            heading = interpolate_imu(heading_samples or [], timestamp)
            if heading is not None and heading.yaw is not None:
                if initial_heading_yaw is None:
                    initial_heading_yaw = heading.yaw
                yaw_sign = float(self.config.get("imu", {}).get("heading_yaw_sign", 1.0))
                self.ekf.set_yaw(
                    yaw0 + yaw_sign * (heading.yaw - initial_heading_yaw)
                )

            depth = interpolate_depth(pressure_samples or [], timestamp)
            if depth is not None:
                self.ekf.update_depth(depth)

            visual: VisualMotion | None = None
            if visual_reference_frame is None:
                visual_reference_frame = frame.copy()
                visual_reference_time = timestamp
            else:
                if processed_idx % 25 == 0:
                    print(
                        f"[VO] frame {source_idx} processed {processed_idx} "
                        f"t={timestamp:.3f}",
                        flush=True,
                    )
                matches = self.matcher.extract_and_match(visual_reference_frame, frame)
                # Prior in image axes (before down-camera remap).
                if self.config.get("visual", {}).get("down_camera_axes", False):
                    # Body prior (x forward, y right) → image (x right, y down/forward)
                    sx = float(self.config["visual"].get("image_y_to_body_x_sign", 1.0))
                    sy = float(self.config["visual"].get("image_x_to_body_y_sign", 1.0))
                    prior_img_x = sy * prior_ty
                    prior_img_y = sx * prior_tx
                else:
                    prior_img_x, prior_img_y = prior_tx, prior_ty
                visual = estimate_planar_motion(
                    matches.kpts0,
                    matches.kpts1,
                    self.camera,
                    self.tile_spacing,
                    self.quality_cfg,
                    tile_spacing_px=self.tile_spacing_px,
                    dt=dt,
                    max_step_m=self.max_step_m,
                    prior_tx=prior_img_x,
                    prior_ty=prior_img_y,
                )
                if (
                    not visual.accepted
                    and getattr(self, "fallback_matcher", None) is not None
                ):
                    fb = self.fallback_matcher.extract_and_match(
                        visual_reference_frame, frame
                    )
                    fallback = estimate_planar_motion(
                        fb.kpts0,
                        fb.kpts1,
                        self.camera,
                        self.tile_spacing,
                        self.quality_cfg,
                        tile_spacing_px=self.tile_spacing_px,
                        dt=dt,
                        max_step_m=self.max_step_m,
                        prior_tx=prior_img_x,
                        prior_ty=prior_img_y,
                    )
                    if fallback.accepted:
                        visual = fallback
                        visual.reason = "ok_lk_fallback"
                if visual.accepted:
                    # Image coordinates are right/down. For a downward-looking
                    # camera, image-y is body-forward and image-x is body-right.
                    if self.config.get("visual", {}).get("down_camera_axes", False):
                        image_x, image_y = visual.tx, visual.ty
                        visual.tx = float(
                            self.config["visual"].get("image_y_to_body_x_sign", 1.0)
                        ) * image_y
                        visual.ty = float(
                            self.config["visual"].get("image_x_to_body_y_sign", 1.0)
                        ) * image_x
                    # Absolute IMU heading already sets yaw; do not also integrate
                    # noisy visual dyaw on a repetitive tile floor.
                    if self.config.get("visual", {}).get("prefer_forward", True):
                        visual.tx = abs(float(visual.tx))
                    dyaw = 0.0 if use_heading_imu else visual.dyaw
                    # After forcing forward, do not reject as reversed.
                    if visual.accepted:
                        self.ekf.update_visual(visual.tx, visual.ty, dyaw, dt=dt)
                        prior_tx = visual.tx
                        prior_ty = visual.ty
                        visual_reference_frame = frame.copy()
                        visual_reference_time = timestamp
                if not visual.accepted:
                    if (
                        max_visual_reference_s > 0
                        and timestamp - visual_reference_time >= max_visual_reference_s
                    ):
                        visual_reference_frame = frame.copy()
                        visual_reference_time = timestamp
                    elif max_visual_reference_s == 0:
                        visual_reference_frame = frame.copy()
                        visual_reference_time = timestamp

            pos = self.ekf.position
            vel = self.ekf.velocity
            results.append(
                FrameResult(
                    frame_idx=source_idx,
                    timestamp=timestamp,
                    visual=visual,
                    position=pos,
                    velocity=vel,
                    yaw=self.ekf.yaw,
                    depth=depth,
                )
            )
            traj.append(pos.copy())

            prev_time = timestamp
            source_idx += 1
            processed_idx += 1

        cap.release()
        return VOPipelineResult(
            frames=results,
            trajectory=np.array(traj) if traj else np.zeros((0, 3)),
        )

    def process_image_pair(
        self, img0: np.ndarray, img1: np.ndarray
    ) -> tuple[VisualMotion, object]:
        matches = self.matcher.extract_and_match(img0, img1)
        motion = estimate_planar_motion(
            matches.kpts0,
            matches.kpts1,
            self.camera,
            self.tile_spacing,
            self.quality_cfg,
            tile_spacing_px=self.tile_spacing_px,
            max_step_m=self.max_step_m,
        )
        return motion, matches
