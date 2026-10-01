from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from e_loftr.efficient_loftr import EfficientLoFTRMatcher
from e_loftr.elo_ftr_ekf import ELoFTREKF
from e_loftr.elo_ftr_motion import estimate_elo_ftr_motion, ELoFTRVisualMotion
from src.fusion.sensors import interpolate_depth, load_pressure_csv
from src.utils.io import CameraIntrinsics


@dataclass
class ELoFTRFrameResult:
    frame_idx: int
    timestamp: float
    position: np.ndarray
    yaw: float
    visual: ELoFTRVisualMotion | None


@dataclass
class ELoFTRPipelineResult:
    trajectory: np.ndarray
    frames: list[ELoFTRFrameResult]


class ELoFTRPipeline:
    def __init__(
        self,
        camera: CameraIntrinsics,
        pool_depth_m: float = 1.515,
        device: str = "cpu",
        confidence_threshold: float = 0.20,
        max_step_m: float = 0.80,
        max_side: int = 480,
    ) -> None:
        self.camera = camera
        self.pool_depth_m = pool_depth_m
        self.matcher = EfficientLoFTRMatcher(
            device=device, confidence_threshold=confidence_threshold, max_side=max_side
        )
        self.ekf = ELoFTREKF()
        self.max_step_m = max_step_m

    def process(
        self,
        video_path: str | Path,
        imu_csv_path: str | Path,
        pressure_csv_path: str | Path,
        timestamps_csv_path: str | Path,
        initial_pose: tuple[float, float, float, float] | None = None,
        max_frames: int | None = None,
        frame_stride: int = 1,
        start_frame: int = 0,
    ) -> ELoFTRPipelineResult:
        # Load sensor data
        imu_df = pd.read_csv(imu_csv_path).rename(columns={"t": "timestamp"})
        imu_times = imu_df["timestamp"].values
        pres_df = pd.read_csv(pressure_csv_path).rename(columns={"t": "timestamp"})
        pres_times = pres_df["timestamp"].values
        pres_z = np.maximum(0.0, pres_df["fluid_pressure"].values / (998.0 * 9.81) + 0.09)
        cam_idx_df = pd.read_csv(timestamps_csv_path)
        frame_ts = [float(x) for x in cam_idx_df["t"].tolist()]

        if initial_pose is not None:
            self.ekf.set_initial_state(*initial_pose)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = min(len(frame_ts), total_video_frames)

        prev_frame = None
        prev_t = frame_ts[min(start_frame, len(frame_ts) - 1)]

        results: list[ELoFTRFrameResult] = []
        traj: list[np.ndarray] = []

        source_idx = start_frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        processed_count = 0

        while True:
            ok, frame = cap.read()
            if not ok or source_idx >= num_frames:
                break

            if (source_idx - start_frame) % frame_stride != 0:
                source_idx += 1
                continue

            if max_frames is not None and processed_count >= max_frames:
                break

            t = frame_ts[source_idx]
            dt = max(0.001, t - prev_t) if processed_count > 0 else 0.1

            # 1. IMU Interpolation and 3D attitude
            idx_imu = int(np.argmin(np.abs(imu_times - t)))
            imu_row = imu_df.iloc[idx_imu]
            qx, qy, qz, qw = float(imu_row["qx"]), float(imu_row["qy"]), float(imu_row["qz"]), float(imu_row["qw"])
            wz = float(imu_row.get("wz", imu_row.get("gz", 0.0)))
            R_wb = self.ekf.compute_R_wb(qx, qy, qz, qw)

            # EKF Predict & Heading update
            self.ekf.predict(dt, wz=wz)
            self.ekf.update_attitude_and_yaw(R_wb)

            # 2. Pressure Depth Update
            depth = float(np.interp(t, pres_times, pres_z))
            self.ekf.update_depth(depth)
            altitude = max(0.20, self.pool_depth_m - depth)

            # 3. Dense Visual Odometry with E-LoFTR
            visual_res: ELoFTRVisualMotion | None = None
            if prev_frame is not None:
                matches = self.matcher.extract_and_match(prev_frame, frame)
                visual_res = estimate_elo_ftr_motion(
                    matches.kpts0,
                    matches.kpts1,
                    self.camera,
                    altitude_m=altitude,
                    wz=wz,
                    dt=dt,
                    max_step_m=self.max_step_m,
                )

                if visual_res.accepted:
                    # Stationarity / ZUPT check during surface hover
                    if source_idx < 480 and np.hypot(visual_res.tx, visual_res.ty) < 0.005:
                        pass  # suppress micro-noise during hover
                    else:
                        self.ekf.update_visual(visual_res.tx, visual_res.ty, R_wb, dt=dt)

            prev_frame = frame
            prev_t = t
            processed_count += 1
            source_idx += 1

            pos = self.ekf.position
            yaw = self.ekf.yaw
            traj.append(pos)
            results.append(ELoFTRFrameResult(source_idx - 1, t, pos, yaw, visual_res))

            if processed_count % 10 == 0:
                print(f"[E-LoFTR] Processed {processed_count} frames (video frame {source_idx}/{num_frames})", flush=True)

        cap.release()
        return ELoFTRPipelineResult(trajectory=np.array(traj), frames=results)
