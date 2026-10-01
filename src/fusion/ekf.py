from __future__ import annotations

import numpy as np


class AUVExtendedKalmanFilter:
    """
    Flat-pool navigation EKF.

    State (7): [x, y, z, vx, vy, vz, yaw]
    IMU drives prediction; visual (x,y,yaw) and pressure (z) correct.
    """

    STATE_DIM = 7
    IDX = {"x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5, "yaw": 6}

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.x = np.zeros(self.STATE_DIM)
        self.P = np.diag([0.1, 0.1, 0.05, 0.5, 0.5, 0.5, 0.1])
        self._gravity = cfg.get("imu", {}).get("gravity", 9.81)

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def yaw(self) -> float:
        return float(self.x[6])

    def set_initial_state(
        self,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        yaw: float = 0,
    ) -> None:
        self.x[:] = [x, y, z, 0, 0, 0, yaw]

    def predict(self, dt: float, imu: dict) -> None:
        """IMU propagation in pool-fixed frame (x forward, y left, z down)."""
        if dt <= 0:
            return

        yaw = self.x[6]
        c, s = np.cos(yaw), np.sin(yaw)

        # Body-frame accel → world (assume roll/pitch small near level flight)
        ax_b = imu.get("ax", 0.0)
        ay_b = imu.get("ay", 0.0)
        az_b = imu.get("az", 0.0)

        # Remove gravity from z if IMU reports gravity-inclusive accel
        az_world = az_b - self._gravity if imu.get("gravity_inclusive", True) else az_b

        ax = c * ax_b - s * ay_b
        ay = s * ax_b + c * ay_b
        az = az_world

        gx, gy, gz = imu.get("gx", 0.0), imu.get("gy", 0.0), imu.get("gz", 0.0)
        dyaw = gz * dt  # yaw rate from gyro z

        # Kinematic update
        self.x[0] += self.x[3] * dt + 0.5 * ax * dt * dt
        self.x[1] += self.x[4] * dt + 0.5 * ay * dt * dt
        self.x[2] += self.x[5] * dt + 0.5 * az * dt * dt
        self.x[3] += ax * dt
        self.x[4] += ay * dt
        self.x[5] += az * dt
        self.x[6] = _wrap_angle(self.x[6] + dyaw)

        # Jacobian F
        F = np.eye(self.STATE_DIM)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        ekf = self.cfg.get("ekf", {})
        q_a = ekf.get("sigma_accel", 0.5)
        q_g = ekf.get("sigma_gyro", 0.05)
        Q = np.diag(
            [
                0.5 * q_a * dt * dt,
                0.5 * q_a * dt * dt,
                0.5 * q_a * dt * dt,
                q_a * dt,
                q_a * dt,
                q_a * dt,
                q_g * dt,
            ]
        )
        self.P = F @ self.P @ F.T + Q

    def update_visual(self, tx: float, ty: float, dyaw: float, dt: float = 0.0) -> None:
        """
        Integrate relative visual odometry (body frame) into world-frame pose.
        tx, ty: meters in body frame; dyaw: radians.
        """
        ekf = self.cfg.get("ekf", {})
        sx = ekf.get("sigma_visual_xy", 0.05)
        syaw = ekf.get("sigma_visual_yaw", 0.03)

        yaw = self.x[6]
        c, s = np.cos(yaw), np.sin(yaw)
        dx_world = c * tx - s * ty
        dy_world = s * tx + c * ty

        self.x[0] += dx_world
        self.x[1] += dy_world
        self.x[6] = _wrap_angle(self.x[6] + dyaw)
        if dt > 1e-3:
            self.x[3] = dx_world / dt
            self.x[4] = dy_world / dt

        # Relative increments add uncertainty; they must not collapse it.
        self.P[0, 0] += sx * sx
        self.P[1, 1] += sx * sx
        self.P[6, 6] += syaw * syaw

    def update_depth(self, depth_m: float) -> None:
        ekf = self.cfg.get("ekf", {})
        sz = ekf.get("sigma_depth", 0.02) ** 2
        R = np.array([[sz]])

        H = np.zeros((1, self.STATE_DIM))
        H[0, 2] = 1.0
        z = np.array([depth_m])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += (K @ y).ravel()
        self.P = (np.eye(self.STATE_DIM) - K @ H) @ self.P

    def update_yaw(self, yaw_rad: float) -> None:
        """Fuse an absolute heading measurement (for example IMU attitude)."""
        ekf = self.cfg.get("ekf", {})
        variance = ekf.get("sigma_imu_attitude", 0.02) ** 2
        gain = self.P[6, 6] / (self.P[6, 6] + variance)
        innovation = _wrap_angle(yaw_rad - self.x[6])
        self.x[6] = _wrap_angle(self.x[6] + gain * innovation)
        self.P[6, 6] *= 1.0 - gain

    def set_yaw(self, yaw_rad: float) -> None:
        """Set heading from an externally filtered absolute orientation source."""
        self.x[6] = _wrap_angle(yaw_rad)

    def update_attitude(self, roll: float, pitch: float) -> None:
        """Optional: fuse IMU attitude (not in state, but damps bad accel projection)."""
        # For minimal filter we skip roll/pitch in state; hook for future expansion.
        _ = (roll, pitch)


def _wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))
