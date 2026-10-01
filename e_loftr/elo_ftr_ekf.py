from __future__ import annotations

import numpy as np


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class ELoFTREKF:
    """
    7-State Extended Kalman Filter for AUV Navigation:
    State vector: [x, y, z, vx, vy, vz, yaw]
    World Frame: North-East-Down (NED) pool frame.
    """
    STATE_DIM = 7

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or {}
        self.x = np.zeros(self.STATE_DIM)
        self.P = np.diag([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.01])

        # Coordinate transformation matrices between ENU IMU frame and NED navigation frame
        self.R_enu_ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
        self.R_flu_frd = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

    def set_initial_state(self, x: float, y: float, z: float, yaw: float) -> None:
        self.x[0] = x
        self.x[1] = y
        self.x[2] = z
        self.x[6] = _wrap_angle(yaw)

    def compute_R_wb(self, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Compute full 3D attitude rotation matrix from body to NED world frame."""
        Ri = np.array([
            [1.0 - 2.0*qy**2 - 2.0*qz**2, 2.0*qx*qy - 2.0*qz*qw, 2.0*qx*qz + 2.0*qy*qw],
            [2.0*qx*qy + 2.0*qz*qw, 1.0 - 2.0*qx**2 - 2.0*qz**2, 2.0*qy*qz - 2.0*qx*qw],
            [2.0*qx*qz - 2.0*qy*qw, 2.0*qy*qz + 2.0*qx*qw, 1.0 - 2.0*qx**2 - 2.0*qy**2],
        ], dtype=float)
        return self.R_enu_ned @ Ri @ self.R_flu_frd

    def predict(self, dt: float, wz: float = 0.0) -> None:
        if dt <= 0.0:
            return
        # Gyro yaw dead-reckoning between updates
        self.x[6] = _wrap_angle(self.x[6] + wz * dt)

        # Velocity damping due to hydrodynamic drag
        drag_decay = np.exp(-dt / 2.0)
        self.x[3] *= drag_decay
        self.x[4] *= drag_decay
        self.x[5] *= drag_decay

        Q = np.diag([
            0.0001, 0.0001, 0.0001,
            0.001 * dt, 0.001 * dt, 0.001 * dt,
            0.0005 * dt,
        ])
        self.P += Q

    def update_attitude_and_yaw(self, R_wb: np.ndarray) -> None:
        """Update yaw state from full 3D attitude matrix."""
        yaw_ned = float(np.arctan2(R_wb[1, 0], R_wb[0, 0]))
        self.x[6] = yaw_ned

    def update_depth(self, depth_m: float, sigma_depth: float = 0.015) -> None:
        """Kalman depth update from pressure sensor."""
        H = np.zeros((1, self.STATE_DIM))
        H[0, 2] = 1.0
        R = np.array([[sigma_depth ** 2]])
        y = np.array([depth_m]) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += (K @ y).ravel()
        self.P = (np.eye(self.STATE_DIM) - K @ H) @ self.P

    def update_visual(self, dx_b: float, dy_b: float, R_wb: np.ndarray, dt: float = 0.0) -> None:
        """
        Rotate body displacement into NED world frame and integrate into position.
        """
        d_w = R_wb[:2, :2] @ np.array([dx_b, dy_b])
        self.x[0] += float(d_w[0])
        self.x[1] += float(d_w[1])
        if dt > 1e-4:
            self.x[3] = float(d_w[0]) / dt
            self.x[4] = float(d_w[1]) / dt

        self.P[0, 0] += 0.001
        self.P[1, 1] += 0.001

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def yaw(self) -> float:
        return float(self.x[6])
