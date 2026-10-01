import numpy as np
from src.fusion.ekf import AUVExtendedKalmanFilter

def test_ekf_predict_and_depth():
    cfg = {"ekf": {"sigma_accel": 0.5, "sigma_gyro": 0.05, "sigma_depth": 0.02}}
    ekf = AUVExtendedKalmanFilter(cfg)
    ekf.set_initial_state(z=0.5)

    ekf.predict(0.1, {"ax": 0.1, "ay": 0.0, "az": 0.0, "gz": 0.0})
    assert ekf.velocity[0] > 0

    ekf.update_depth(0.55)
    assert abs(ekf.position[2] - 0.55) < 0.1


def test_ekf_visual_update():
    cfg = {"ekf": {"sigma_visual_xy": 0.05, "sigma_visual_yaw": 0.03}}
    ekf = AUVExtendedKalmanFilter(cfg)
    ekf.update_visual(0.1, 0.0, 0.01)
    assert ekf.position[0] > 0
