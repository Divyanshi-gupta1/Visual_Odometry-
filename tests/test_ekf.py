import unittest
import numpy as np

from src.fusion.ekf import AUVExtendedKalmanFilter
from e_loftr.elo_ftr_ekf import ELoFTREKF


class TestEKF(unittest.TestCase):
    def test_baseline_ekf_predict_and_depth(self):
        cfg = {"ekf": {"sigma_accel": 0.5, "sigma_gyro": 0.05, "sigma_depth": 0.02}}
        ekf = AUVExtendedKalmanFilter(cfg)
        ekf.set_initial_state(z=0.5)

        ekf.predict(0.1, {"ax": 0.1, "ay": 0.0, "az": 0.0, "gz": 0.0})
        self.assertGreater(ekf.velocity[0], 0)

        ekf.update_depth(0.55)
        self.assertLess(abs(ekf.position[2] - 0.55), 0.1)

    def test_baseline_ekf_visual_update(self):
        cfg = {"ekf": {"sigma_visual_xy": 0.05, "sigma_visual_yaw": 0.03}}
        ekf = AUVExtendedKalmanFilter(cfg)
        ekf.update_visual(0.1, 0.0, 0.01)
        self.assertGreater(ekf.position[0], 0)

    def test_elo_ftr_ekf(self):
        ekf = ELoFTREKF()
        ekf.set_initial_state(x=1.0, y=2.0, z=0.5, yaw=0.0)
        self.assertEqual(ekf.position[0], 1.0)
        self.assertEqual(ekf.position[1], 2.0)
        self.assertEqual(ekf.position[2], 0.5)

        ekf.predict(dt=0.1, wz=0.05)
        self.assertAlmostEqual(ekf.yaw, 0.005, places=3)

        ekf.update_depth(0.60)
        self.assertAlmostEqual(ekf.position[2], 0.60, places=1)


if __name__ == "__main__":
    unittest.main()
