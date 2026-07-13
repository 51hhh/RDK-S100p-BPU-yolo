import unittest

import numpy as np

from volleyball_catch_controller.extrinsic_calibration import solve_rigid_transform
from volleyball_catch_controller.tracking import quaternion_matrix


class ExtrinsicCalibrationTest(unittest.TestCase):
    def test_recovers_camera_to_base_transform(self):
        quaternion = np.array(
            [-0.43045933, 0.43045933, -0.56098553, 0.56098553], dtype=float
        )
        rotation = quaternion_matrix(quaternion)
        translation = np.array([0.12, -0.03, 1.056], dtype=float)
        camera = np.array(
            [
                [-0.30, -0.10, 1.00],
                [0.25, -0.15, 1.10],
                [-0.20, 0.30, 1.35],
                [0.35, 0.25, 1.50],
                [0.00, -0.35, 1.70],
                [0.10, 0.10, 2.00],
            ],
            dtype=float,
        )
        base = camera @ rotation.T + translation
        result = solve_rigid_transform(camera, base)
        np.testing.assert_allclose(result['rotation'], rotation, atol=1e-9)
        np.testing.assert_allclose(result['translation'], translation, atol=1e-9)
        recovered_rotation = quaternion_matrix(result['quaternion'])
        np.testing.assert_allclose(recovered_rotation, rotation, atol=1e-9)
        self.assertLess(result['rmse_m'], 1e-9)

    def test_rejects_collinear_points(self):
        camera = [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        with self.assertRaisesRegex(ValueError, 'must not be collinear'):
            solve_rigid_transform(camera, camera)


if __name__ == '__main__':
    unittest.main()
