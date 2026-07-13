import math
import unittest

import numpy as np

from volleyball_catch_controller.tracking import BallisticTracker, covariance3


class BallisticTrackerTest(unittest.TestCase):
    def test_requires_configured_measurement_count(self):
        tracker = BallisticTracker(min_updates=3)
        covariance = np.eye(3) * 0.01
        self.assertTrue(tracker.update([0.0, 0.0, 2.0], covariance, 1.00))
        self.assertTrue(tracker.update([0.1, 0.0, 2.1], covariance, 1.05))
        self.assertIsNone(tracker.landing())
        self.assertTrue(tracker.update([0.2, 0.0, 2.15], covariance, 1.10))
        self.assertIsNotNone(tracker.landing())

    def test_rejects_out_of_order_and_large_outlier(self):
        tracker = BallisticTracker(min_updates=2, innovation_gate_chi2=11.34)
        covariance = np.eye(3) * 0.01
        self.assertTrue(tracker.update([0.0, 0.0, 2.0], covariance, 1.00))
        self.assertFalse(tracker.update([0.1, 0.0, 2.0], covariance, 0.99))
        self.assertFalse(tracker.update([100.0, 0.0, 2.0], covariance, 1.05))
        self.assertEqual(tracker.updates, 1)

    def test_predicts_forward_landing(self):
        tracker = BallisticTracker(
            min_updates=3,
            process_accel_mps2=5.0,
            max_predict_time_s=3.0,
        )
        covariance = np.eye(3) * 0.0025
        for index in range(6):
            timestamp = index * 0.02
            position = [
                2.0 * timestamp,
                0.0,
                2.0 + 3.0 * timestamp - 0.5 * 9.81 * timestamp * timestamp,
            ]
            self.assertTrue(tracker.update(position, covariance, timestamp))
        landing, flight_time = tracker.landing()
        self.assertTrue(math.isfinite(flight_time))
        self.assertGreater(flight_time, 0.0)
        self.assertGreater(landing[0], 1.0)
        self.assertAlmostEqual(landing[2], 0.0)

    def test_invalid_covariance_uses_conservative_default(self):
        covariance = covariance3([0.0] * 9, 0.25)
        np.testing.assert_allclose(covariance, np.eye(3) * 0.25)


if __name__ == '__main__':
    unittest.main()
