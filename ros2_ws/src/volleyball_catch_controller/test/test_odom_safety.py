import unittest

from volleyball_catch_controller.odom_safety import OdomContractGuard


class OdomContractGuardTest(unittest.TestCase):
    def sample(self, guard, timestamp, receive=None, frame='odom', child='base_link'):
        return guard.accept(
            timestamp,
            timestamp + 0.005 if receive is None else receive,
            frame,
            child,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )

    def test_accepts_monotonic_chassis_odom_and_measures_rate(self):
        guard = OdomContractGuard()
        for index in range(10):
            self.assertTrue(self.sample(guard, 1.0 + index * 0.02).accepted)
        self.assertAlmostEqual(guard.rate_hz, 50.0)
        self.assertTrue(guard.valid_at(1.2, 0.1))

    def test_rejects_wrong_frames(self):
        guard = OdomContractGuard()
        self.assertFalse(self.sample(guard, 1.0, frame='map').accepted)
        self.assertFalse(self.sample(guard, 1.0, child='chassis').accepted)

    def test_rejects_stale_future_and_non_monotonic_samples(self):
        guard = OdomContractGuard(max_message_age_s=0.05, max_future_skew_s=0.02)
        self.assertFalse(self.sample(guard, 1.0, receive=1.1).accepted)
        self.assertFalse(self.sample(guard, 1.1, receive=1.0).accepted)
        self.assertTrue(self.sample(guard, 2.0).accepted)
        self.assertFalse(self.sample(guard, 2.0).accepted)


if __name__ == '__main__':
    unittest.main()
