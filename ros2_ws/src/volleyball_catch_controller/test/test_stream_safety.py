import unittest

from volleyball_catch_controller.stream_safety import (
    StreamOrderGuard,
    d435_timing_valid,
)


class StreamRecoveryTest(unittest.TestCase):
    def test_same_epoch_replay_is_rejected_after_disconnect(self):
        guard = StreamOrderGuard()
        self.assertTrue(guard.accept(10, 1.00, 100).accepted)
        self.assertFalse(guard.accept(10, 0.95, 99).accepted)
        self.assertTrue(guard.accept(10, 1.05, 101).accepted)

    def test_restarted_publisher_can_restart_frame_counter(self):
        guard = StreamOrderGuard()
        self.assertTrue(guard.accept(10, 1.00, 100).accepted)
        restarted = guard.accept(11, 1.05, 1)
        self.assertTrue(restarted.accepted)
        self.assertTrue(restarted.new_epoch)
        self.assertFalse(guard.accept(10, 1.06, 101).accepted)


class D435TimingTest(unittest.TestCase):
    def test_accepts_synchronized_rgbd(self):
        self.assertTrue(d435_timing_valid(500000, 100000, True, 2000000, 2000000))

    def test_rejects_bad_mapping_delta_and_uncertainty(self):
        self.assertFalse(d435_timing_valid(0, 0, False, 2000000, 2000000))
        self.assertFalse(d435_timing_valid(3000000, 0, True, 2000000, 2000000))
        self.assertFalse(d435_timing_valid(0, 3000000, True, 2000000, 2000000))


if __name__ == '__main__':
    unittest.main()
