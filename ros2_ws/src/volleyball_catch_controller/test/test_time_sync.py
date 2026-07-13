import unittest

from volleyball_catch_controller.time_sync import (
    SERVO_LOCKED,
    SyncSample,
    TimeSyncGuard,
    parse_chrony_tracking_csv,
    sequence_is_newer,
)


class TimeSyncGuardTest(unittest.TestCase):
    def sample(self, sequence=1, sample_time=100.001, offset=0.001, epoch=7):
        return SyncSample(
            source_epoch=epoch,
            sequence=sequence,
            sample_time_nx_s=sample_time,
            offset_s=offset,
            uncertainty_s=0.0002,
            synchronized=True,
            servo_state=SERVO_LOCKED,
            clock_source='chrony:test',
        )

    def test_corrects_nx_timestamp_to_rdk_time(self):
        guard = TimeSyncGuard()
        result = guard.ingest(self.sample(), receive_time_s=100.002)
        self.assertTrue(result.accepted)
        corrected, reason = guard.corrected_time(101.001, 7, 100.1)
        self.assertEqual(reason, 'valid')
        self.assertAlmostEqual(corrected, 101.0)

    def test_rejects_replayed_status_after_network_recovery(self):
        guard = TimeSyncGuard()
        self.assertTrue(guard.ingest(self.sample(sequence=10), 100.002).accepted)
        replay = guard.ingest(self.sample(sequence=10), 100.003)
        self.assertFalse(replay.accepted)
        self.assertIn('replayed', replay.reason)

    def test_rejects_old_measurement_even_when_just_received(self):
        guard = TimeSyncGuard(timeout_s=0.5)
        stale = guard.ingest(
            self.sample(sequence=1, sample_time=10.001), receive_time_s=11.0
        )
        self.assertFalse(stale.accepted)

    def test_detects_large_offset_step(self):
        guard = TimeSyncGuard(max_offset_step_s=0.002)
        self.assertTrue(guard.ingest(self.sample(sequence=1), 100.002).accepted)
        stepped = guard.ingest(
            self.sample(sequence=2, sample_time=100.106, offset=0.006), 100.101
        )
        self.assertTrue(stepped.accepted)
        self.assertTrue(stepped.clock_step)

    def test_rejects_retired_epoch_replay(self):
        guard = TimeSyncGuard()
        self.assertTrue(guard.ingest(self.sample(sequence=1, epoch=7), 100.002).accepted)
        self.assertTrue(
            guard.ingest(
                self.sample(sequence=1, epoch=8, sample_time=100.101), 100.102
            ).accepted
        )
        replay = guard.ingest(
            self.sample(sequence=2, epoch=7, sample_time=100.201), 100.202
        )
        self.assertFalse(replay.accepted)
        self.assertIn('retired', replay.reason)

    def test_warn_range_reduces_trust(self):
        guard = TimeSyncGuard(warn_offset_s=0.005, reject_offset_s=0.020)
        sample = self.sample(offset=0.010, sample_time=100.010)
        self.assertTrue(guard.ingest(sample, 100.001).accepted)
        self.assertGreater(guard.trust_scale(), 0.1)
        self.assertLess(guard.trust_scale(), 1.0)

    def test_sequence_wrap(self):
        self.assertTrue(sequence_is_newer(1, 0xFFFFFFFFFFFFFFFF))
        self.assertFalse(sequence_is_newer(5, 5))


class ChronyParserTest(unittest.TestCase):
    def test_parses_locked_tracking_report(self):
        report = (
            'A9FE2A14,2,1783944000.0,0.000400000,-0.000100000,'
            '0.000200000,1.0,0.0,0.1,0.000200000,0.000300000,1.0,Normal\n'
        )
        status = parse_chrony_tracking_csv(report)
        self.assertEqual(status.offset_ns, 400000)
        self.assertEqual(status.uncertainty_ns, 400000)
        self.assertTrue(status.synchronized)
        self.assertEqual(status.servo_state, SERVO_LOCKED)


if __name__ == '__main__':
    unittest.main()
