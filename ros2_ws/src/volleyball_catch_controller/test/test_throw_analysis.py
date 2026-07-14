import unittest

from volleyball_catch_controller.throw_analysis import (
    ObservationSample,
    PredictionSample,
    TrialGroundTruth,
    analyze_session,
    analyze_trial,
    percentile,
    reachable_distance_from_rest,
)
from volleyball_catch_controller.analyze_d435_throw_bag import (
    matched_state_tti,
    nearest_sample,
    same_held_prediction,
)


class ThrowAnalysisTest(unittest.TestCase):
    def test_nearest_sample_can_match_same_cycle_message_published_after_landing(self):
        samples = [(1.90, 'old'), (2.01, 'same-cycle')]
        self.assertEqual(
            nearest_sample(samples, [item[0] for item in samples], 2.0, 0.2)[1],
            'same-cycle',
        )

    def test_state_tti_is_shifted_to_landing_timestamp(self):
        # State at 2.01 says 0.39 s remaining, so at landing message time 2.00
        # the same predicted impact is 0.40 s away.
        self.assertAlmostEqual(matched_state_tti(2.0, (2.01, 0.39, True, 2)), 0.40)
        self.assertIsNone(matched_state_tti(2.0, (2.01, 0.39, False, 2)))

    def test_republished_held_landing_is_deduplicated(self):
        first = PredictionSample(2.00, 1.0, -0.1, 0.40, 0.2)
        repeated = PredictionSample(2.01, 1.0, -0.1, 0.39, 0.19)
        changed = PredictionSample(2.02, 1.01, -0.1, 0.38, 0.18)
        self.assertTrue(same_held_prediction(first, repeated))
        self.assertFalse(same_held_prediction(first, changed))

    def test_percentile_interpolates_and_ignores_non_finite(self):
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, float('nan')], 50), 2.0)
        self.assertIsNone(percentile([], 90))

    def test_reachable_distance_respects_acceleration_speed_and_margin(self):
        self.assertAlmostEqual(reachable_distance_from_rest(0.5, 0.4, 0.8), 0.1)
        self.assertAlmostEqual(reachable_distance_from_rest(1.0, 0.4, 0.8), 0.3)
        self.assertAlmostEqual(
            reachable_distance_from_rest(1.0, 0.4, 0.8, braking_margin_s=0.15),
            0.24,
        )

    def test_trial_report_bins_prediction_error_by_actual_tti(self):
        trial = TrialGroundTruth('t1', 10.0, 11.0, 11.0, 1.0, 0.0)
        observations = [
            ObservationSample(10.0 + index * 0.02, 10.01 + index * 0.02, True, True, True, 0.9)
            for index in range(50)
        ]
        predictions = [
            PredictionSample(10.2, 1.05, 0.0, 0.82, 0.10),
            PredictionSample(10.6, 1.02, 0.0, 0.41, 0.04),
        ]
        report = analyze_trial(
            trial,
            observations,
            predictions,
            {
                'effective_target_radius_m': 0.10,
                'min_first_prediction_lead_s': 0.7,
            },
        )
        self.assertEqual(report['verdict'], 'PASS')
        self.assertAlmostEqual(report['prediction']['landing_error_by_tti_m']['0.80'], 0.05)
        self.assertAlmostEqual(report['prediction']['landing_error_by_tti_m']['0.40'], 0.02)

    def test_missing_truth_produces_incomplete_not_false_pass(self):
        trial = TrialGroundTruth('raw', 1.0, 2.0)
        observations = [ObservationSample(1.5, 1.51, True, True, True)]
        report = analyze_session([trial], observations, [])
        self.assertEqual(report['summary']['verdict'], 'INCOMPLETE')
        self.assertEqual(report['trials'][0]['prediction']['landing_error_p90_m'], None)

    def test_raw_session_idle_frames_do_not_create_false_perception_failure(self):
        trial = TrialGroundTruth('raw', 1.0, 2.0)
        observations = [ObservationSample(1.5, 1.51, False, False, True)]
        report = analyze_trial(trial, observations, [])
        checks = {item['name']: item['status'] for item in report['checks']}
        self.assertEqual(checks['detection_rate'], 'NOT_CONFIGURED')
        self.assertEqual(checks['rgbd_rate'], 'NOT_CONFIGURED')
        self.assertEqual(report['verdict'], 'INCOMPLETE')

    def test_perception_failure_fails_trial(self):
        trial = TrialGroundTruth('bad', 1.0, 2.0, 2.0, 0.0, 0.0)
        observations = [
            ObservationSample(1.0 + index * 0.1, 1.01 + index * 0.1, index == 0, False, False)
            for index in range(5)
        ]
        report = analyze_trial(trial, observations, [], {'effective_target_radius_m': 0.1})
        self.assertEqual(report['verdict'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
