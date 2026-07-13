import math
import unittest

import numpy as np

from volleyball_catch_controller.tracking import (
    BallisticTracker,
    OdomHistory,
    association_gate,
    covariance3,
    landing_to_catcher_goal,
    quaternion_matrix,
)


class BallisticTrackerTest(unittest.TestCase):
    def test_requires_configured_measurement_count(self):
        tracker = BallisticTracker(min_updates=3)
        covariance = np.eye(3) * 0.01
        self.assertTrue(tracker.update([0.0, 0.0, 2.0], covariance, 1.00))
        self.assertTrue(tracker.update([0.1, 0.0, 2.1], covariance, 1.05))
        self.assertIsNone(tracker.landing())
        self.assertTrue(tracker.update([0.2, 0.0, 2.15], covariance, 1.10))
        self.assertIsNotNone(tracker.landing())

    def test_rejects_out_of_order_and_large_outlier_without_history_pollution(self):
        tracker = BallisticTracker(min_updates=2)
        covariance = np.eye(3) * 0.01
        self.assertTrue(tracker.update([0.0, 0.0, 2.0], covariance, 1.00))
        self.assertFalse(tracker.update([0.1, 0.0, 2.0], covariance, 0.99))
        self.assertEqual(len(tracker.samples), 1)
        self.assertFalse(tracker.update([100.0, 0.0, 2.0], covariance, 1.05))
        self.assertLess(tracker.last_student_w, 0.1)
        self.assertLess(float(tracker.position[0]), 1.0)
        self.assertEqual(tracker.updates, 1)
        self.assertEqual(len(tracker.samples), 1)

    def test_predicts_forward_landing(self):
        tracker = BallisticTracker(
            min_updates=3,
            q_velocity=0.5,
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

    def test_trajectory_ends_at_predicted_landing(self):
        tracker = BallisticTracker(min_updates=3, q_velocity=0.5)
        covariance = np.eye(3) * 0.0025
        for index in range(6):
            timestamp = index * 0.02
            self.assertTrue(
                tracker.update(
                    [1.5 * timestamp, 0.2, 2.0 + 2.5 * timestamp - 4.905 * timestamp**2],
                    covariance,
                    timestamp,
                )
            )
        points, flight_time = tracker.trajectory(step_s=0.01)
        landing, expected_time = tracker.landing()
        self.assertGreater(len(points), 2)
        self.assertAlmostEqual(flight_time, expected_time)
        np.testing.assert_allclose(points[-1], landing)

    def test_trajectory_ends_at_ball_center_contact_plane(self):
        tracker = BallisticTracker(min_updates=3, q_velocity=0.5)
        covariance = np.eye(3) * 0.0025
        for index in range(6):
            timestamp = index * 0.02
            self.assertTrue(
                tracker.update(
                    [1.5 * timestamp, 0.0, 2.0 + 2.5 * timestamp - 4.905 * timestamp**2],
                    covariance,
                    timestamp,
                )
            )
        points, _ = tracker.trajectory(0.1075, 0.01)
        self.assertAlmostEqual(points[-1][2], 0.1075)

    def test_two_trackers_are_independent(self):
        far = BallisticTracker(min_updates=1)
        near = BallisticTracker(min_updates=1)
        covariance = np.eye(3) * 0.01
        self.assertTrue(far.update([5.0, 0.0, 3.0], covariance, 1.0))
        self.assertTrue(near.update([1.0, 0.0, 1.5], covariance, 1.0))
        self.assertNotEqual(float(far.position[0]), float(near.position[0]))

    def test_matches_nx_student_t_drag_reference_sequence(self):
        tracker = BallisticTracker(
            gravity_mps2=9.81,
            drag_coefficient=0.10,
            air_density=1.225,
            ball_mass_kg=0.270,
            ball_radius_m=0.105,
            student_t_nu=12.0,
            q_position=1e-4,
            q_velocity=1.5,
            min_updates=2,
            min_speed_mps=0.2,
            rk4_dt_s=0.008,
        )
        position = np.array([0.0, 0.0, 2.0], dtype=float)
        velocity = np.array([2.2, 0.5, 4.0], dtype=float)
        dt = 0.02
        drag_k = (
            0.5 * 0.10 * 1.225 * math.pi * 0.105 * 0.105 / 0.270
        )
        f_b = 1411.848699
        for index in range(24):
            timestamp = index * dt
            if index:
                acceleration = (
                    np.array([0.0, 0.0, -9.81])
                    - drag_k * np.linalg.norm(velocity) * velocity
                )
                position = position + velocity * dt + 0.5 * acceleration * dt * dt
                velocity = velocity + acceleration * dt
            observation = position + np.array(
                [
                    0.002 * math.sin(index),
                    -0.001 * math.cos(index),
                    0.003 * math.sin(0.7 * index),
                ]
            )
            depth = max(float(observation[2]), 0.5)
            sigma_z = depth * depth / f_b * 0.4
            sigma_xy = max(0.004, 0.0012 * depth)
            covariance = np.diag(
                [sigma_xy * sigma_xy, sigma_xy * sigma_xy, sigma_z * sigma_z]
            )
            self.assertTrue(
                tracker.update(observation, covariance, timestamp)
            )
        trajectory, flight_time = tracker.trajectory(0.0, 0.02)
        np.testing.assert_allclose(
            tracker.position, [1.00433, 0.22932, 2.79295], atol=2e-5
        )
        np.testing.assert_allclose(
            tracker.velocity, [2.11061, 0.50185, -0.62887], atol=2e-5
        )
        np.testing.assert_allclose(
            trajectory[-1], [2.45941, 0.57530, 0.0], atol=4e-4
        )
        self.assertAlmostEqual(flight_time, 0.696476, places=5)

    def test_invalid_covariance_uses_conservative_default(self):
        covariance = covariance3([0.0] * 9, 0.25)
        np.testing.assert_allclose(covariance, np.eye(3) * 0.25)


class OdomHistoryTest(unittest.TestCase):
    def test_nominal_d435_mount_rotation_maps_optical_axes(self):
        rotation = quaternion_matrix(
            [-0.43045933, 0.43045933, -0.56098553, 0.56098553]
        )
        pitch = math.radians(15.0)
        np.testing.assert_allclose(
            rotation @ [0.0, 0.0, 1.0],
            [math.cos(pitch), 0.0, math.sin(pitch)],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            rotation @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-7
        )
        np.testing.assert_allclose(
            rotation @ [0.0, 1.0, 0.0],
            [math.sin(pitch), 0.0, -math.cos(pitch)],
            atol=1e-7,
        )

    def test_landing_goal_accounts_for_catcher_offset(self):
        goal = landing_to_catcher_goal(
            [1.0, 0.0, 0.1075],
            [0.0, 0.0, 0.0],
            0.0,
            [0.2, 0.0, 0.0],
        )
        np.testing.assert_allclose(goal, [0.8, 0.0])

    def test_landing_goal_applies_empirical_plate_correction(self):
        goal = landing_to_catcher_goal(
            [1.0, 0.0, 0.6635],
            [0.0, 0.0, 0.0],
            0.0,
            [0.591, 0.0, 0.556],
            [0.03, -0.02],
        )
        np.testing.assert_allclose(goal, [0.439, -0.02])

    def test_interpolates_pose_and_compensates_camera_observation(self):
        history = OdomHistory(history_s=3.0, max_gap_s=1.1, max_extrapolation_s=0.1)
        self.assertTrue(history.add(1.0, [0.0, 0.0, 0.0], 0.0))
        self.assertTrue(history.add(2.0, [1.0, 0.0, 0.0], math.pi / 2.0))
        position, yaw = history.pose_at(1.5)
        np.testing.assert_allclose(position, [0.5, 0.0, 0.0])
        self.assertAlmostEqual(yaw, math.pi / 4.0)

        transformed, covariance = history.transform_observation(
            [1.0, 0.0, 1.0],
            np.eye(3) * 0.01,
            1.5,
            [0.0, 0.0, 0.0],
            np.eye(3),
        )
        expected = [0.5 + math.sqrt(0.5), math.sqrt(0.5), 1.0]
        np.testing.assert_allclose(transformed, expected)
        np.testing.assert_allclose(covariance, np.eye(3) * 0.01, atol=1e-12)

    def test_short_extrapolation_uses_chassis_velocity(self):
        history = OdomHistory(history_s=3.0, max_gap_s=0.2, max_extrapolation_s=0.05)
        history.add(1.0, [0.0, 0.0, 0.0], 0.0)
        history.add(1.1, [0.2, 0.0, 0.0], 0.1)
        position, yaw = history.pose_at(1.12)
        np.testing.assert_allclose(position, [0.24, 0.0, 0.0])
        self.assertAlmostEqual(yaw, 0.12)
        self.assertIsNone(history.pose_at(1.2))

    def test_full_orientation_compensates_pitch(self):
        history = OdomHistory(history_s=3.0, max_gap_s=0.2, max_extrapolation_s=0.05)
        pitch = math.pi / 2.0
        quaternion = [0.0, math.sin(pitch / 2.0), 0.0, math.cos(pitch / 2.0)]
        self.assertTrue(history.add(1.0, [0.0, 0.0, 0.0], quaternion))
        transformed, _ = history.transform_observation(
            [0.0, 0.0, 1.0], np.eye(3), 1.0, [0.0, 0.0, 0.0], np.eye(3)
        )
        np.testing.assert_allclose(transformed, [1.0, 0.0, 0.0], atol=1e-9)


class AssociationGateTest(unittest.TestCase):
    def test_accepts_same_target_and_rejects_other_ball(self):
        covariance = np.eye(3) * 0.02
        accepted, distance = association_gate(
            [1.0, 2.0, 3.0], covariance, [1.1, 2.0, 3.0], covariance, 11.34
        )
        self.assertTrue(accepted)
        self.assertLess(distance, 11.34)
        rejected, distance = association_gate(
            [1.0, 2.0, 3.0], covariance, [3.0, 2.0, 3.0], covariance, 11.34
        )
        self.assertFalse(rejected)
        self.assertGreater(distance, 11.34)


if __name__ == '__main__':
    unittest.main()
