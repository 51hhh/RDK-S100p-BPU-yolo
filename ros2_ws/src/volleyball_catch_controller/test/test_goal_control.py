import math
import unittest

from volleyball_catch_controller.goal_control import (
    GoalControlParams,
    GoalVelocityController,
)


class GoalVelocityControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = GoalVelocityController(
            GoalControlParams(
                max_linear_speed_mps=0.4,
                max_linear_accel_mps2=0.8,
                stop_radius_m=0.1,
                braking_margin_s=0.15,
                minimum_horizon_s=0.2,
                position_kp=1.0,
                max_control_dt_s=0.1,
            )
        )

    def test_invalid_control_resets_to_zero(self):
        self.controller.step((1.0, 0.0), 1.0, 0.1)
        self.assertEqual(
            self.controller.step((1.0, 0.0), 1.0, 0.1, enabled=False),
            (0.0, 0.0),
        )

    def test_vector_acceleration_is_limited(self):
        velocity = self.controller.step((1.0, 1.0), 1.0, 0.05)
        self.assertAlmostEqual(math.hypot(*velocity), 0.04, places=9)

    def test_vector_speed_is_limited(self):
        velocity = (0.0, 0.0)
        for _ in range(20):
            velocity = self.controller.step((10.0, 10.0), 0.3, 0.1)
        self.assertLessEqual(math.hypot(*velocity), 0.4 + 1e-12)

    def test_stop_radius_publishes_immediate_zero(self):
        self.controller.step((1.0, 0.0), 1.0, 0.1)
        self.assertEqual(self.controller.step((0.05, 0.05), 1.0, 0.1), (0.0, 0.0))

    def test_time_to_impact_sets_required_direction_and_horizon(self):
        controller = GoalVelocityController(
            GoalControlParams(
                max_linear_speed_mps=10.0,
                max_linear_accel_mps2=100.0,
                stop_radius_m=0.0,
                braking_margin_s=0.2,
                minimum_horizon_s=0.1,
                position_kp=1.0,
                max_control_dt_s=1.0,
            )
        )
        velocity = controller.step((0.8, -0.4), 1.0, 1.0)
        self.assertAlmostEqual(velocity[0], 1.0)
        self.assertAlmostEqual(velocity[1], -0.5)

    def test_missing_impact_time_falls_back_to_position_gain(self):
        controller = GoalVelocityController(
            GoalControlParams(
                max_linear_speed_mps=10.0,
                max_linear_accel_mps2=100.0,
                stop_radius_m=0.0,
                position_kp=0.5,
                max_control_dt_s=1.0,
            )
        )
        self.assertEqual(controller.step((0.8, -0.4), math.nan, 1.0), (0.4, -0.2))


if __name__ == '__main__':
    unittest.main()
