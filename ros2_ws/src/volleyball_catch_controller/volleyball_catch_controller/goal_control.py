from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GoalControlParams:
    max_linear_speed_mps: float = 0.40
    max_linear_accel_mps2: float = 0.80
    stop_radius_m: float = 0.10
    braking_margin_s: float = 0.15
    minimum_horizon_s: float = 0.20
    position_kp: float = 1.0
    max_control_dt_s: float = 0.10


def _limit_norm(vector, maximum):
    x, y = float(vector[0]), float(vector[1])
    norm = math.hypot(x, y)
    maximum = max(0.0, float(maximum))
    if norm <= maximum or norm <= 1e-12:
        return x, y
    scale = maximum / norm
    return x * scale, y * scale


class GoalVelocityController:
    """Planar goal controller with vector speed and acceleration limiting."""

    def __init__(self, params=None):
        self.params = params or GoalControlParams()
        self.velocity = (0.0, 0.0)

    def reset(self):
        self.velocity = (0.0, 0.0)
        return self.velocity

    def step(self, goal_xy, time_to_impact_s, dt_s, enabled=True):
        if not enabled or goal_xy is None:
            return self.reset()
        try:
            x, y = float(goal_xy[0]), float(goal_xy[1])
            dt_s = float(dt_s)
            time_to_impact_s = float(time_to_impact_s)
        except (IndexError, TypeError, ValueError):
            return self.reset()
        if not all(math.isfinite(value) for value in (x, y, dt_s)) or dt_s <= 0.0:
            return self.reset()
        if math.hypot(x, y) <= max(0.0, self.params.stop_radius_m):
            return self.reset()

        if math.isfinite(time_to_impact_s) and time_to_impact_s > 0.0:
            horizon = max(
                self.params.minimum_horizon_s,
                time_to_impact_s - self.params.braking_margin_s,
            )
            desired = (x / horizon, y / horizon)
        else:
            desired = (self.params.position_kp * x, self.params.position_kp * y)
        desired = _limit_norm(desired, self.params.max_linear_speed_mps)

        dt_s = min(dt_s, max(0.0, self.params.max_control_dt_s))
        maximum_delta = max(0.0, self.params.max_linear_accel_mps2) * dt_s
        delta = (desired[0] - self.velocity[0], desired[1] - self.velocity[1])
        delta = _limit_norm(delta, maximum_delta)
        self.velocity = (
            self.velocity[0] + delta[0],
            self.velocity[1] + delta[1],
        )
        return self.velocity
