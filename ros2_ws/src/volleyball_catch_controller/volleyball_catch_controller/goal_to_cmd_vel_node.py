import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from volleyball_interfaces.msg import CatchState

from .goal_control import GoalControlParams, GoalVelocityController


class GoalToCmdVel(Node):
    """Convert fresh D435 landing goals into the chassis visual velocity input."""

    def __init__(self):
        super().__init__('goal_to_cmd_vel')
        defaults = {
            'base_frame': 'base_link',
            'goal_topic': '/auto/goal_pose',
            'goal_valid_topic': '/auto/goal_valid',
            'state_topic': '/catch/state',
            'cmd_vel_topic': '/vision/cmd_vel',
            'publish_rate_hz': 50.0,
            'goal_timeout_s': 0.10,
            'state_timeout_s': 0.25,
            'max_linear_speed_mps': 0.40,
            'max_linear_accel_mps2': 0.80,
            'stop_radius_m': 0.10,
            'braking_margin_s': 0.15,
            'minimum_horizon_s': 0.20,
            'position_kp': 1.0,
            'max_control_dt_s': 0.10,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        rate = float(self.p['publish_rate_hz'])
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError('publish_rate_hz must be positive')

        self.controller = GoalVelocityController(
            GoalControlParams(
                max_linear_speed_mps=float(self.p['max_linear_speed_mps']),
                max_linear_accel_mps2=float(self.p['max_linear_accel_mps2']),
                stop_radius_m=float(self.p['stop_radius_m']),
                braking_margin_s=float(self.p['braking_margin_s']),
                minimum_horizon_s=float(self.p['minimum_horizon_s']),
                position_kp=float(self.p['position_kp']),
                max_control_dt_s=float(self.p['max_control_dt_s']),
            )
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        goal_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            PoseStamped, self.p['goal_topic'], self.on_goal, goal_qos
        )
        self.create_subscription(
            Bool, self.p['goal_valid_topic'], self.on_valid, latched_qos
        )
        self.create_subscription(
            CatchState, self.p['state_topic'], self.on_state, latched_qos
        )
        self.cmd_pub = self.create_publisher(Twist, self.p['cmd_vel_topic'], goal_qos)

        self.goal_xy = None
        self.goal_receive_s = -math.inf
        self.goal_valid = False
        self.state = None
        self.state_receive_s = -math.inf
        self.last_tick_s = self.now_s()
        self.create_timer(1.0 / rate, self.tick)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_goal(self, message):
        x = float(message.pose.position.x)
        y = float(message.pose.position.y)
        if message.header.frame_id != self.p['base_frame'] or not all(
            math.isfinite(value) for value in (x, y)
        ):
            self.goal_xy = None
            return
        self.goal_xy = (x, y)
        self.goal_receive_s = self.now_s()

    def on_valid(self, message):
        self.goal_valid = bool(message.data)

    def on_state(self, message):
        self.state = message
        self.state_receive_s = self.now_s()

    def control_enabled(self, now):
        state = self.state
        goal_age = now - self.goal_receive_s
        state_age = now - self.state_receive_s
        return bool(
            self.goal_valid
            and self.goal_xy is not None
            and 0.0 <= goal_age <= float(self.p['goal_timeout_s'])
            and state is not None
            and 0.0 <= state_age <= float(self.p['state_timeout_s'])
            and int(state.state) == CatchState.NEAR_D435
            and int(state.active_source) == CatchState.SOURCE_D435
            and bool(state.goal_valid)
        )

    def publish_velocity(self, velocity):
        message = Twist()
        message.linear.x = float(velocity[0])
        message.linear.y = float(velocity[1])
        self.cmd_pub.publish(message)

    def tick(self):
        now = self.now_s()
        dt = now - self.last_tick_s
        self.last_tick_s = now
        enabled = self.control_enabled(now)
        time_to_impact = (
            float(self.state.time_to_impact_s) if enabled else math.nan
        )
        velocity = self.controller.step(
            self.goal_xy, time_to_impact, dt, enabled=enabled
        )
        self.publish_velocity(velocity)

    def stop(self):
        self.controller.reset()
        self.publish_velocity((0.0, 0.0))


def main(args=None):
    rclpy.init(args=args)
    node = GoalToCmdVel()
    try:
        rclpy.spin(node)
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
