import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .odom_safety import OdomContractGuard


def stamp_s(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class ChassisOdomCheck(Node):
    def __init__(self):
        super().__init__('chassis_odom_check')
        defaults = {
            'topic': '/odom',
            'world_frame': 'odom',
            'base_frame': 'base_link',
            'duration_s': 5.0,
            'min_rate_hz': 20.0,
            'max_message_age_s': 0.05,
            'max_future_skew_s': 0.02,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        self.guard = OdomContractGuard(
            world_frame=self.p['world_frame'],
            base_frame=self.p['base_frame'],
            max_message_age_s=self.p['max_message_age_s'],
            max_future_skew_s=self.p['max_future_skew_s'],
        )
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Odometry, self.p['topic'], self.on_odom, qos)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, message):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        decision = self.guard.accept(
            stamp_s(message.header.stamp),
            self.now_s(),
            message.header.frame_id,
            message.child_frame_id,
            [position.x, position.y, position.z],
            [orientation.x, orientation.y, orientation.z, orientation.w],
        )
        if not decision.accepted and self.guard.rejected <= 5:
            self.get_logger().error(decision.reason)

    def result(self):
        rate = self.guard.rate_hz
        duration = max(0.5, float(self.p['duration_s']))
        minimum_samples = max(10, int(float(self.p['min_rate_hz']) * duration * 0.5))
        valid_now = self.guard.valid_at(
            self.now_s(), max(0.1, 2.0 / max(1.0, float(self.p['min_rate_hz'])))
        )
        passed = bool(
            self.guard.accepted >= minimum_samples
            and self.guard.rejected == 0
            and math.isfinite(rate)
            and rate >= float(self.p['min_rate_hz'])
            and valid_now
        )
        summary = (
            f'topic={self.p["topic"]}, accepted={self.guard.accepted}, '
            f'rejected={self.guard.rejected}, median_rate={rate:.1f} Hz, '
            f'last={self.guard.last_reason}'
        )
        return passed, summary


def main(args=None):
    rclpy.init(args=args)
    node = ChassisOdomCheck()
    started = node.now_s()
    duration = max(0.5, float(node.p['duration_s']))
    try:
        while rclpy.ok() and node.now_s() - started < duration:
            rclpy.spin_once(node, timeout_sec=0.1)
        passed, summary = node.result()
        if passed:
            node.get_logger().info(f'chassis odometry check passed: {summary}')
        else:
            node.get_logger().error(f'chassis odometry check failed: {summary}')
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not passed:
        raise SystemExit(1)
