import copy

import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from volleyball_interfaces.msg import (
    D435BallObservation,
    NxBallObservation,
    TimeSyncStatus,
    TransportDiagnostics,
)

from .node import CatchController


class RecoverySimulator(Node):
    """Exercise DDS deadline, disconnect, stale replay and epoch recovery locally."""

    def __init__(self):
        super().__init__('transport_recovery_simulator')
        nx_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            deadline=Duration(seconds=0.05),
        )
        d435_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            deadline=Duration(seconds=0.04),
        )
        sync_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            deadline=Duration(seconds=0.5),
        )
        self.nx_pub = self.create_publisher(
            NxBallObservation, '/nx/ball/observation', nx_qos
        )
        self.d435_pub = self.create_publisher(
            D435BallObservation, '/d435/ball/observation', d435_qos
        )
        self.sync_pub = self.create_publisher(
            TimeSyncStatus, '/diagnostics/time_sync', sync_qos
        )
        self.odom_pub = self.create_publisher(
            Odometry,
            '/odom',
            QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(
            TransportDiagnostics,
            '/diagnostics/transport',
            self.on_diagnostics,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.started_s = self.now_s()
        self.frame_id = 0
        self.sync_sequence = 0
        self.epoch = 10
        self.replayed = False
        self.saved_nx = None
        self.saved_sync = None
        self.latest = None
        self.saw_initial_online = False
        self.saw_disconnect = False
        self.done = False
        self.failure = ''
        self.create_timer(0.02, self.tick)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_diagnostics(self, message):
        self.latest = message
        elapsed = self.now_s() - self.started_s
        if elapsed < 0.4 and message.nx_online and message.time_sync_online:
            self.saw_initial_online = True
        if 0.5 < elapsed < 1.1 and not message.nx_online:
            self.saw_disconnect = True

    def publish_odom(self, stamp):
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = 'odom'
        message.child_frame_id = 'base_link'
        message.pose.pose.orientation.w = 1.0
        self.odom_pub.publish(message)

    def publish_d435(self, stamp):
        message = D435BallObservation()
        message.header.stamp = stamp
        message.header.frame_id = 'camera_color_optical_frame'
        message.source_epoch = 20
        message.frame_id = self.frame_id
        message.timestamp_mapping_valid = True
        message.timestamp_domain = message.TIMESTAMP_GLOBAL_TIME
        message.timestamp_uncertainty_ns = 100000
        self.d435_pub.publish(message)

    def nx_message(self, stamp):
        message = NxBallObservation()
        message.header.stamp = stamp
        message.header.frame_id = 'nx_camera_optical_frame'
        message.source_epoch = self.epoch
        message.frame_id = self.frame_id
        message.track_id = 1
        message.class_id = 0
        message.detection_confidence = 0.9
        message.bbox_left_xyxy = [10.0, 10.0, 30.0, 30.0]
        message.bbox_right_xyxy = [8.0, 10.0, 28.0, 30.0]
        message.position.x = 2.0
        message.position.y = 0.0
        message.position.z = 2.0
        message.position_covariance = [
            0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01
        ]
        message.depth_sigma_m = 0.05
        message.match_confidence = 0.9
        message.detection_valid = True
        message.stereo_valid = True
        return message

    def sync_message(self, stamp):
        self.sync_sequence += 1
        message = TimeSyncStatus()
        message.header.stamp = stamp
        message.header.frame_id = 'nx_system_clock'
        message.source_epoch = self.epoch
        message.sequence = self.sync_sequence
        message.offset_ns = 0
        message.uncertainty_ns = 100000
        message.synchronized = True
        message.servo_state = message.SERVO_LOCKED
        message.clock_source = 'recovery-test'
        return message

    def finish(self):
        message = self.latest
        checks = {
            'initial online state was not observed': self.saw_initial_online,
            'NX disconnect was not diagnosed': self.saw_disconnect,
            'final diagnostics missing': message is not None,
            'NX did not recover': bool(message and message.nx_online),
            'time sync did not recover': bool(message and message.time_sync_online),
            'time sync did not relock': bool(message and message.time_sync_valid),
            'NX deadline miss not counted': bool(message and message.nx_deadline_misses),
            'sync deadline miss not counted': bool(
                message and message.time_sync_deadline_misses
            ),
            'stale NX replay not rejected': bool(message and message.nx_rejected),
            'stale sync replay not rejected': bool(
                message and message.time_sync_rejected
            ),
            'NX epoch recovery not counted': bool(message and message.nx_epoch_changes),
        }
        failures = [description for description, passed in checks.items() if not passed]
        self.failure = '; '.join(failures)
        self.done = True

    def tick(self):
        elapsed = self.now_s() - self.started_s
        stamp = self.get_clock().now().to_msg()
        self.frame_id += 1
        self.publish_odom(stamp)
        self.publish_d435(stamp)
        if elapsed < 0.4:
            sync = self.sync_message(stamp)
            nx = self.nx_message(stamp)
            self.sync_pub.publish(sync)
            self.nx_pub.publish(nx)
            if self.saved_nx is None:
                self.saved_nx = copy.deepcopy(nx)
                self.saved_sync = copy.deepcopy(sync)
        elif elapsed < 1.05:
            return
        elif not self.replayed:
            self.sync_pub.publish(self.saved_sync)
            self.nx_pub.publish(self.saved_nx)
            self.replayed = True
        elif elapsed < 2.0:
            if self.epoch == 10:
                self.epoch = 11
                self.sync_sequence = 0
            self.sync_pub.publish(self.sync_message(stamp))
            self.nx_pub.publish(self.nx_message(stamp))
        else:
            self.finish()


def main(args=None):
    rclpy.init(args=args)
    controller = CatchController()
    simulator = RecoverySimulator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(controller)
    executor.add_node(simulator)
    try:
        while rclpy.ok() and not simulator.done:
            executor.spin_once(timeout_sec=0.1)
        if simulator.failure:
            raise RuntimeError(simulator.failure)
        simulator.get_logger().info('DDS disconnect/replay recovery test passed')
    finally:
        executor.shutdown()
        simulator.destroy_node()
        controller.destroy_node()
        rclpy.shutdown()
