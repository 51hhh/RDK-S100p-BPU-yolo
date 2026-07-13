from pathlib import Path
import subprocess
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from volleyball_interfaces.msg import TimeSyncStatus

from .time_sync import (
    SERVO_UNLOCKED,
    parse_chrony_tracking_csv,
)


class NxTimeSyncPublisher(Node):
    """Publish the NX system-clock offset reported by chronyd.

    chronyd may synchronize directly to RDK or consume a PTP PHC refclock. The
    source epoch must be the same value used by the NX observation publisher.
    """

    def __init__(self):
        super().__init__('nx_time_sync_publisher')
        defaults = {
            'topic': '/diagnostics/time_sync',
            'publish_rate_hz': 10.0,
            'source_epoch': 0,
            'source_epoch_file': '/run/volleyball/nx_source_epoch',
            'chronyc_executable': 'chronyc',
            'command_timeout_s': 0.20,
            # chronyc "System time" is local clock minus reference time.
            'offset_sign': 1.0,
            'max_locked_uncertainty_s': 0.002,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            deadline=Duration(seconds=0.5),
        )
        self.publisher = self.create_publisher(
            TimeSyncStatus, self.p['topic'], qos
        )
        # Realtime nanoseconds keep the sequence increasing if only this node
        # restarts while the NX observation source_epoch remains unchanged.
        self.sequence = max(1, time.time_ns() & 0xFFFFFFFFFFFFFFFF)
        self.last_query_error = ''
        self.epoch_error_logged = False
        rate = max(1.0, float(self.p['publish_rate_hz']))
        self.create_timer(1.0 / rate, self.publish_status)

    def _source_epoch(self):
        configured = int(self.p['source_epoch'])
        if configured > 0:
            return configured & 0xFFFFFFFF
        path = Path(str(self.p['source_epoch_file']))
        try:
            value = int(path.read_text(encoding='utf-8').strip(), 0)
        except (OSError, ValueError):
            return 0
        return value & 0xFFFFFFFF

    def _query_chrony(self):
        result = subprocess.run(
            [str(self.p['chronyc_executable']), '-c', 'tracking'],
            check=True,
            capture_output=True,
            text=True,
            timeout=max(0.02, float(self.p['command_timeout_s'])),
        )
        return parse_chrony_tracking_csv(
            result.stdout, offset_sign=float(self.p['offset_sign'])
        )

    def publish_status(self):
        epoch = self._source_epoch()
        sample_time = self.get_clock().now()
        self.sequence = 1 if self.sequence == 0xFFFFFFFFFFFFFFFF else self.sequence + 1
        message = TimeSyncStatus()
        message.header.stamp = sample_time.to_msg()
        message.header.frame_id = 'nx_system_clock'
        message.source_epoch = epoch
        message.sequence = self.sequence
        message.offset_ns = 0
        message.uncertainty_ns = 0xFFFFFFFFFFFFFFFF
        message.synchronized = False
        message.servo_state = SERVO_UNLOCKED
        message.clock_source = 'chrony:unavailable'
        try:
            status = self._query_chrony()
            message.offset_ns = status.offset_ns
            message.uncertainty_ns = status.uncertainty_ns
            message.servo_state = status.servo_state
            message.clock_source = status.clock_source
            message.synchronized = bool(
                epoch != 0
                and status.synchronized
                and status.uncertainty_ns
                <= int(float(self.p['max_locked_uncertainty_s']) * 1e9)
            )
            self.last_query_error = ''
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            detail = str(error)
            if detail != self.last_query_error:
                self.get_logger().error(f'chrony time status unavailable: {detail}')
                self.last_query_error = detail
        if epoch == 0 and not self.epoch_error_logged:
            self.get_logger().error(
                'NX source epoch is zero; configure source_epoch or make the '
                'NX observation process write source_epoch_file'
            )
            self.epoch_error_logged = True
        elif epoch != 0:
            self.epoch_error_logged = False
        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = NxTimeSyncPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
