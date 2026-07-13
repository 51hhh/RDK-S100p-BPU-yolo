import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from volleyball_interfaces.msg import D435BallObservation


class D435CalibrationCapture(Node):
    """Robustly average a stationary volleyball center for hand-eye calibration."""

    def __init__(self):
        super().__init__('d435_calibration_capture')
        defaults = {
            'topic': '/d435/ball/observation',
            'base_point': [0.0, 0.0, 0.0],
            'sample_count': 60,
            'min_confidence': 0.50,
            'min_depth_samples': 8,
            'max_depth_spread_m': 0.05,
            'output_path': '/tmp/d435_handeye_samples.jsonl',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        base_point = np.asarray(self.p['base_point'], dtype=float)
        if base_point.shape != (3,) or not np.all(np.isfinite(base_point)):
            raise ValueError('base_point must be [x_forward, y_left, z_up] in metres')
        self.base_point = base_point
        self.required_samples = max(10, int(self.p['sample_count']))
        self.samples = []
        self.source_epoch = 0
        self.done = False
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            D435BallObservation, self.p['topic'], self.on_observation, qos
        )
        self.get_logger().info(
            f'collecting {self.required_samples} samples for base point '
            f'{self.base_point.tolist()}'
        )

    def on_observation(self, message):
        if self.done:
            return
        epoch = int(message.source_epoch)
        if self.source_epoch and epoch != self.source_epoch:
            self.samples.clear()
        self.source_epoch = epoch
        point = np.array(
            [message.position.x, message.position.y, message.position.z], dtype=float
        )
        valid = (
            epoch != 0
            and bool(message.detection_valid)
            and bool(message.rgbd_valid)
            and float(message.detection_confidence) >= float(self.p['min_confidence'])
            and int(message.valid_depth_samples) >= int(self.p['min_depth_samples'])
            and math.isfinite(float(message.depth_spread_m))
            and float(message.depth_spread_m) <= float(self.p['max_depth_spread_m'])
            and np.all(np.isfinite(point))
        )
        if not valid:
            return
        self.samples.append(point)
        if len(self.samples) >= self.required_samples:
            self.finish()

    def finish(self):
        points = np.asarray(self.samples, dtype=float)
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        median_distance = float(np.median(distances))
        threshold = max(0.005, 3.0 * 1.4826 * median_distance)
        inliers = points[distances <= threshold]
        if len(inliers) < max(6, self.required_samples // 2):
            self.get_logger().error('too few stable inliers; keep the ball stationary')
            self.samples.clear()
            return
        camera_point = np.mean(inliers, axis=0)
        record = {
            'camera': camera_point.tolist(),
            'base': self.base_point.tolist(),
            'samples': int(len(points)),
            'inliers': int(len(inliers)),
            'spread_m': float(math.sqrt(np.mean(np.sum((inliers - camera_point) ** 2, axis=1)))),
        }
        line = json.dumps(record, ensure_ascii=False)
        output_path = str(self.p['output_path']).strip()
        if output_path:
            with open(output_path, 'a', encoding='utf-8') as stream:
                stream.write(line + '\n')
            self.get_logger().info(f'appended calibration point to {output_path}')
        self.get_logger().info(line)
        self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = D435CalibrationCapture()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
