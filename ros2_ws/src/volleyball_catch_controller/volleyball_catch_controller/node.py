from collections import deque
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from volleyball_interfaces.msg import (
    CatchEvent,
    CatchState,
    D435BallObservation,
    NxBallObservation,
)

from .tracking import BallisticTracker, covariance3, finite_vector


IDLE, FAR_NX, WAIT_D435, NEAR_D435 = 0, 1, 2, 3


def stamp_s(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_of(quaternion):
    values = np.array(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=float
    )
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    x, y, z, w = values / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_matrix(values):
    quaternion = np.asarray(values, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        return None
    norm = np.linalg.norm(quaternion)
    if norm < 1e-9:
        return None
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def valid_bbox(values):
    if not finite_vector(values, 4):
        return False
    x1, y1, x2, y2 = values
    return x2 > x1 and y2 > y1


class CatchController(Node):
    def __init__(self):
        super().__init__('catch_controller')
        defaults = {
            'world_frame': 'odom',
            'base_frame': 'base_link',
            'nx_frame': 'nx_camera_optical_frame',
            'd435_frame': 'camera_color_optical_frame',
            'nx_topic': '/nx/ball/observation',
            'd435_topic': '/d435/ball/observation',
            'odom_topic': '/odom',
            'event_topic': '/catch/event',
            'goal_topic': '/auto/goal_pose',
            'goal_valid_topic': '/auto/goal_valid',
            'landing_topic': '/ball/landing',
            'state_topic': '/catch/state',
            'volleyball_class_id': 0,
            'nx_min_confidence': 0.30,
            'arrival_radius_m': 0.80,
            'd435_confirm_frames': 1,
            'd435_min_confidence': 0.30,
            'nx_lost_timeout_s': 0.30,
            'd435_lost_timeout_s': 0.80,
            'impact_grace_s': 0.15,
            'max_cycle_extension_s': 1.0,
            'max_message_age_s': 0.20,
            'max_future_skew_s': 0.02,
            'odom_history_s': 3.0,
            'odom_timeout_s': 0.10,
            'max_odom_gap_s': 0.05,
            'max_odom_extrapolation_s': 0.02,
            'ground_z_m': 0.0,
            'gravity_mps2': 9.81,
            'tracker_process_accel_mps2': 20.0,
            'tracker_innovation_gate_chi2': 11.34,
            'far_min_updates': 3,
            'near_min_updates': 3,
            'max_predict_time_s': 3.0,
            'default_position_variance_m2': 0.25,
            'catcher_point_base': [0.0, 0.0, 0.0],
            'nx_camera_translation': [0.0, 0.0, 0.0],
            'nx_camera_quaternion': [0.0, 0.0, 0.0, 1.0],
            'nx_extrinsics_calibrated': False,
            'd435_camera_translation': [0.0, 0.0, 0.0],
            'd435_camera_quaternion': [0.0, 0.0, 0.0, 1.0],
            'd435_extrinsics_calibrated': False,
            'require_calibrated_extrinsics': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}

        self.nx_t, self.nx_r, self.nx_extrinsics_ok = self._load_extrinsics('nx')
        self.d435_t, self.d435_r, self.d435_extrinsics_ok = self._load_extrinsics('d435')

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        control_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        event_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            NxBallObservation, self.p['nx_topic'], self.on_nx, sensor_qos
        )
        self.create_subscription(
            D435BallObservation, self.p['d435_topic'], self.on_d435, sensor_qos
        )
        self.create_subscription(
            Odometry,
            self.p['odom_topic'],
            self.on_odom,
            QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(CatchEvent, self.p['event_topic'], self.on_event, event_qos)
        self.goal_pub = self.create_publisher(PoseStamped, self.p['goal_topic'], control_qos)
        self.valid_pub = self.create_publisher(Bool, self.p['goal_valid_topic'], latched_qos)
        self.land_pub = self.create_publisher(PointStamped, self.p['landing_topic'], control_qos)
        self.state_pub = self.create_publisher(CatchState, self.p['state_topic'], latched_qos)

        tracker_args = {
            'gravity_mps2': self.p['gravity_mps2'],
            'process_accel_mps2': self.p['tracker_process_accel_mps2'],
            'innovation_gate_chi2': self.p['tracker_innovation_gate_chi2'],
            'max_predict_time_s': self.p['max_predict_time_s'],
        }
        self.far_tracker = BallisticTracker(
            min_updates=self.p['far_min_updates'], **tracker_args
        )
        self.near_tracker = BallisticTracker(
            min_updates=self.p['near_min_updates'], **tracker_args
        )
        self.odom = deque()
        self.last_odom_receive = 0.0
        self.state = IDLE
        self.source_epoch = 0
        self.catch_id = 0
        self.next_catch_id = 1
        self.nx_track_id = None
        self.arrived = False
        self.d435_confirm = 0
        self.last_nx_receive = 0.0
        self.last_d435_detection = 0.0
        self.last_nx_stamp = -math.inf
        self.last_d435_stamp = -math.inf
        self.last_nx_frame = -1
        self.last_d435_frame = -1
        self.impact_time = 0.0
        self.impact_deadline = 0.0
        self.landing = None
        self.landing_source = 0
        self.distance_to_landing = math.nan
        self.goal_valid = False
        self.reset_reason = CatchState.RESET_NONE
        self.detail = 'startup'
        self.last_state_publish = 0.0
        self.create_timer(0.01, self.tick)
        self.publish_valid(False)
        self.publish_state(force=True)

    def _load_extrinsics(self, prefix):
        translation = np.asarray(self.p[f'{prefix}_camera_translation'], dtype=float)
        rotation = quaternion_matrix(self.p[f'{prefix}_camera_quaternion'])
        explicitly_calibrated = bool(self.p[f'{prefix}_extrinsics_calibrated'])
        valid = translation.shape == (3,) and np.all(np.isfinite(translation)) and rotation is not None
        if self.p['require_calibrated_extrinsics']:
            valid = valid and explicitly_calibrated
        if not valid:
            self.get_logger().error(
                f'{prefix} camera extrinsics are invalid or not marked calibrated; '
                'related control output is inhibited'
            )
            translation = np.zeros(3, dtype=float)
            rotation = np.eye(3, dtype=float)
        return translation, rotation, valid

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def message_fresh(self, timestamp):
        age = self.now_s() - timestamp
        return -self.p['max_future_skew_s'] <= age <= self.p['max_message_age_s']

    def on_odom(self, message):
        timestamp = stamp_s(message.header.stamp)
        if timestamp <= 0.0:
            timestamp = self.now_s()
        position = message.pose.pose.position
        yaw = yaw_of(message.pose.pose.orientation)
        if yaw is None or not finite_vector([position.x, position.y, position.z], 3):
            return
        if self.odom and timestamp <= self.odom[-1][0]:
            return
        self.odom.append(
            (timestamp, np.array([position.x, position.y, position.z], dtype=float), yaw)
        )
        self.last_odom_receive = self.now_s()
        while self.odom and timestamp - self.odom[0][0] > self.p['odom_history_s']:
            self.odom.popleft()

    def odom_at(self, timestamp):
        if not self.odom:
            return None
        if timestamp >= self.odom[-1][0]:
            if timestamp - self.odom[-1][0] <= self.p['max_odom_extrapolation_s']:
                return self.odom[-1][1], self.odom[-1][2]
            return None
        for index in range(1, len(self.odom)):
            before, after = self.odom[index - 1], self.odom[index]
            if before[0] <= timestamp <= after[0]:
                gap = after[0] - before[0]
                if gap <= 0.0 or gap > self.p['max_odom_gap_s']:
                    return None
                ratio = (timestamp - before[0]) / gap
                position = before[1] * (1.0 - ratio) + after[1] * ratio
                yaw = before[2] + wrap(after[2] - before[2]) * ratio
                return position, yaw
        return None

    def transform_observation(self, position, covariance, timestamp, translation, rotation):
        odom_pose = self.odom_at(timestamp)
        if odom_pose is None:
            return None
        base_position, yaw = odom_pose
        cosine, sine = math.cos(yaw), math.sin(yaw)
        yaw_rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        total_rotation = yaw_rotation @ rotation
        point_base = rotation @ np.asarray(position, dtype=float) + translation
        point_odom = base_position + yaw_rotation @ point_base
        covariance_odom = total_rotation @ covariance @ total_rotation.T
        return point_odom, covariance_odom

    def _ordered_message(self, timestamp, frame_id, source):
        if source == 'nx':
            if timestamp <= self.last_nx_stamp or frame_id <= self.last_nx_frame:
                return False
            self.last_nx_stamp, self.last_nx_frame = timestamp, frame_id
        else:
            if timestamp <= self.last_d435_stamp:
                return False
            if frame_id <= self.last_d435_frame and frame_id != 0:
                return False
            self.last_d435_stamp, self.last_d435_frame = timestamp, frame_id
        return True

    def _start_cycle(self, message):
        self.state = FAR_NX
        self.source_epoch = int(message.source_epoch)
        self.nx_track_id = int(message.track_id)
        self.catch_id = self.next_catch_id
        self.next_catch_id = 1 if self.next_catch_id == 0xFFFFFFFF else self.next_catch_id + 1
        self.arrived = False
        self.d435_confirm = 0
        self.impact_time = 0.0
        self.impact_deadline = 0.0
        self.landing = None
        self.landing_source = 0
        self.distance_to_landing = math.nan
        self.reset_reason = CatchState.RESET_NONE
        self.detail = 'new NX observation cycle'
        self.far_tracker.reset()
        self.near_tracker.reset()
        self.last_nx_stamp = -math.inf
        self.last_nx_frame = -1

    def _update_landing(self, tracker, source, timestamp):
        prediction = tracker.landing(self.p['ground_z_m'])
        if prediction is None:
            return
        landing, time_to_land = prediction
        proposed_impact = timestamp + time_to_land
        if not math.isfinite(proposed_impact) or proposed_impact <= self.now_s():
            return
        if self.impact_deadline <= 0.0:
            self.impact_deadline = proposed_impact + self.p['max_cycle_extension_s']
        self.impact_time = min(proposed_impact, self.impact_deadline)
        self.landing = landing
        self.landing_source = source

    def on_nx(self, message):
        timestamp = stamp_s(message.header.stamp)
        frame_id = int(message.frame_id)
        valid = (
            self.message_fresh(timestamp)
            and message.header.frame_id == self.p['nx_frame']
            and int(message.class_id) == self.p['volleyball_class_id']
            and message.detection_valid
            and message.stereo_valid
            and not message.fallback_observation
            and message.detection_confidence >= self.p['nx_min_confidence']
            and valid_bbox(message.bbox_left_xyxy)
            and valid_bbox(message.bbox_right_xyxy)
            and finite_vector([message.position.x, message.position.y, message.position.z], 3)
        )
        if not valid:
            return
        if self.state == NEAR_D435:
            return

        if self.state == IDLE:
            self._start_cycle(message)
        elif int(message.source_epoch) != self.source_epoch:
            self.reset(CatchState.RESET_BALL_LOST, 'NX source epoch changed')
            self._start_cycle(message)
        elif int(message.track_id) != self.nx_track_id:
            return
        if not self._ordered_message(timestamp, frame_id, 'nx'):
            return

        self.last_nx_receive = self.now_s()
        covariance = covariance3(
            message.position_covariance, self.p['default_position_variance_m2']
        )
        transformed = self.transform_observation(
            [message.position.x, message.position.y, message.position.z],
            covariance,
            timestamp,
            self.nx_t,
            self.nx_r,
        )
        if transformed is None:
            self.detail = 'waiting for odom at NX capture time'
            return
        position, covariance_odom = transformed
        if self.far_tracker.update(position, covariance_odom, timestamp):
            self._update_landing(self.far_tracker, CatchState.SOURCE_NX, timestamp)

    def on_d435(self, message):
        timestamp = stamp_s(message.header.stamp)
        frame_id = int(message.frame_id)
        detection_valid = (
            self.message_fresh(timestamp)
            and message.header.frame_id == self.p['d435_frame']
            and int(message.class_id) == self.p['volleyball_class_id']
            and message.detection_valid
            and message.detection_confidence >= self.p['d435_min_confidence']
            and valid_bbox(message.bbox_xyxy)
        )
        if not self._ordered_message(timestamp, frame_id, 'd435'):
            return
        if detection_valid:
            self.last_d435_detection = self.now_s()
            self.d435_confirm += 1
        else:
            self.d435_confirm = 0
            return

        if (
            self.state == WAIT_D435
            and self.arrived
            and self.d435_confirm >= self.p['d435_confirm_frames']
            and self.d435_extrinsics_ok
        ):
            self.state = NEAR_D435
            self.detail = 'D435 detected after base arrival; holding NX landing'
            self.near_tracker.reset()

        if self.state != NEAR_D435 or not message.rgbd_valid:
            return
        position = [message.position.x, message.position.y, message.position.z]
        if not finite_vector(position, 3):
            return
        covariance = covariance3(
            message.position_covariance, self.p['default_position_variance_m2']
        )
        transformed = self.transform_observation(
            position, covariance, timestamp, self.d435_t, self.d435_r
        )
        if transformed is None:
            self.detail = 'waiting for odom at D435 capture time'
            return
        position_odom, covariance_odom = transformed
        if self.near_tracker.update(position_odom, covariance_odom, timestamp):
            if self.near_tracker.ready:
                self.detail = 'D435 near-field trajectory active'
            self._update_landing(self.near_tracker, CatchState.SOURCE_D435, timestamp)

    def on_event(self, message):
        if (
            self.state != IDLE
            and (int(message.source_epoch), int(message.catch_id))
            == (self.source_epoch, self.catch_id)
            and message.event in (message.RETURNED, message.CAUGHT, message.FAILED)
        ):
            self.reset(CatchState.RESET_RETURNED, 'catch mechanism event')

    def publish_valid(self, valid):
        self.goal_valid = bool(valid)
        message = Bool()
        message.data = self.goal_valid
        self.valid_pub.publish(message)

    def reset(self, reason, detail):
        self.publish_valid(False)
        self.state = IDLE
        self.reset_reason = int(reason)
        self.detail = detail
        self.arrived = False
        self.d435_confirm = 0
        self.landing = None
        self.landing_source = 0
        self.distance_to_landing = math.nan
        self.impact_time = 0.0
        self.impact_deadline = 0.0
        self.far_tracker.reset()
        self.near_tracker.reset()
        self.publish_state(force=True)
        self.source_epoch = 0
        self.catch_id = 0
        self.nx_track_id = None

    def current_odom(self):
        now = self.now_s()
        if not self.odom or now - self.last_odom_receive > self.p['odom_timeout_s']:
            return None
        timestamp, position, yaw = self.odom[-1]
        if now - timestamp > self.p['odom_timeout_s']:
            return None
        return position, yaw

    def tick(self):
        now = self.now_s()
        if (
            self.state != IDLE
            and self.impact_time > 0.0
            and now >= self.impact_time + self.p['impact_grace_s']
        ):
            self.reset(CatchState.RESET_IMPACT_TIME, 'impact time reached')
            return
        if (
            self.state in (FAR_NX, WAIT_D435)
            and now - self.last_nx_receive > self.p['nx_lost_timeout_s']
        ):
            self.reset(CatchState.RESET_BALL_LOST, 'NX ball lost')
            return
        if (
            self.state == NEAR_D435
            and now - self.last_d435_detection > self.p['d435_lost_timeout_s']
        ):
            self.reset(CatchState.RESET_BALL_LOST, 'D435 ball lost')
            return

        odom_pose = self.current_odom()
        if self.landing is None or odom_pose is None:
            if self.goal_valid:
                self.publish_valid(False)
            if odom_pose is None and self.state != IDLE:
                self.detail = 'odometry stale; control inhibited'
            self.publish_state()
            return

        base_position, yaw = odom_pose
        catcher_offset = np.asarray(self.p['catcher_point_base'], dtype=float)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        yaw_rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        catcher_position = base_position + yaw_rotation @ catcher_offset
        self.distance_to_landing = float(
            np.linalg.norm((self.landing - catcher_position)[0:2])
        )
        if self.state == FAR_NX and self.distance_to_landing <= self.p['arrival_radius_m']:
            self.arrived = True
            self.state = WAIT_D435
            self.d435_confirm = 0
            self.detail = 'base arrived at NX landing'
        self.publish_outputs(base_position, yaw)
        self.publish_state()

    def publish_outputs(self, base_position, yaw):
        stamp = self.get_clock().now().to_msg()
        landing = PointStamped()
        landing.header.stamp = stamp
        landing.header.frame_id = self.p['world_frame']
        landing.point.x, landing.point.y, landing.point.z = map(float, self.landing)
        self.land_pub.publish(landing)

        required_extrinsics_ok = (
            self.nx_extrinsics_ok
            if self.landing_source == CatchState.SOURCE_NX
            else self.d435_extrinsics_ok
        )
        if self.state == IDLE or not required_extrinsics_ok:
            if self.goal_valid:
                self.publish_valid(False)
            return

        delta = self.landing - base_position
        cosine, sine = math.cos(yaw), math.sin(yaw)
        goal = PoseStamped()
        goal.header.stamp = stamp
        goal.header.frame_id = self.p['base_frame']
        goal.pose.position.x = float(cosine * delta[0] + sine * delta[1])
        goal.pose.position.y = float(-sine * delta[0] + cosine * delta[1])
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        if not self.goal_valid:
            self.publish_valid(True)

    def publish_state(self, force=False):
        now = self.now_s()
        if not force and now - self.last_state_publish < 0.1:
            return
        self.last_state_publish = now
        message = CatchState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.p['world_frame']
        message.source_epoch = self.source_epoch
        message.catch_id = self.catch_id
        message.state = self.state
        message.active_source = self.landing_source
        message.reset_reason = self.reset_reason
        message.base_arrived_latched = self.arrived
        message.near_filter_initialized = self.near_tracker.ready
        message.goal_valid = self.goal_valid
        message.distance_to_landing_m = float(self.distance_to_landing)
        message.time_to_impact_s = (
            max(0.0, self.impact_time - now) if self.impact_time > 0.0 else math.nan
        )
        message.detail = self.detail
        self.state_pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = CatchController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
