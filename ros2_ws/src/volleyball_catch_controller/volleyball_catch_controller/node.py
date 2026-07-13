import math
import secrets
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.event_handler import SubscriptionEventCallbacks
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from volleyball_interfaces.msg import (
    CatchEvent,
    CatchState,
    D435BallObservation,
    NxBallObservation,
    TimeSyncStatus,
    TransportDiagnostics,
)

from .stream_safety import StreamOrderGuard, d435_timing_valid
from .time_sync import SyncSample, TimeSyncGuard
from .odom_safety import OdomContractGuard

from .tracking import (
    BallisticTracker,
    OdomHistory,
    association_gate,
    catcher_geometry_from_camera,
    covariance3,
    finite_vector,
    landing_to_catcher_goal,
)


IDLE, FAR_NX, WAIT_D435, NEAR_D435 = 0, 1, 2, 3


def stamp_s(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


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
            'sensor_mode': 'joint',
            'world_frame': 'odom',
            'base_frame': 'base_link',
            'nx_frame': 'nx_left_rectified_optical_frame',
            'd435_frame': 'camera_color_optical_frame',
            'nx_topic': '/nx/ball/observation',
            'd435_topic': '/d435/ball/observation',
            'time_sync_topic': '/diagnostics/time_sync',
            'transport_diagnostics_topic': '/diagnostics/transport',
            'odom_topic': '/odom',
            'event_topic': '/catch/event',
            'goal_topic': '/auto/goal_pose',
            'goal_valid_topic': '/auto/goal_valid',
            'landing_topic': '/ball/landing',
            'filtered_position_topic': '/ball/filtered_position',
            'filtered_velocity_topic': '/ball/filtered_velocity',
            'predicted_path_topic': '/ball/predicted_path',
            'state_topic': '/catch/state',
            'volleyball_class_id': 0,
            'nx_min_confidence': 0.30,
            'arrival_radius_m': 0.80,
            'd435_confirm_frames': 3,
            'd435_min_confidence': 0.30,
            'nx_lost_timeout_s': 0.30,
            'd435_lost_timeout_s': 0.80,
            'impact_grace_s': 0.15,
            'max_cycle_extension_s': 1.0,
            'nx_max_message_age_s': 0.05,
            'd435_max_message_age_s': 0.05,
            'max_future_skew_s': 0.02,
            'require_nx_time_sync': True,
            'time_sync_timeout_s': 0.50,
            'time_sync_warn_offset_s': 0.005,
            'time_sync_reject_offset_s': 0.020,
            'time_sync_max_uncertainty_s': 0.002,
            'time_sync_max_offset_step_s': 0.002,
            'nx_max_stereo_delta_ns': 1000000,
            'nx_max_depth_sigma_m': 0.75,
            'd435_max_rgb_depth_delta_ns': 2000000,
            'd435_max_timestamp_uncertainty_s': 0.002,
            'nx_deadline_s': 0.050,
            'd435_deadline_s': 0.040,
            'time_sync_deadline_s': 0.500,
            'transport_diagnostics_rate_hz': 5.0,
            'odom_history_s': 3.0,
            'odom_timeout_s': 0.10,
            'odom_max_message_age_s': 0.05,
            'max_odom_gap_s': 0.05,
            'max_odom_extrapolation_s': 0.02,
            'ground_z_m': 0.0,
            'intercept_plane_z_m': 0.1075,
            'gravity_mps2': 9.81,
            'drag_coefficient': 0.10,
            'air_density': 1.225,
            'ball_mass_kg': 0.270,
            'ball_radius_m': 0.105,
            'student_t_nu': 12.0,
            'filter_q_position': 0.0001,
            'filter_q_velocity': 1.5,
            'filter_innovation_gate_chi2': 25.0,
            'filter_max_dt_s': 0.5,
            'filter_min_speed_mps': 0.5,
            'far_min_updates': 2,
            'near_min_updates': 2,
            'prediction_min_confidence': 0.70,
            'prediction_min_speed_mps': 0.80,
            'prediction_min_student_weight': 0.15,
            'prediction_min_time_to_land_s': 0.25,
            'prediction_max_time_to_land_s': 2.20,
            'prediction_stable_frames': 3,
            'prediction_max_stable_interval_s': 0.15,
            'prediction_max_stable_jump_m': 0.35,
            'prediction_allow_polynomial': False,
            'handover_position_gate_chi2': 11.34,
            'handover_max_landing_delta_m': 0.75,
            'handover_min_rgbd_frames': 3,
            'max_predict_time_s': 3.0,
            'rk4_dt_s': 0.008,
            'trajectory_step_s': 0.02,
            'poly_min_frames': 6,
            'trajectory_history_max': 20,
            'diagnostics_enabled': True,
            'diagnostics_rate_hz': 15.0,
            'default_position_variance_m2': 0.25,
            'catcher_point_base': [0.0, 0.0, 0.0],
            'use_d435_catcher_geometry': False,
            'd435_catcher_offset_base': [0.0, 0.0, 0.0],
            'landing_goal_correction_base': [0.0, 0.0],
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
        self.sensor_mode = str(self.p['sensor_mode']).strip().lower()
        if self.sensor_mode not in ('joint', 'd435_only'):
            raise ValueError(
                f"sensor_mode must be 'joint' or 'd435_only', got {self.sensor_mode!r}"
            )
        self.d435_only = self.sensor_mode == 'd435_only'

        self.nx_t, self.nx_r, self.nx_extrinsics_ok = self._load_extrinsics('nx')
        self.d435_t, self.d435_r, self.d435_extrinsics_ok = self._load_extrinsics('d435')
        self.catcher_point_base = np.asarray(
            self.p['catcher_point_base'], dtype=float
        )
        if bool(self.p['use_d435_catcher_geometry']):
            geometry = catcher_geometry_from_camera(
                self.d435_t,
                self.p['d435_catcher_offset_base'],
                self.p['ball_radius_m'],
            )
            if geometry is None:
                raise ValueError('D435-to-catcher geometry is invalid')
            self.catcher_point_base, intercept_height = geometry
            self.p['intercept_plane_z_m'] = intercept_height
        if (
            self.catcher_point_base.shape != (3,)
            or not np.all(np.isfinite(self.catcher_point_base))
        ):
            raise ValueError('catcher_point_base must contain three finite values')

        self.sensor_group = ReentrantCallbackGroup()
        self.odom_group = ReentrantCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.pending_lock = threading.Lock()
        self.pending_nx = None
        self.pending_d435 = None
        self.pending_time_sync = None
        self.pending_events = []

        self.transport = {
            source: {
                'messages': 0,
                'deadline_misses': 0,
                'rejected': 0,
                'epoch_changes': 0,
                'last_receive': 0.0,
            }
            for source in ('nx', 'd435', 'time_sync', 'odom')
        }

        nx_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            deadline=Duration(seconds=float(self.p['nx_deadline_s'])),
        )
        d435_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            deadline=Duration(seconds=float(self.p['d435_deadline_s'])),
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
        time_sync_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            deadline=Duration(seconds=float(self.p['time_sync_deadline_s'])),
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        diagnostic_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.nx_subscription = None
        self.time_sync_subscription = None
        if not self.d435_only:
            self.nx_subscription = self.create_subscription(
                NxBallObservation, self.p['nx_topic'], self.on_nx, nx_qos,
                callback_group=self.sensor_group,
                event_callbacks=self._subscription_events('nx'),
            )
        self.create_subscription(
            D435BallObservation, self.p['d435_topic'], self.on_d435, d435_qos,
            callback_group=self.sensor_group,
            event_callbacks=self._subscription_events('d435'),
        )
        if not self.d435_only:
            self.time_sync_subscription = self.create_subscription(
                TimeSyncStatus, self.p['time_sync_topic'], self.on_time_sync, time_sync_qos,
                callback_group=self.sensor_group,
                event_callbacks=self._subscription_events('time_sync'),
            )
        self.create_subscription(
            Odometry,
            self.p['odom_topic'],
            self.on_odom,
            QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT),
            callback_group=self.odom_group,
        )
        self.create_subscription(
            CatchEvent, self.p['event_topic'], self.on_event, event_qos,
            callback_group=self.sensor_group,
        )
        self.goal_pub = self.create_publisher(PoseStamped, self.p['goal_topic'], control_qos)
        self.valid_pub = self.create_publisher(Bool, self.p['goal_valid_topic'], latched_qos)
        self.land_pub = self.create_publisher(PointStamped, self.p['landing_topic'], control_qos)
        self.filtered_position_pub = self.create_publisher(
            PointStamped, self.p['filtered_position_topic'], diagnostic_qos
        )
        self.filtered_velocity_pub = self.create_publisher(
            Vector3Stamped, self.p['filtered_velocity_topic'], diagnostic_qos
        )
        self.predicted_path_pub = self.create_publisher(
            Path, self.p['predicted_path_topic'], diagnostic_qos
        )
        self.state_pub = self.create_publisher(CatchState, self.p['state_topic'], latched_qos)
        self.transport_pub = self.create_publisher(
            TransportDiagnostics, self.p['transport_diagnostics_topic'], diagnostic_qos
        )

        common_tracker_args = {
            'gravity_mps2': self.p['gravity_mps2'],
            'drag_coefficient': self.p['drag_coefficient'],
            'air_density': self.p['air_density'],
            'ball_mass_kg': self.p['ball_mass_kg'],
            'ball_radius_m': self.p['ball_radius_m'],
            'student_t_nu': self.p['student_t_nu'],
            'q_position': self.p['filter_q_position'],
            'q_velocity': self.p['filter_q_velocity'],
            'innovation_gate_chi2': self.p['filter_innovation_gate_chi2'],
            'max_dt_s': self.p['filter_max_dt_s'],
            'min_speed_mps': self.p['filter_min_speed_mps'],
            'max_predict_time_s': self.p['max_predict_time_s'],
            'rk4_dt_s': self.p['rk4_dt_s'],
            'poly_min_frames': self.p['poly_min_frames'],
            'history_max': self.p['trajectory_history_max'],
        }
        self.far_tracker = BallisticTracker(
            min_updates=self.p['far_min_updates'],
            **common_tracker_args,
        )
        self.near_tracker = BallisticTracker(
            min_updates=self.p['near_min_updates'],
            **common_tracker_args,
        )
        self.odom = OdomHistory(
            history_s=self.p['odom_history_s'],
            max_gap_s=self.p['max_odom_gap_s'],
            max_extrapolation_s=self.p['max_odom_extrapolation_s'],
        )
        self.odom_guard = OdomContractGuard(
            world_frame=self.p['world_frame'],
            base_frame=self.p['base_frame'],
            max_message_age_s=self.p['odom_max_message_age_s'],
            max_future_skew_s=self.p['max_future_skew_s'],
        )
        self.last_odom_receive = 0.0
        self.state = IDLE
        self.source_epoch = 0
        self.catch_id = 0
        self.next_catch_id = secrets.randbelow(0xFFFFFFFE) + 1
        self.nx_track_id = None
        self.arrived = False
        self.d435_confirm = 0
        self.last_nx_receive = 0.0
        self.last_d435_detection = 0.0
        self.nx_order = StreamOrderGuard()
        self.d435_order = StreamOrderGuard()
        self.last_d435_epoch = 0
        self.time_sync_guard = TimeSyncGuard(
            timeout_s=self.p['time_sync_timeout_s'],
            max_future_skew_s=self.p['max_future_skew_s'],
            warn_offset_s=self.p['time_sync_warn_offset_s'],
            reject_offset_s=self.p['time_sync_reject_offset_s'],
            max_uncertainty_s=self.p['time_sync_max_uncertainty_s'],
            max_offset_step_s=self.p['time_sync_max_offset_step_s'],
        )
        self.last_time_sync_warning_s = 0.0
        self.impact_time = 0.0
        self.impact_deadline = 0.0
        self.landing = None
        self.filtered_position = None
        self.filtered_velocity = None
        self.predicted_trajectory = []
        self.landing_stable_frames = {
            CatchState.SOURCE_NX: 0,
            CatchState.SOURCE_D435: 0,
        }
        self.last_landing_candidate = {
            CatchState.SOURCE_NX: None,
            CatchState.SOURCE_D435: None,
        }
        self.last_landing_candidate_time = {
            CatchState.SOURCE_NX: None,
            CatchState.SOURCE_D435: None,
        }
        self.near_rgbd_frames = 0
        self.landing_source = 0
        self.distance_to_landing = math.nan
        self.goal_valid = False
        self.reset_reason = CatchState.RESET_NONE
        self.detail = 'startup'
        self.last_state_publish = 0.0
        self.last_diagnostic_publish = 0.0
        self.last_transport_publish = 0.0
        self.last_d435_rgbd = 0.0
        self.create_timer(0.01, self.tick, callback_group=self.control_group)
        self.publish_valid(False)
        self.publish_state(force=True)

    def _load_extrinsics(self, prefix):
        translation = np.asarray(self.p[f'{prefix}_camera_translation'], dtype=float)
        rotation = quaternion_matrix(self.p[f'{prefix}_camera_quaternion'])
        explicitly_calibrated = bool(self.p[f'{prefix}_extrinsics_calibrated'])
        geometry_valid = (
            translation.shape == (3,)
            and np.all(np.isfinite(translation))
            and rotation is not None
        )
        if not geometry_valid:
            self.get_logger().error(
                f'{prefix} camera extrinsics are mathematically invalid; '
                'identity is used for diagnostics and control is inhibited'
            )
            translation = np.zeros(3, dtype=float)
            rotation = np.eye(3, dtype=float)
        control_valid = geometry_valid and (
            explicitly_calibrated or not self.p['require_calibrated_extrinsics']
        )
        if geometry_valid and not control_valid:
            self.get_logger().error(
                f'{prefix} camera extrinsics are not marked calibrated; '
                'configured geometry is used for diagnostics but control is inhibited'
            )
        return translation, rotation, control_valid

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _subscription_events(self, source):
        return SubscriptionEventCallbacks(
            deadline=lambda event: self._on_deadline_missed(source, event),
        )

    def _on_deadline_missed(self, source, event):
        change = max(1, int(getattr(event, 'total_count_change', 1)))
        with self.pending_lock:
            self.transport[source]['deadline_misses'] += change

    def _record_receive(self, source):
        self.transport[source]['messages'] += 1
        self.transport[source]['last_receive'] = self.now_s()

    def _record_reject(self, source, reason=None):
        self.transport[source]['rejected'] += 1
        if reason:
            self.detail = reason

    def message_fresh(self, timestamp, max_age_s):
        age = self.now_s() - timestamp
        return -self.p['max_future_skew_s'] <= age <= max_age_s

    def on_time_sync(self, message):
        with self.pending_lock:
            self._record_receive('time_sync')
            self.pending_time_sync = message

    def on_nx(self, message):
        with self.pending_lock:
            self._record_receive('nx')
            self.pending_nx = message

    def on_d435(self, message):
        with self.pending_lock:
            self._record_receive('d435')
            self.pending_d435 = message

    def on_event(self, message):
        with self.pending_lock:
            if len(self.pending_events) >= 100:
                self.pending_events.pop(0)
            self.pending_events.append(message)

    def _drain_pending(self):
        with self.pending_lock:
            time_sync = self.pending_time_sync
            nx_message = self.pending_nx
            d435_message = self.pending_d435
            events = self.pending_events
            self.pending_time_sync = None
            self.pending_nx = None
            self.pending_d435 = None
            self.pending_events = []
        return time_sync, nx_message, d435_message, events

    def _process_time_sync(self, message):
        receive_time = self.now_s()
        sample = SyncSample(
            source_epoch=int(message.source_epoch),
            sequence=int(message.sequence),
            sample_time_nx_s=stamp_s(message.header.stamp),
            offset_s=float(message.offset_ns) * 1e-9,
            uncertainty_s=float(message.uncertainty_ns) * 1e-9,
            synchronized=bool(message.synchronized),
            servo_state=int(message.servo_state),
            clock_source=str(message.clock_source),
        )
        result = self.time_sync_guard.ingest(sample, receive_time)
        if not result.accepted:
            self._record_reject('time_sync', result.reason)
            return
        if result.clock_step:
            if self.state != IDLE:
                self.reset(CatchState.RESET_BALL_LOST, 'NX clock offset stepped')
            self.nx_order.reset()
        if (
            abs(sample.offset_s) > self.p['time_sync_warn_offset_s']
            and receive_time - self.last_time_sync_warning_s > 1.0
        ):
            self.get_logger().warning(
                f'NX clock offset is {sample.offset_s * 1e3:.2f} ms; '
                f'trust scale={self.time_sync_guard.trust_scale():.2f}'
            )
            self.last_time_sync_warning_s = receive_time

    def nx_capture_time(self, message):
        timestamp = stamp_s(message.header.stamp)
        if not self.p['require_nx_time_sync']:
            return timestamp
        corrected, reason = self.time_sync_guard.corrected_time(
            timestamp, int(message.source_epoch), self.now_s()
        )
        if corrected is None:
            self._record_reject('nx', reason)
            return None
        return corrected

    def on_odom(self, message):
        timestamp = stamp_s(message.header.stamp)
        receive_time = self.now_s()
        with self.pending_lock:
            self._record_receive('odom')
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        decision = self.odom_guard.accept(
            timestamp,
            receive_time,
            message.header.frame_id,
            message.child_frame_id,
            [position.x, position.y, position.z],
            quaternion,
        )
        if not decision.accepted:
            self._record_reject('odom', decision.reason)
            return
        if self.odom.add(
            timestamp, [position.x, position.y, position.z], quaternion
        ):
            self.last_odom_receive = receive_time
        else:
            self._record_reject('odom', 'odom history rejected sample')

    def odom_at(self, timestamp):
        return self.odom.pose_at(timestamp)

    def transform_observation(self, position, covariance, timestamp, translation, rotation):
        return self.odom.transform_observation(
            position, covariance, timestamp, translation, rotation
        )

    def _initialize_cycle(self, state, source_epoch, detail):
        self.state = int(state)
        self.source_epoch = int(source_epoch)
        self.nx_track_id = None
        self.catch_id = self.next_catch_id
        self.next_catch_id = 1 if self.next_catch_id == 0xFFFFFFFF else self.next_catch_id + 1
        self.arrived = False
        self.d435_confirm = 0
        self.impact_time = 0.0
        self.impact_deadline = 0.0
        self.landing = None
        self.filtered_position = None
        self.filtered_velocity = None
        self.predicted_trajectory = []
        self.landing_source = 0
        self.landing_stable_frames = {
            CatchState.SOURCE_NX: 0,
            CatchState.SOURCE_D435: 0,
        }
        self.last_landing_candidate = {
            CatchState.SOURCE_NX: None,
            CatchState.SOURCE_D435: None,
        }
        self.last_landing_candidate_time = {
            CatchState.SOURCE_NX: None,
            CatchState.SOURCE_D435: None,
        }
        self.near_rgbd_frames = 0
        self.distance_to_landing = math.nan
        self.reset_reason = CatchState.RESET_NONE
        self.detail = str(detail)
        self.far_tracker.reset()
        self.near_tracker.reset()
        self.last_nx_stamp = -math.inf
        self.last_nx_frame = -1

    def _start_cycle(self, message):
        self._initialize_cycle(
            FAR_NX, int(message.source_epoch), 'new NX observation cycle'
        )
        self.nx_track_id = int(message.track_id)

    def _start_d435_cycle(self, message):
        self._initialize_cycle(
            NEAR_D435,
            int(message.source_epoch),
            'new D435-only observation cycle',
        )
        self.d435_confirm = 1

    def _reset_prediction_gate(self, source=None):
        sources = (
            (CatchState.SOURCE_NX, CatchState.SOURCE_D435)
            if source is None
            else (source,)
        )
        for current_source in sources:
            self.landing_stable_frames[current_source] = 0
            self.last_landing_candidate[current_source] = None
            self.last_landing_candidate_time[current_source] = None

    def _prediction_candidate(self, tracker, source, timestamp):
        prediction = tracker.trajectory(
            self.p['intercept_plane_z_m'], self.p['trajectory_step_s']
        )
        if prediction is None:
            self._reset_prediction_gate(source)
            return None
        trajectory, time_to_land = prediction
        landing = trajectory[-1]
        if (
            tracker.last_prediction_method == 'polynomial'
            and not self.p['prediction_allow_polynomial']
        ):
            self._reset_prediction_gate(source)
            return None
        if tracker.last_prediction_confidence < self.p['prediction_min_confidence']:
            self._reset_prediction_gate(source)
            return None
        if tracker.speed < self.p['prediction_min_speed_mps']:
            self._reset_prediction_gate(source)
            return None
        if (
            tracker.last_prediction_method != 'polynomial'
            and tracker.last_student_w < self.p['prediction_min_student_weight']
        ):
            self._reset_prediction_gate(source)
            return None
        if not (
            self.p['prediction_min_time_to_land_s']
            <= time_to_land
            <= self.p['prediction_max_time_to_land_s']
        ):
            self._reset_prediction_gate(source)
            return None
        previous = self.last_landing_candidate[source]
        previous_time = self.last_landing_candidate_time[source]
        interval = math.inf if previous_time is None else timestamp - previous_time
        if previous is None:
            self.landing_stable_frames[source] = 1
        else:
            jump = float(np.linalg.norm((landing - previous)[0:2]))
            self.landing_stable_frames[source] = (
                self.landing_stable_frames[source] + 1
                if 0.0 <= interval <= self.p['prediction_max_stable_interval_s']
                and jump <= self.p['prediction_max_stable_jump_m']
                else 1
            )
        self.last_landing_candidate[source] = landing.copy()
        self.last_landing_candidate_time[source] = float(timestamp)
        if self.landing_stable_frames[source] < self.p['prediction_stable_frames']:
            self.detail = 'landing prediction stabilizing'
            return None
        return {
            'landing': landing.copy(),
            'trajectory': [point.copy() for point in trajectory],
            'time_to_land': float(time_to_land),
            'confidence': float(tracker.last_prediction_confidence),
        }

    def _commit_candidate(self, tracker, source, timestamp, candidate):
        if candidate is None:
            return False
        proposed_impact = timestamp + candidate['time_to_land']
        if not math.isfinite(proposed_impact) or proposed_impact <= self.now_s():
            return False
        if self.impact_deadline <= 0.0:
            self.impact_deadline = proposed_impact + self.p['max_cycle_extension_s']
        self.impact_time = min(proposed_impact, self.impact_deadline)
        self.landing = candidate['landing']
        self.landing_source = source
        self.filtered_position = tracker.position
        self.filtered_velocity = tracker.velocity
        self.predicted_trajectory = candidate['trajectory']
        return True

    def _set_filter_diagnostics(self, tracker, source, candidate=None):
        active = (
            source == CatchState.SOURCE_NX and self.state != NEAR_D435
        ) or (
            source == CatchState.SOURCE_D435 and self.state == NEAR_D435
        )
        if not active:
            return
        self.filtered_position = tracker.position
        self.filtered_velocity = tracker.velocity
        if candidate is not None:
            self.predicted_trajectory = candidate['trajectory']

    def _handover_consistent(self, timestamp, candidate):
        far_state = self.far_tracker.predict_state(timestamp)
        if far_state is None or self.near_tracker.position is None:
            return False
        far_position, _, far_covariance = far_state
        near_position = self.near_tracker.position
        near_covariance = self.near_tracker.position_covariance
        accepted, _ = association_gate(
            far_position,
            far_covariance,
            near_position,
            near_covariance,
            self.p['handover_position_gate_chi2'],
        )
        if not accepted:
            self.detail = 'D435 candidate rejected by NX position gate'
            return False
        if self.landing is not None:
            landing_delta = float(np.linalg.norm(candidate['landing'] - self.landing))
            if landing_delta > self.p['handover_max_landing_delta_m']:
                self.detail = 'D435 candidate rejected by landing consistency gate'
                return False
        return True

    def _process_nx(self, message):
        if self.d435_only:
            return
        timestamp = self.nx_capture_time(message)
        if timestamp is None:
            return
        frame_id = int(message.frame_id)
        valid = (
            self.message_fresh(timestamp, self.p['nx_max_message_age_s'])
            and message.header.frame_id == self.p['nx_frame']
            and int(message.class_id) == self.p['volleyball_class_id']
            and message.detection_valid
            and message.stereo_valid
            and not message.fallback_observation
            and message.detection_confidence >= self.p['nx_min_confidence']
            and valid_bbox(message.bbox_left_xyxy)
            and valid_bbox(message.bbox_right_xyxy)
            and finite_vector([message.position.x, message.position.y, message.position.z], 3)
            and abs(int(message.stereo_timestamp_delta_ns))
            <= self.p['nx_max_stereo_delta_ns']
            and math.isfinite(float(message.depth_sigma_m))
            and 0.0 <= float(message.depth_sigma_m) <= self.p['nx_max_depth_sigma_m']
        )
        if not valid:
            self._record_reject('nx')
            return
        if self.state == NEAR_D435:
            return

        previous_epoch = self.nx_order.epoch
        order = self.nx_order.accept(int(message.source_epoch), timestamp, frame_id)
        if not order.accepted:
            self._record_reject('nx', order.reason)
            return
        if order.new_epoch and previous_epoch != 0:
            self.transport['nx']['epoch_changes'] += 1

        if self.state == IDLE:
            self._start_cycle(message)
        elif int(message.source_epoch) != self.source_epoch:
            self.reset(CatchState.RESET_BALL_LOST, 'NX source epoch changed')
            self._start_cycle(message)
        elif int(message.track_id) != self.nx_track_id:
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
        if self.far_tracker.update(
            position,
            covariance_odom,
            timestamp,
            trust=float(np.clip(message.match_confidence, 0.0, 1.0))
            * (
                self.time_sync_guard.trust_scale()
                if self.p['require_nx_time_sync']
                else 1.0
            ),
            consistency=1.0,
        ):
            candidate = self._prediction_candidate(
                self.far_tracker, CatchState.SOURCE_NX, timestamp
            )
            self._set_filter_diagnostics(
                self.far_tracker, CatchState.SOURCE_NX, candidate
            )
            self._commit_candidate(
                self.far_tracker, CatchState.SOURCE_NX, timestamp, candidate
            )

    def _process_d435(self, message):
        timestamp = stamp_s(message.header.stamp)
        frame_id = int(message.frame_id)
        source_epoch = int(message.source_epoch)
        envelope_valid = (
            self.message_fresh(timestamp, self.p['d435_max_message_age_s'])
            and source_epoch != 0
            and message.header.frame_id == self.p['d435_frame']
        )
        if not envelope_valid:
            self._record_reject('d435', 'D435 message envelope invalid or stale')
            return
        previous_epoch = self.d435_order.epoch
        order = self.d435_order.accept(source_epoch, timestamp, frame_id)
        if not order.accepted:
            self._record_reject('d435', order.reason)
            return
        if order.new_epoch and previous_epoch != 0:
            self.transport['d435']['epoch_changes'] += 1
        if self.last_d435_epoch and order.new_epoch:
            if self.state == NEAR_D435:
                self.reset(CatchState.RESET_BALL_LOST, 'D435 source epoch changed')
                return
            self.near_tracker.reset()
            self.near_rgbd_frames = 0
            self.d435_confirm = 0
            self.last_d435_rgbd = 0.0
            self._reset_prediction_gate(CatchState.SOURCE_D435)
        self.last_d435_epoch = source_epoch
        detection_valid = (
            int(message.class_id) == self.p['volleyball_class_id']
            and message.detection_valid
            and message.detection_confidence >= self.p['d435_min_confidence']
            and valid_bbox(message.bbox_xyxy)
        )
        if detection_valid:
            self.last_d435_detection = self.now_s()
            self.d435_confirm += 1
        else:
            self.d435_confirm = 0
            return

        eligible_state = (
            self.state in (IDLE, NEAR_D435)
            if self.d435_only
            else self.state in (WAIT_D435, NEAR_D435)
        )
        if not eligible_state or not message.rgbd_valid:
            return
        if not d435_timing_valid(
            message.rgb_depth_timestamp_delta_ns,
            message.timestamp_uncertainty_ns,
            message.timestamp_mapping_valid,
            self.p['d435_max_rgb_depth_delta_ns'],
            int(float(self.p['d435_max_timestamp_uncertainty_s']) * 1e9),
        ):
            self._record_reject('d435', 'D435 RGB/depth timing invalid')
            return
        if (
            int(message.valid_depth_samples) <= 0
            or not math.isfinite(float(message.depth_spread_m))
            or float(message.depth_spread_m) < 0.0
        ):
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
        if self.d435_only and self.state == IDLE:
            self._start_d435_cycle(message)
        sample_trust = min(1.0, float(message.valid_depth_samples) / 32.0)
        spread_trust = math.exp(-max(0.0, float(message.depth_spread_m)) / 0.10)
        if self.near_tracker.update(
            position_odom,
            covariance_odom,
            timestamp,
            trust=sample_trust * spread_trust,
            consistency=1.0,
        ):
            self.last_d435_rgbd = self.now_s()
            self.near_rgbd_frames += 1
            candidate = self._prediction_candidate(
                self.near_tracker, CatchState.SOURCE_D435, timestamp
            )
            self._set_filter_diagnostics(
                self.near_tracker, CatchState.SOURCE_D435, candidate
            )
            if self.state == WAIT_D435:
                if (
                    self.arrived
                    and self.d435_confirm >= self.p['d435_confirm_frames']
                    and self.near_rgbd_frames >= self.p['handover_min_rgbd_frames']
                    and self.d435_extrinsics_ok
                    and candidate is not None
                    and self._handover_consistent(timestamp, candidate)
                ):
                    self.state = NEAR_D435
                    self.detail = 'D435 handover committed after dual-source consistency gate'
                    self._commit_candidate(
                        self.near_tracker,
                        CatchState.SOURCE_D435,
                        timestamp,
                        candidate,
                    )
            else:
                self.detail = 'D435 near-field trajectory active'
                self._set_filter_diagnostics(
                    self.near_tracker, CatchState.SOURCE_D435, candidate
                )
                self._commit_candidate(
                    self.near_tracker,
                    CatchState.SOURCE_D435,
                    timestamp,
                    candidate,
                )

    def _process_event(self, message):
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
        self.filtered_position = None
        self.filtered_velocity = None
        self.predicted_trajectory = []
        self.landing_stable_frames = {
            CatchState.SOURCE_NX: 0,
            CatchState.SOURCE_D435: 0,
        }
        self.last_landing_candidate = {
            CatchState.SOURCE_NX: None,
            CatchState.SOURCE_D435: None,
        }
        self.last_landing_candidate_time = {
            CatchState.SOURCE_NX: None,
            CatchState.SOURCE_D435: None,
        }
        self.near_rgbd_frames = 0
        self.last_d435_rgbd = 0.0
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
        latest = self.odom.latest
        if latest is None or now - self.last_odom_receive > self.p['odom_timeout_s']:
            return None
        timestamp, position, yaw = latest
        if now - timestamp > self.p['odom_timeout_s']:
            return None
        return position, yaw

    def tick(self):
        time_sync, nx_message, d435_message, events = self._drain_pending()
        if time_sync is not None:
            self._process_time_sync(time_sync)
        for event in events:
            self._process_event(event)
        if nx_message is not None:
            self._process_nx(nx_message)
        if d435_message is not None:
            self._process_d435(d435_message)

        now = self.now_s()
        self.publish_transport_diagnostics(now)
        if (
            self.state != IDLE
            and self.impact_time > 0.0
            and now >= self.impact_time + self.p['impact_grace_s']
        ):
            self.reset(CatchState.RESET_IMPACT_TIME, 'impact time reached')
            return
        if self.state == FAR_NX and now - self.last_nx_receive > self.p['nx_lost_timeout_s']:
            self.reset(CatchState.RESET_BALL_LOST, 'NX ball lost')
            return
        if (
            self.state == WAIT_D435
            and now - self.last_nx_receive > self.p['nx_lost_timeout_s']
            and now - self.last_d435_detection > self.p['d435_lost_timeout_s']
        ):
            self.reset(CatchState.RESET_BALL_LOST, 'NX and D435 ball lost during handover')
            return
        if (
            self.state == NEAR_D435
            and now - self.last_d435_rgbd > self.p['d435_lost_timeout_s']
        ):
            self.reset(CatchState.RESET_BALL_LOST, 'D435 RGB-D timing/data lost')
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
        catcher_offset = self.catcher_point_base
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
            self.near_rgbd_frames = 0
            self.near_tracker.reset()
            self._reset_prediction_gate(CatchState.SOURCE_D435)
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
        self.publish_diagnostics(stamp)

        required_extrinsics_ok = (
            self.nx_extrinsics_ok
            if self.landing_source == CatchState.SOURCE_NX
            else self.d435_extrinsics_ok
        )
        if self.state == IDLE or not required_extrinsics_ok:
            if self.goal_valid:
                self.publish_valid(False)
            return

        goal_xy = landing_to_catcher_goal(
            self.landing,
            base_position,
            yaw,
            self.catcher_point_base,
            self.p['landing_goal_correction_base'],
        )
        if goal_xy is None:
            if self.goal_valid:
                self.publish_valid(False)
            self.detail = 'landing/catcher transform invalid; control inhibited'
            return
        goal = PoseStamped()
        goal.header.stamp = stamp
        goal.header.frame_id = self.p['base_frame']
        goal.pose.position.x = float(goal_xy[0])
        goal.pose.position.y = float(goal_xy[1])
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        if not self.goal_valid:
            self.publish_valid(True)

    def publish_diagnostics(self, stamp):
        if not self.p['diagnostics_enabled']:
            return
        now = self.now_s()
        rate = max(0.1, float(self.p['diagnostics_rate_hz']))
        if now - self.last_diagnostic_publish < 1.0 / rate:
            return
        self.last_diagnostic_publish = now

        if self.filtered_position is not None:
            filtered = PointStamped()
            filtered.header.stamp = stamp
            filtered.header.frame_id = self.p['world_frame']
            filtered.point.x, filtered.point.y, filtered.point.z = map(
                float, self.filtered_position
            )
            self.filtered_position_pub.publish(filtered)

        if self.filtered_velocity is not None:
            velocity = Vector3Stamped()
            velocity.header.stamp = stamp
            velocity.header.frame_id = self.p['world_frame']
            velocity.vector.x, velocity.vector.y, velocity.vector.z = map(
                float, self.filtered_velocity
            )
            self.filtered_velocity_pub.publish(velocity)

        if self.predicted_trajectory:
            path = Path()
            path.header.stamp = stamp
            path.header.frame_id = self.p['world_frame']
            for point in self.predicted_trajectory:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(
                    float, point
                )
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
            self.predicted_path_pub.publish(path)

    def publish_transport_diagnostics(self, now):
        rate = max(0.2, float(self.p['transport_diagnostics_rate_hz']))
        if now - self.last_transport_publish < 1.0 / rate:
            return
        self.last_transport_publish = now

        def age(source):
            received = self.transport[source]['last_receive']
            return math.inf if received <= 0.0 else max(0.0, now - received)

        ages = {source: age(source) for source in self.transport}
        nx_online = ages['nx'] <= max(0.1, 2.0 * float(self.p['nx_deadline_s']))
        d435_online = ages['d435'] <= max(
            0.1, 2.0 * float(self.p['d435_deadline_s'])
        )
        time_sync_online = ages['time_sync'] <= self.p['time_sync_timeout_s']
        sync_sample = self.time_sync_guard.sample
        time_sync_valid = False
        if sync_sample is not None:
            time_sync_valid, _ = self.time_sync_guard.status(
                sync_sample.source_epoch, now
            )
        odom_online = ages['odom'] <= self.p['odom_timeout_s']
        odom_valid = self.odom_guard.valid_at(now, self.p['odom_timeout_s'])
        message = TransportDiagnostics()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.p['world_frame']
        message.nx_online = nx_online
        message.d435_online = d435_online
        message.time_sync_online = time_sync_online
        message.time_sync_valid = time_sync_valid
        message.odom_online = odom_online
        message.odom_valid = odom_valid
        message.nx_receive_age_s = float(ages['nx'])
        message.d435_receive_age_s = float(ages['d435'])
        message.time_sync_receive_age_s = float(ages['time_sync'])
        message.odom_receive_age_s = float(ages['odom'])
        message.odom_rate_hz = float(self.odom_guard.rate_hz)
        for source in ('nx', 'd435', 'time_sync', 'odom'):
            metrics = self.transport[source]
            setattr(message, f'{source}_messages', metrics['messages'])
            setattr(message, f'{source}_deadline_misses', metrics['deadline_misses'])
        message.nx_rejected = self.transport['nx']['rejected']
        message.d435_rejected = self.transport['d435']['rejected']
        message.time_sync_rejected = self.transport['time_sync']['rejected']
        message.odom_rejected = self.transport['odom']['rejected']
        message.nx_epoch_changes = self.transport['nx']['epoch_changes']
        message.d435_epoch_changes = self.transport['d435']['epoch_changes']
        required_links = [
            ('d435', d435_online),
            ('odom', odom_online),
        ] if self.d435_only else [
            ('nx', nx_online),
            ('d435', d435_online),
            ('time_sync', time_sync_online),
            ('odom', odom_online),
        ]
        offline = [
            name
            for name, online in required_links
            if not online
        ]
        if odom_online and not odom_valid:
            offline.append('odom-invalid')
        online_detail = (
            'all D435-only required links online'
            if self.d435_only
            else 'all monitored links online'
        )
        message.detail = online_detail if not offline else (
            'offline/stale: ' + ','.join(offline)
        )
        self.transport_pub.publish(message)

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
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
