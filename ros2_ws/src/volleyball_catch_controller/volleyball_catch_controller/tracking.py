from collections import deque
import math
import threading

import numpy as np


def finite_vector(values, size=None):
    array = np.asarray(values, dtype=float)
    return (size is None or array.size == size) and bool(np.all(np.isfinite(array)))


def covariance3(values, default_variance):
    array = np.asarray(values, dtype=float)
    if array.size != 9 or not np.all(np.isfinite(array)):
        return np.eye(3, dtype=float) * default_variance
    matrix = array.reshape(3, 3)
    matrix = 0.5 * (matrix + matrix.T)
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0.0):
        return np.eye(3, dtype=float) * default_variance
    return matrix


def association_gate(
    first_position,
    first_covariance,
    second_position,
    second_covariance,
    gate_chi2,
    extra_variance=0.01,
):
    first_position = np.asarray(first_position, dtype=float)
    second_position = np.asarray(second_position, dtype=float)
    first_covariance = np.asarray(first_covariance, dtype=float)
    second_covariance = np.asarray(second_covariance, dtype=float)
    if (
        first_position.shape != (3,)
        or second_position.shape != (3,)
        or first_covariance.shape != (3, 3)
        or second_covariance.shape != (3, 3)
        or not np.all(np.isfinite(first_position))
        or not np.all(np.isfinite(second_position))
        or not np.all(np.isfinite(first_covariance))
        or not np.all(np.isfinite(second_covariance))
    ):
        return False, math.inf
    innovation = second_position - first_position
    covariance = (
        first_covariance
        + second_covariance
        + np.eye(3, dtype=float) * max(0.0, float(extra_variance))
    )
    try:
        distance = float(innovation.T @ np.linalg.inv(covariance) @ innovation)
    except np.linalg.LinAlgError:
        return False, math.inf
    return math.isfinite(distance) and distance <= float(gate_chi2), distance


def wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def landing_to_catcher_goal(
    landing_world,
    base_position_world,
    base_yaw,
    catcher_point_base,
    goal_correction_base=(0.0, 0.0),
):
    """Return the planar catcher displacement expressed in ``base_link``."""
    landing = np.asarray(landing_world, dtype=float)
    base_position = np.asarray(base_position_world, dtype=float)
    catcher_offset = np.asarray(catcher_point_base, dtype=float)
    goal_correction = np.asarray(goal_correction_base, dtype=float)
    yaw = float(base_yaw)
    if (
        landing.shape != (3,)
        or base_position.shape != (3,)
        or catcher_offset.shape != (3,)
        or goal_correction.shape != (2,)
        or not np.all(np.isfinite(landing))
        or not np.all(np.isfinite(base_position))
        or not np.all(np.isfinite(catcher_offset))
        or not np.all(np.isfinite(goal_correction))
        or not math.isfinite(yaw)
    ):
        return None
    delta = landing - base_position
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            cosine * delta[0] + sine * delta[1] - catcher_offset[0]
            + goal_correction[0],
            -sine * delta[0] + cosine * delta[1] - catcher_offset[1]
            + goal_correction[1],
        ],
        dtype=float,
    )


def catcher_geometry_from_camera(
    camera_translation_base, catcher_offset_from_camera_base, ball_radius_m
):
    """Resolve the plate center and ball-center contact height in base axes."""
    camera_translation = np.asarray(camera_translation_base, dtype=float)
    catcher_offset = np.asarray(catcher_offset_from_camera_base, dtype=float)
    ball_radius = float(ball_radius_m)
    if (
        camera_translation.shape != (3,)
        or catcher_offset.shape != (3,)
        or not np.all(np.isfinite(camera_translation))
        or not np.all(np.isfinite(catcher_offset))
        or not math.isfinite(ball_radius)
        or ball_radius <= 0.0
    ):
        return None
    catcher_point = camera_translation + catcher_offset
    return catcher_point, float(catcher_point[2] + ball_radius)


def normalize_quaternion(values):
    quaternion = np.asarray(values, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        return None
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        return None
    return quaternion / norm


def yaw_quaternion(yaw):
    half = 0.5 * float(yaw)
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=float)


def quaternion_yaw(quaternion):
    x, y, z, w = normalize_quaternion(quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_matrix(quaternion):
    normalized = normalize_quaternion(quaternion)
    if normalized is None:
        return None
    x, y, z, w = normalized
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quaternion_slerp(first, second, ratio):
    first = normalize_quaternion(first)
    second = normalize_quaternion(second)
    if first is None or second is None:
        return None
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion(first + float(ratio) * (second - first))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return normalize_quaternion(
        math.sin((1.0 - float(ratio)) * angle) / sine * first
        + math.sin(float(ratio) * angle) / sine * second
    )


class OdomHistory:
    """Timestamped planar odometry used to motion-compensate camera observations."""

    def __init__(self, history_s=3.0, max_gap_s=0.05, max_extrapolation_s=0.02):
        self.history_s = float(history_s)
        self.max_gap_s = float(max_gap_s)
        self.max_extrapolation_s = float(max_extrapolation_s)
        self.samples = deque()
        self.lock = threading.RLock()

    def clear(self):
        with self.lock:
            self.samples.clear()

    def add(self, timestamp, position, orientation):
        timestamp = float(timestamp)
        position = np.asarray(position, dtype=float)
        orientation_array = np.asarray(orientation, dtype=float)
        quaternion = (
            yaw_quaternion(float(orientation_array))
            if orientation_array.ndim == 0
            else normalize_quaternion(orientation_array)
        )
        if (
            not math.isfinite(timestamp)
            or position.shape != (3,)
            or not np.all(np.isfinite(position))
            or quaternion is None
        ):
            return False
        with self.lock:
            if self.samples and timestamp <= self.samples[-1][0]:
                return False
            self.samples.append((timestamp, position.copy(), quaternion.copy()))
            while self.samples and timestamp - self.samples[0][0] > self.history_s:
                self.samples.popleft()
        return True

    @property
    def latest(self):
        with self.lock:
            if not self.samples:
                return None
            timestamp, position, quaternion = self.samples[-1]
            return timestamp, position.copy(), quaternion_yaw(quaternion)

    @property
    def latest_se3(self):
        with self.lock:
            if not self.samples:
                return None
            timestamp, position, quaternion = self.samples[-1]
            return timestamp, position.copy(), quaternion.copy()

    def pose_at(self, timestamp):
        pose = self.pose_at_se3(timestamp)
        if pose is None:
            return None
        position, quaternion = pose
        return position, quaternion_yaw(quaternion)

    def pose_at_se3(self, timestamp):
        timestamp = float(timestamp)
        with self.lock:
            samples = list(self.samples)
        if not samples or not math.isfinite(timestamp):
            return None
        if timestamp >= samples[-1][0]:
            delta = timestamp - samples[-1][0]
            if delta > self.max_extrapolation_s:
                return None
            last = samples[-1]
            if len(samples) < 2 or delta <= 0.0:
                return last[1].copy(), last[2].copy()
            before = samples[-2]
            sample_dt = last[0] - before[0]
            if sample_dt <= 0.0 or sample_dt > self.max_gap_s:
                return last[1].copy(), last[2].copy()
            velocity = (last[1] - before[1]) / sample_dt
            orientation = quaternion_slerp(
                before[2], last[2], 1.0 + delta / sample_dt
            )
            return last[1] + velocity * delta, orientation

        for index in range(1, len(samples)):
            before, after = samples[index - 1], samples[index]
            if before[0] <= timestamp <= after[0]:
                gap = after[0] - before[0]
                if gap <= 0.0 or gap > self.max_gap_s:
                    return None
                ratio = (timestamp - before[0]) / gap
                position = before[1] * (1.0 - ratio) + after[1] * ratio
                orientation = quaternion_slerp(before[2], after[2], ratio)
                return position, orientation
        return None

    def transform_observation(
        self, position, covariance, timestamp, camera_translation, camera_rotation
    ):
        odom_pose = self.pose_at_se3(timestamp)
        position = np.asarray(position, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        translation = np.asarray(camera_translation, dtype=float)
        rotation = np.asarray(camera_rotation, dtype=float)
        if (
            odom_pose is None
            or position.shape != (3,)
            or covariance.shape != (3, 3)
            or translation.shape != (3,)
            or rotation.shape != (3, 3)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(covariance))
            or not np.all(np.isfinite(translation))
            or not np.all(np.isfinite(rotation))
        ):
            return None
        base_position, base_quaternion = odom_pose
        base_rotation = quaternion_matrix(base_quaternion)
        total_rotation = base_rotation @ rotation
        point_base = rotation @ position + translation
        point_odom = base_position + base_rotation @ point_base
        covariance_odom = total_rotation @ covariance @ total_rotation.T
        return point_odom, covariance_odom


class BallisticTracker:
    """NX-equivalent Student-t drag EKF operating in odom (Z-up) coordinates."""

    def __init__(
        self,
        gravity_mps2=9.81,
        drag_coefficient=0.10,
        air_density=1.225,
        ball_mass_kg=0.270,
        ball_radius_m=0.105,
        student_t_nu=12.0,
        q_position=1e-4,
        q_velocity=1.5,
        innovation_gate_chi2=25.0,
        max_dt_s=0.5,
        min_updates=2,
        min_speed_mps=0.5,
        max_predict_time_s=3.0,
        rk4_dt_s=0.008,
        poly_min_frames=6,
        history_max=20,
    ):
        self.gravity = float(gravity_mps2)
        self.drag_coefficient = float(drag_coefficient)
        self.air_density = float(air_density)
        self.ball_mass = float(ball_mass_kg)
        self.ball_radius = float(ball_radius_m)
        self.student_t_nu = float(student_t_nu)
        self.q_position = float(q_position)
        self.q_velocity = float(q_velocity)
        self.innovation_gate = float(innovation_gate_chi2)
        self.max_dt = float(max_dt_s)
        self.min_updates = int(min_updates)
        self.min_speed = float(min_speed_mps)
        self.max_predict_time = float(max_predict_time_s)
        self.rk4_dt = float(rk4_dt_s)
        self.poly_min_frames = int(poly_min_frames)
        self.history_max = int(history_max)
        area = math.pi * self.ball_radius * self.ball_radius
        self.drag_k = (
            0.5 * self.drag_coefficient * self.air_density * area
            / max(self.ball_mass, 1e-9)
        )
        self.reset()

    def reset(self):
        self.x = np.zeros(6, dtype=float)
        self.p = np.eye(6, dtype=float)
        self.timestamp = None
        self.updates = 0
        self.last_student_w = 1.0
        self.last_innovation = np.zeros(3, dtype=float)
        self.last_prediction_method = 'none'
        self.last_prediction_confidence = 0.0
        self.samples = deque()

    @property
    def initialized(self):
        return self.timestamp is not None

    @property
    def ready(self):
        return self.updates >= self.min_updates

    @property
    def position(self):
        return self.x[0:3].copy() if self.initialized else None

    @property
    def velocity(self):
        return self.x[3:6].copy() if self.initialized else None

    @property
    def speed(self):
        return float(np.linalg.norm(self.x[3:6])) if self.initialized else 0.0

    @property
    def position_covariance(self):
        return self.p[0:3, 0:3].copy() if self.initialized else None

    def _acceleration(self, velocity):
        velocity = np.asarray(velocity, dtype=float)
        return np.array([0.0, 0.0, -self.gravity], dtype=float) - (
            self.drag_k * np.linalg.norm(velocity) * velocity
        )

    def _predict_in_place(self, timestamp):
        dt = float(timestamp) - float(self.timestamp)
        if dt <= 0.0:
            return False
        velocity = self.x[3:6].copy()
        speed = float(np.linalg.norm(velocity))
        acceleration = self._acceleration(velocity)

        transition = np.eye(6, dtype=float)
        transition[0:3, 3:6] = np.eye(3) * dt
        if speed > 1e-3:
            drag_jacobian = -self.drag_k * (
                speed * np.eye(3) + np.outer(velocity, velocity) / speed
            )
            transition[3:6, 3:6] += drag_jacobian * dt

        self.x[0:3] += velocity * dt + 0.5 * acceleration * dt * dt
        self.x[3:6] += acceleration * dt
        process = np.zeros((6, 6), dtype=float)
        process[0:3, 0:3] = np.eye(3) * self.q_position * dt
        process[3:6, 3:6] = np.eye(3) * self.q_velocity * dt
        self.p = transition @ self.p @ transition.T + process
        self.timestamp = float(timestamp)
        return True

    def _append_sample(self, timestamp, measurement):
        self.samples.append((float(timestamp), np.asarray(measurement, dtype=float).copy()))
        while len(self.samples) > self.history_max:
            self.samples.popleft()

    def update(
        self,
        position,
        measurement_covariance,
        timestamp,
        trust=1.0,
        consistency=1.0,
    ):
        measurement = np.asarray(position, dtype=float)
        covariance = np.asarray(measurement_covariance, dtype=float)
        timestamp = float(timestamp)
        if (
            measurement.shape != (3,)
            or covariance.shape != (3, 3)
            or not np.all(np.isfinite(measurement))
            or not np.all(np.isfinite(covariance))
            or not math.isfinite(timestamp)
        ):
            return False
        covariance = 0.5 * (covariance + covariance.T)
        if np.any(np.diag(covariance) <= 0.0):
            return False
        if not self.initialized:
            self.x[0:3] = measurement
            self.p = np.diag([0.05, 0.05, 0.05, 25.0, 25.0, 25.0])
            self.timestamp = timestamp
            self.updates = 1
            self._append_sample(timestamp, measurement)
            return True

        dt = timestamp - float(self.timestamp)
        if dt <= 0.0:
            return False
        if dt > self.max_dt:
            self.x[0:3] = measurement
            self.p = np.diag([0.05, 0.05, 0.05, 25.0, 25.0, 25.0])
            self.timestamp = timestamp
            self.updates = 1
            self.last_student_w = 1.0
            self.last_innovation.fill(0.0)
            self.samples.clear()
            self._append_sample(timestamp, measurement)
            return True
        if not self._predict_in_place(timestamp):
            return False

        consistency = min(1.0, max(0.05, float(consistency)))
        trust = min(1.0, max(0.0, float(trust)))
        inflate = (1.0 + 2.0 * (1.0 - consistency)) * (1.0 + (1.0 - trust))
        measurement_r = covariance * (inflate * inflate)
        innovation = measurement - self.x[0:3]
        innovation_covariance = self.p[0:3, 0:3] + measurement_r
        try:
            inverse = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            return True
        delta = float(innovation.T @ inverse @ innovation)
        nu = max(1.0, self.student_t_nu)
        weight = (nu + 3.0) / (nu + max(0.0, delta))
        self.last_student_w = float(weight)
        self.last_innovation = innovation.copy()
        if not math.isfinite(delta) or delta > self.innovation_gate:
            return False

        effective_r = measurement_r / max(0.05, weight)
        innovation_covariance = self.p[0:3, 0:3] + effective_r
        try:
            inverse = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            return True
        observation = np.zeros((3, 6), dtype=float)
        observation[:, 0:3] = np.eye(3)
        gain = self.p @ observation.T @ inverse
        self.x += gain @ innovation
        residual = np.eye(6, dtype=float) - gain @ observation
        self.p = residual @ self.p @ residual.T + gain @ effective_r @ gain.T
        self.p = 0.5 * (self.p + self.p.T)
        self.updates += 1
        self._append_sample(timestamp, measurement)
        return True

    def predict_state(self, timestamp):
        if not self.initialized:
            return None
        timestamp = float(timestamp)
        dt = timestamp - float(self.timestamp)
        if not math.isfinite(dt) or dt < 0.0 or dt > self.max_dt:
            return None
        position = self.x[0:3].copy()
        velocity = self.x[3:6].copy()
        if dt > 0.0:
            acceleration = self._acceleration(velocity)
            position += velocity * dt + 0.5 * acceleration * dt * dt
            velocity += acceleration * dt
        transition = np.eye(6, dtype=float)
        transition[0:3, 3:6] = np.eye(3) * dt
        covariance = transition @ self.p @ transition.T
        return position, velocity, covariance[0:3, 0:3]

    def _rollout(self, ground_z_m, output_step_s):
        if not self.ready or self.speed < self.min_speed:
            return None
        output_step_s = float(output_step_s)
        if not math.isfinite(output_step_s) or output_step_s <= 0.0:
            return None
        position = self.x[0:3].copy()
        velocity = self.x[3:6].copy()
        if position[2] <= float(ground_z_m) + 0.05:
            return None

        def derivative(state):
            result = np.empty(6, dtype=float)
            result[0:3] = state[3:6]
            result[3:6] = self._acceleration(state[3:6])
            return result

        state = np.concatenate([position, velocity])
        points = [position.copy()]
        next_output = output_step_s
        elapsed = 0.0
        height_previous = position[2] - float(ground_z_m)
        max_steps = int(self.max_predict_time / max(self.rk4_dt, 1e-4))
        for _ in range(max_steps):
            previous = state.copy()
            k1 = derivative(state)
            k2 = derivative(state + 0.5 * self.rk4_dt * k1)
            k3 = derivative(state + 0.5 * self.rk4_dt * k2)
            k4 = derivative(state + self.rk4_dt * k3)
            state += (self.rk4_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            elapsed += self.rk4_dt
            height = state[2] - float(ground_z_m)
            while elapsed >= next_output and height > 0.0:
                points.append(state[0:3].copy())
                next_output += output_step_s
            if height <= 0.0 < height_previous:
                fraction = height_previous / max(1e-12, height_previous - height)
                landing = previous[0:3] + fraction * (state[0:3] - previous[0:3])
                landing[2] = float(ground_z_m)
                flight_time = elapsed - self.rk4_dt + fraction * self.rk4_dt
                if 0.02 < flight_time <= self.max_predict_time:
                    points.append(landing)
                    return points, flight_time
                return None
            height_previous = height
        return None

    def _polynomial_trajectory(self, ground_z_m, output_step_s):
        if len(self.samples) < self.poly_min_frames:
            return None
        timestamps = np.asarray([item[0] for item in self.samples], dtype=float)
        positions = np.asarray([item[1] for item in self.samples], dtype=float)
        times = timestamps - timestamps[-1]
        try:
            horizontal_x = np.polyfit(times, positions[:, 0], 1)
            horizontal_y = np.polyfit(times, positions[:, 1], 1)
            vertical = np.polyfit(times, positions[:, 2] - float(ground_z_m), 2)
        except (ValueError, np.linalg.LinAlgError):
            return None
        if vertical[0] >= 0.0:
            return None
        roots = np.roots(vertical)
        future = [float(root.real) for root in roots if abs(root.imag) < 1e-8 and root.real > 0.02]
        if not future:
            return None
        flight_time = min(future)
        vertical_linear = np.polyfit(times, positions[:, 2], 1)
        speed = math.sqrt(
            horizontal_x[0] ** 2
            + horizontal_y[0] ** 2
            + vertical_linear[0] ** 2
        )
        if flight_time > self.max_predict_time or speed < self.min_speed:
            return None
        count = max(1, int(math.ceil(flight_time / output_step_s)))
        sample_times = np.linspace(0.0, flight_time, count + 1)
        points = []
        for t in sample_times:
            points.append(
                np.array(
                    [
                        np.polyval(horizontal_x, t),
                        np.polyval(horizontal_y, t),
                        float(ground_z_m) + np.polyval(vertical, t),
                    ],
                    dtype=float,
                )
            )
        points[-1][2] = float(ground_z_m)
        return points, flight_time

    def landing(self, ground_z_m=0.0):
        prediction = self.trajectory(ground_z_m, self.rk4_dt)
        if prediction is None:
            return None
        points, flight_time = prediction
        return points[-1].copy(), flight_time

    def trajectory(self, ground_z_m=0.0, step_s=0.02):
        ballistic = self._rollout(ground_z_m, step_s)
        polynomial = self._polynomial_trajectory(ground_z_m, step_s)
        if ballistic is not None:
            confidence = 0.55 + 0.35 * float(np.clip(self.last_student_w, 0.0, 1.0))
            if polynomial is not None:
                landing_delta = float(
                    np.linalg.norm(ballistic[0][-1][0:2] - polynomial[0][-1][0:2])
                )
                confidence = (
                    max(confidence, 0.85)
                    if landing_delta < 1.0
                    else min(confidence, 0.70)
                )
            self.last_prediction_method = 'student_t_ekf_rk4'
            self.last_prediction_confidence = min(0.95, max(0.0, confidence))
            return ballistic
        if polynomial is not None:
            self.last_prediction_method = 'polynomial'
            self.last_prediction_confidence = 0.5
            return polynomial
        self.last_prediction_method = 'none'
        self.last_prediction_confidence = 0.0
        return None
