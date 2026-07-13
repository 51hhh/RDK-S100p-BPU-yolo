import math

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


class BallisticTracker:
    """Small constant-velocity Kalman filter with gravity-aware prediction."""

    def __init__(
        self,
        gravity_mps2=9.81,
        process_accel_mps2=20.0,
        innovation_gate_chi2=11.34,
        min_updates=3,
        max_predict_time_s=3.0,
    ):
        self.gravity = float(gravity_mps2)
        self.process_accel = float(process_accel_mps2)
        self.innovation_gate = float(innovation_gate_chi2)
        self.min_updates = int(min_updates)
        self.max_predict_time = float(max_predict_time_s)
        self.reset()

    def reset(self):
        self.x = np.zeros(6, dtype=float)
        self.p = np.eye(6, dtype=float)
        self.timestamp = None
        self.updates = 0

    @property
    def initialized(self):
        return self.timestamp is not None

    @property
    def ready(self):
        return self.updates >= self.min_updates

    def _predict_in_place(self, timestamp):
        dt = float(timestamp) - self.timestamp
        if dt <= 0.0:
            return False
        dt = min(dt, 0.2)
        transition = np.eye(6, dtype=float)
        transition[0:3, 3:6] = np.eye(3) * dt
        acceleration = np.array([0.0, 0.0, -self.gravity], dtype=float)
        self.x[0:3] += self.x[3:6] * dt + 0.5 * acceleration * dt * dt
        self.x[3:6] += acceleration * dt
        q = self.process_accel * self.process_accel
        process = np.zeros((6, 6), dtype=float)
        process[0:3, 0:3] = np.eye(3) * q * dt**4 / 4.0
        process[0:3, 3:6] = np.eye(3) * q * dt**3 / 2.0
        process[3:6, 0:3] = np.eye(3) * q * dt**3 / 2.0
        process[3:6, 3:6] = np.eye(3) * q * dt**2
        self.p = transition @ self.p @ transition.T + process
        self.timestamp = float(timestamp)
        return True

    def update(self, position, measurement_covariance, timestamp):
        measurement = np.asarray(position, dtype=float)
        covariance = np.asarray(measurement_covariance, dtype=float)
        if measurement.shape != (3,) or covariance.shape != (3, 3):
            return False
        if not np.all(np.isfinite(measurement)) or not np.all(np.isfinite(covariance)):
            return False
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            return False

        if not self.initialized:
            self.x[0:3] = measurement
            self.p = np.zeros((6, 6), dtype=float)
            self.p[0:3, 0:3] = covariance
            self.p[3:6, 3:6] = np.eye(3) * 25.0
            self.timestamp = timestamp
            self.updates = 1
            return True

        if timestamp <= self.timestamp or not self._predict_in_place(timestamp):
            return False

        innovation = measurement - self.x[0:3]
        innovation_covariance = self.p[0:3, 0:3] + covariance
        try:
            inverse = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            return False
        mahalanobis = float(innovation.T @ inverse @ innovation)
        if not math.isfinite(mahalanobis) or mahalanobis > self.innovation_gate:
            return False

        gain = self.p[:, 0:3] @ inverse
        identity = np.eye(6, dtype=float)
        observation = np.zeros((3, 6), dtype=float)
        observation[:, 0:3] = np.eye(3)
        self.x += gain @ innovation
        residual = identity - gain @ observation
        self.p = residual @ self.p @ residual.T + gain @ covariance @ gain.T
        self.updates += 1
        return True

    def landing(self, ground_z_m=0.0):
        if not self.ready:
            return None
        height = float(self.x[2] - ground_z_m)
        vertical_speed = float(self.x[5])
        discriminant = vertical_speed * vertical_speed + 2.0 * self.gravity * height
        if height <= 0.0 or discriminant < 0.0:
            return None
        flight_time = (vertical_speed + math.sqrt(discriminant)) / self.gravity
        if not 0.02 < flight_time <= self.max_predict_time:
            return None
        point = self.x[0:3] + self.x[3:6] * flight_time
        point[2] = float(ground_z_m)
        if not np.all(np.isfinite(point)):
            return None
        return point, flight_time
