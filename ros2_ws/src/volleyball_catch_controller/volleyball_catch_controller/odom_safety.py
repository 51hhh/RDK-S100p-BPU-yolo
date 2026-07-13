from collections import deque
from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class OdomDecision:
    accepted: bool
    reason: str


class OdomContractGuard:
    """Validate the timestamp/frame contract of an external chassis odometry stream."""

    def __init__(
        self,
        world_frame='odom',
        base_frame='base_link',
        max_message_age_s=0.05,
        max_future_skew_s=0.02,
        rate_window=100,
    ):
        self.world_frame = str(world_frame)
        self.base_frame = str(base_frame)
        self.max_message_age_s = float(max_message_age_s)
        self.max_future_skew_s = float(max_future_skew_s)
        self.last_timestamp = -math.inf
        self.last_accept_receive_s = -math.inf
        self.intervals = deque(maxlen=max(2, int(rate_window)))
        self.accepted = 0
        self.rejected = 0
        self.last_reason = 'no odometry received'

    def accept(
        self,
        timestamp,
        receive_time_s,
        frame_id,
        child_frame_id,
        position,
        quaternion,
    ):
        timestamp = float(timestamp)
        receive_time_s = float(receive_time_s)
        values = [timestamp, receive_time_s, *position, *quaternion]
        reason = None
        if not all(math.isfinite(float(value)) for value in values):
            reason = 'odom contains non-finite values'
        elif timestamp <= 0.0:
            reason = 'odom timestamp is zero'
        elif str(frame_id) != self.world_frame:
            reason = f'odom frame_id must be {self.world_frame}'
        elif str(child_frame_id) != self.base_frame:
            reason = f'odom child_frame_id must be {self.base_frame}'
        else:
            age = receive_time_s - timestamp
            if age < -self.max_future_skew_s:
                reason = 'odom timestamp is in the future'
            elif age > self.max_message_age_s:
                reason = 'odom message is stale'
            elif timestamp <= self.last_timestamp:
                reason = 'odom timestamp is not monotonic'
            else:
                quaternion_norm = math.sqrt(
                    sum(float(value) * float(value) for value in quaternion)
                )
                if quaternion_norm < 1e-6:
                    reason = 'odom quaternion is invalid'

        if reason is not None:
            self.rejected += 1
            self.last_reason = reason
            return OdomDecision(False, reason)

        if math.isfinite(self.last_timestamp):
            self.intervals.append(timestamp - self.last_timestamp)
        self.last_timestamp = timestamp
        self.last_accept_receive_s = receive_time_s
        self.accepted += 1
        self.last_reason = 'valid'
        return OdomDecision(True, 'valid')

    @property
    def rate_hz(self):
        valid = [interval for interval in self.intervals if interval > 0.0]
        if not valid:
            return 0.0
        return 1.0 / statistics.median(valid)

    def valid_at(self, now_s, timeout_s):
        return (
            self.accepted > 0
            and float(now_s) - self.last_accept_receive_s <= float(timeout_s)
        )
