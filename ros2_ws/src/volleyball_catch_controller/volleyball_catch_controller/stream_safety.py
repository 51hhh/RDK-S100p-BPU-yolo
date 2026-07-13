from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StreamDecision:
    accepted: bool
    new_epoch: bool
    reason: str


class StreamOrderGuard:
    """Reject replayed frames while allowing a restarted publisher to start at frame zero."""

    def __init__(self):
        self.epoch = 0
        self.timestamp = -math.inf
        self.frame_id = -1
        self.retired_epochs = []

    def reset(self):
        self.epoch = 0
        self.timestamp = -math.inf
        self.frame_id = -1
        self.retired_epochs = []

    def accept(self, epoch, timestamp, frame_id):
        epoch = int(epoch)
        timestamp = float(timestamp)
        frame_id = int(frame_id)
        if epoch == 0 or not math.isfinite(timestamp) or frame_id < 0:
            return StreamDecision(False, False, 'invalid stream identity')
        if epoch in self.retired_epochs:
            return StreamDecision(False, False, 'replayed retired source epoch')
        new_epoch = epoch != self.epoch
        if new_epoch:
            if self.epoch != 0:
                self.retired_epochs.append(self.epoch)
                self.retired_epochs = self.retired_epochs[-8:]
            self.epoch = epoch
            self.timestamp = -math.inf
            self.frame_id = -1
        if timestamp <= self.timestamp:
            return StreamDecision(False, new_epoch, 'replayed or regressing timestamp')
        if frame_id != 0 and frame_id <= self.frame_id:
            return StreamDecision(False, new_epoch, 'replayed frame id')
        self.timestamp = timestamp
        self.frame_id = frame_id
        return StreamDecision(True, new_epoch, 'accepted')


def d435_timing_valid(
    rgb_depth_delta_ns,
    timestamp_uncertainty_ns,
    mapping_valid,
    max_delta_ns,
    max_uncertainty_ns,
):
    return bool(
        mapping_valid
        and abs(int(rgb_depth_delta_ns)) <= int(max_delta_ns)
        and 0 <= int(timestamp_uncertainty_ns) <= int(max_uncertainty_ns)
    )
