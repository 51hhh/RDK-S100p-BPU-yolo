from dataclasses import dataclass
import csv
import math


SERVO_UNKNOWN = 0
SERVO_UNLOCKED = 1
SERVO_LOCKED = 2
SERVO_HOLDOVER = 3


def sequence_is_newer(sequence, previous, bits=64):
    sequence = int(sequence)
    previous = int(previous)
    modulus = 1 << bits
    delta = (sequence - previous) % modulus
    return 0 < delta < modulus // 2


@dataclass(frozen=True)
class SyncSample:
    source_epoch: int
    sequence: int
    sample_time_nx_s: float
    offset_s: float
    uncertainty_s: float
    synchronized: bool
    servo_state: int = SERVO_UNKNOWN
    clock_source: str = ''


@dataclass(frozen=True)
class SyncIngestResult:
    accepted: bool
    reason: str
    clock_step: bool = False


class TimeSyncGuard:
    def __init__(
        self,
        timeout_s=0.5,
        max_future_skew_s=0.02,
        warn_offset_s=0.005,
        reject_offset_s=0.020,
        max_uncertainty_s=0.002,
        max_offset_step_s=0.002,
    ):
        self.timeout_s = float(timeout_s)
        self.max_future_skew_s = float(max_future_skew_s)
        self.warn_offset_s = float(warn_offset_s)
        self.reject_offset_s = float(reject_offset_s)
        self.max_uncertainty_s = float(max_uncertainty_s)
        self.max_offset_step_s = float(max_offset_step_s)
        self.sample = None
        self.receive_time_s = -math.inf
        self.corrected_sample_time_s = -math.inf
        self.retired_epochs = []

    def ingest(self, sample, receive_time_s):
        receive_time_s = float(receive_time_s)
        values = (
            sample.sample_time_nx_s,
            sample.offset_s,
            sample.uncertainty_s,
            receive_time_s,
        )
        if sample.source_epoch == 0 or sample.sequence == 0 or not all(
            math.isfinite(value) for value in values
        ):
            return SyncIngestResult(False, 'invalid sync fields')
        if sample.uncertainty_s < 0.0:
            return SyncIngestResult(False, 'negative sync uncertainty')
        if sample.source_epoch in self.retired_epochs:
            return SyncIngestResult(False, 'replayed retired sync epoch')
        corrected_sample_time = sample.sample_time_nx_s - sample.offset_s
        age = receive_time_s - corrected_sample_time
        if age < -self.max_future_skew_s or age > self.timeout_s:
            return SyncIngestResult(False, 'stale or future sync measurement')

        same_epoch = self.sample is not None and (
            sample.source_epoch == self.sample.source_epoch
        )
        if same_epoch:
            if not sequence_is_newer(sample.sequence, self.sample.sequence):
                return SyncIngestResult(False, 'replayed sync sequence')
            if corrected_sample_time <= self.corrected_sample_time_s:
                return SyncIngestResult(False, 'regressing sync measurement time')
        clock_step = bool(
            same_epoch
            and self.sample.synchronized
            and sample.synchronized
            and abs(sample.offset_s - self.sample.offset_s) > self.max_offset_step_s
        )
        if self.sample is not None and not same_epoch:
            self.retired_epochs.append(self.sample.source_epoch)
            self.retired_epochs = self.retired_epochs[-8:]
        self.sample = sample
        self.receive_time_s = receive_time_s
        self.corrected_sample_time_s = corrected_sample_time
        return SyncIngestResult(True, 'accepted', clock_step)

    def status(self, source_epoch, now_s):
        now_s = float(now_s)
        if self.sample is None:
            return False, 'time sync unavailable'
        if self.sample.source_epoch != int(source_epoch):
            return False, 'time sync epoch mismatch'
        if now_s - self.receive_time_s > self.timeout_s:
            return False, 'time sync receive timeout'
        sample_age = now_s - self.corrected_sample_time_s
        if sample_age < -self.max_future_skew_s or sample_age > self.timeout_s:
            return False, 'time sync measurement timeout'
        if not self.sample.synchronized or self.sample.servo_state != SERVO_LOCKED:
            return False, 'time sync servo unlocked'
        if abs(self.sample.offset_s) > self.reject_offset_s:
            return False, 'time sync offset rejected'
        if self.sample.uncertainty_s > self.max_uncertainty_s:
            return False, 'time sync uncertainty rejected'
        return True, 'valid'

    def corrected_time(self, nx_time_s, source_epoch, now_s):
        valid, reason = self.status(source_epoch, now_s)
        if not valid:
            return None, reason
        return float(nx_time_s) - self.sample.offset_s, 'valid'

    def trust_scale(self):
        if self.sample is None:
            return 0.0
        magnitude = abs(self.sample.offset_s)
        if magnitude <= self.warn_offset_s:
            return 1.0
        span = max(1e-9, self.reject_offset_s - self.warn_offset_s)
        return max(0.10, min(1.0, (self.reject_offset_s - magnitude) / span))


@dataclass(frozen=True)
class ChronyTracking:
    offset_ns: int
    uncertainty_ns: int
    synchronized: bool
    servo_state: int
    clock_source: str


def parse_chrony_tracking_csv(text, offset_sign=1.0):
    rows = list(csv.reader(line for line in text.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) < 13:
        raise ValueError('chronyc tracking CSV must contain at least 13 fields')
    fields = [field.strip() for field in rows[0]]
    stratum = int(fields[1])
    system_offset_s = float(fields[3]) * float(offset_sign)
    last_offset_s = abs(float(fields[4]))
    rms_offset_s = abs(float(fields[5]))
    root_delay_s = abs(float(fields[9]))
    root_dispersion_s = abs(float(fields[10]))
    leap = fields[12].lower()
    leap_normal = leap in ('normal', '0')
    synchronized = 0 < stratum < 16 and leap_normal
    uncertainty_s = max(
        rms_offset_s,
        last_offset_s,
        root_dispersion_s + 0.5 * root_delay_s,
    )
    reference = fields[0] or 'unknown'
    return ChronyTracking(
        offset_ns=int(round(system_offset_s * 1e9)),
        uncertainty_ns=max(0, int(math.ceil(uncertainty_s * 1e9))),
        synchronized=synchronized,
        servo_state=SERVO_LOCKED if synchronized else SERVO_UNLOCKED,
        clock_source=f'chrony:{reference}',
    )
