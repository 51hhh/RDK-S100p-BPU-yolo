from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class ObservationSample:
    timestamp_s: float
    receive_time_s: float
    detection_valid: bool
    rgbd_valid: bool
    timing_valid: bool
    confidence: float = math.nan
    valid_depth_samples: int = 0
    depth_spread_m: float = math.nan
    source_epoch: int = 0
    frame_id: int = 0
    position_x_m: float = math.nan
    position_y_m: float = math.nan
    position_z_m: float = math.nan
    bbox_width_px: float = math.nan
    bbox_height_px: float = math.nan


@dataclass(frozen=True)
class PredictionSample:
    timestamp_s: float
    x_m: float
    y_m: float
    predicted_tti_s: Optional[float] = None
    goal_distance_m: Optional[float] = None


@dataclass(frozen=True)
class TrialGroundTruth:
    trial_id: str
    start_s: float
    end_s: float
    impact_s: Optional[float] = None
    impact_x_m: Optional[float] = None
    impact_y_m: Optional[float] = None


DEFAULT_THRESHOLDS = {
    'min_detection_rate': 0.95,
    'min_rgbd_rate': 0.90,
    'min_timing_valid_rate': 0.99,
    'max_frame_gap_p99_s': 0.050,
    'min_first_prediction_lead_s': 0.80,
    'max_impact_time_error_p90_s': 0.050,
    'effective_target_radius_m': None,
    'max_linear_speed_mps': 0.40,
    'max_linear_accel_mps2': 0.80,
    'braking_margin_s': 0.15,
    'tti_bins_s': [1.0, 0.8, 0.6, 0.4, 0.25],
    'tti_bin_half_width_s': 0.08,
}


def percentile(values: Iterable[float], percent: float) -> Optional[float]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = max(0.0, min(100.0, float(percent))) * (len(finite) - 1) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def rate(samples: Sequence, predicate) -> Optional[float]:
    if not samples:
        return None
    return sum(1 for sample in samples if predicate(sample)) / len(samples)


def reachable_distance_from_rest(
    time_s: float,
    max_speed_mps: float,
    max_accel_mps2: float,
    braking_margin_s: float = 0.0,
) -> float:
    available = max(0.0, float(time_s) - max(0.0, float(braking_margin_s)))
    speed = max(0.0, float(max_speed_mps))
    accel = max(0.0, float(max_accel_mps2))
    if available <= 0.0 or speed <= 0.0 or accel <= 0.0:
        return 0.0
    acceleration_time = speed / accel
    if available <= acceleration_time:
        return 0.5 * accel * available * available
    return 0.5 * accel * acceleration_time * acceleration_time + speed * (
        available - acceleration_time
    )


def _metric_check(name: str, value: Optional[float], threshold: Optional[float], minimum: bool):
    if threshold is None:
        return {'name': name, 'status': 'NOT_CONFIGURED', 'value': value, 'threshold': None}
    if value is None:
        return {'name': name, 'status': 'NO_DATA', 'value': None, 'threshold': threshold}
    passed = value >= threshold if minimum else value <= threshold
    return {
        'name': name,
        'status': 'PASS' if passed else 'FAIL',
        'value': value,
        'threshold': threshold,
    }


def _trial_verdict(checks: Sequence[Dict]) -> str:
    statuses = [check['status'] for check in checks]
    if 'FAIL' in statuses:
        return 'FAIL'
    if statuses and all(status == 'PASS' for status in statuses):
        return 'PASS'
    return 'INCOMPLETE'


def analyze_trial(
    trial: TrialGroundTruth,
    observations: Sequence[ObservationSample],
    predictions: Sequence[PredictionSample],
    thresholds: Optional[Dict] = None,
) -> Dict:
    limits = dict(DEFAULT_THRESHOLDS)
    limits.update(thresholds or {})
    obs = sorted(
        (sample for sample in observations if trial.start_s <= sample.timestamp_s <= trial.end_s),
        key=lambda sample: sample.timestamp_s,
    )
    pred = sorted(
        (sample for sample in predictions if trial.start_s <= sample.timestamp_s <= trial.end_s),
        key=lambda sample: sample.timestamp_s,
    )

    frame_gaps = [
        current.timestamp_s - previous.timestamp_s
        for previous, current in zip(obs, obs[1:])
        if current.timestamp_s > previous.timestamp_s
    ]
    latencies = [sample.receive_time_s - sample.timestamp_s for sample in obs]
    confidences = [sample.confidence for sample in obs if sample.detection_valid]
    spreads = [sample.depth_spread_m for sample in obs if sample.rgbd_valid]

    landing_errors = []
    impact_time_errors = []
    actual_impact = (
        trial.impact_s is not None
        and trial.impact_x_m is not None
        and trial.impact_y_m is not None
    )
    if actual_impact:
        for sample in pred:
            landing_errors.append(
                math.hypot(sample.x_m - trial.impact_x_m, sample.y_m - trial.impact_y_m)
            )
            if sample.predicted_tti_s is not None and math.isfinite(sample.predicted_tti_s):
                actual_tti = trial.impact_s - sample.timestamp_s
                impact_time_errors.append(sample.predicted_tti_s - actual_tti)

    first_prediction_lead = None
    first_goal_distance = None
    reachable_distance = None
    first_prediction = pred[0] if pred else None
    if first_prediction is not None and trial.impact_s is not None:
        first_prediction_lead = trial.impact_s - first_prediction.timestamp_s
        first_goal_distance = first_prediction.goal_distance_m
        reachable_distance = reachable_distance_from_rest(
            first_prediction_lead,
            limits['max_linear_speed_mps'],
            limits['max_linear_accel_mps2'],
            limits['braking_margin_s'],
        )

    tti_errors = {}
    if actual_impact:
        half_width = float(limits['tti_bin_half_width_s'])
        for target in limits['tti_bins_s']:
            candidates = [
                (
                    abs((trial.impact_s - sample.timestamp_s) - float(target)),
                    math.hypot(sample.x_m - trial.impact_x_m, sample.y_m - trial.impact_y_m),
                )
                for sample in pred
                if abs((trial.impact_s - sample.timestamp_s) - float(target)) <= half_width
            ]
            tti_errors[f'{float(target):.2f}'] = min(candidates)[1] if candidates else None

    checks = [
        _metric_check(
            'detection_rate', rate(obs, lambda sample: sample.detection_valid),
            limits['min_detection_rate'] if trial.impact_s is not None else None, True,
        ),
        _metric_check(
            'rgbd_rate', rate(obs, lambda sample: sample.rgbd_valid),
            limits['min_rgbd_rate'] if trial.impact_s is not None else None, True,
        ),
        _metric_check(
            'timing_valid_rate', rate(obs, lambda sample: sample.timing_valid),
            limits['min_timing_valid_rate'], True,
        ),
        _metric_check(
            'frame_gap_p99_s', percentile(frame_gaps, 99),
            limits['max_frame_gap_p99_s'], False,
        ),
        _metric_check(
            'first_prediction_lead_s', first_prediction_lead,
            limits['min_first_prediction_lead_s'], True,
        ),
        _metric_check(
            'landing_error_p90_m', percentile(landing_errors, 90),
            limits['effective_target_radius_m'], False,
        ),
        _metric_check(
            'impact_time_abs_error_p90_s',
            percentile((abs(value) for value in impact_time_errors), 90),
            limits['max_impact_time_error_p90_s'], False,
        ),
    ]
    if first_goal_distance is not None:
        checks.append(
            _metric_check(
                'first_goal_reachable_from_rest',
                first_goal_distance,
                reachable_distance,
                False,
            )
        )

    return {
        'trial': asdict(trial),
        'counts': {
            'observations': len(obs),
            'detections': sum(sample.detection_valid for sample in obs),
            'rgbd_valid': sum(sample.rgbd_valid for sample in obs),
            'predictions': len(pred),
        },
        'perception': {
            'detection_rate': rate(obs, lambda sample: sample.detection_valid),
            'rgbd_rate': rate(obs, lambda sample: sample.rgbd_valid),
            'timing_valid_rate': rate(obs, lambda sample: sample.timing_valid),
            'frame_gap_p50_s': percentile(frame_gaps, 50),
            'frame_gap_p99_s': percentile(frame_gaps, 99),
            'capture_to_bag_latency_p50_s': percentile(latencies, 50),
            'capture_to_bag_latency_p95_s': percentile(latencies, 95),
            'confidence_p10': percentile(confidences, 10),
            'depth_spread_p90_m': percentile(spreads, 90),
        },
        'prediction': {
            'first_prediction_lead_s': first_prediction_lead,
            'landing_error_p50_m': percentile(landing_errors, 50),
            'landing_error_p90_m': percentile(landing_errors, 90),
            'impact_time_error_p50_s': percentile(impact_time_errors, 50),
            'impact_time_abs_error_p90_s': percentile(
                (abs(value) for value in impact_time_errors), 90
            ),
            'landing_error_by_tti_m': tti_errors,
        },
        'reachability': {
            'first_goal_distance_m': first_goal_distance,
            'reachable_distance_from_rest_m': reachable_distance,
            'model': 'speed/acceleration limited, braking margin reserved',
        },
        'checks': checks,
        'verdict': _trial_verdict(checks),
    }


def analyze_session(
    trials: Sequence[TrialGroundTruth],
    observations: Sequence[ObservationSample],
    predictions: Sequence[PredictionSample],
    thresholds: Optional[Dict] = None,
) -> Dict:
    reports = [analyze_trial(trial, observations, predictions, thresholds) for trial in trials]
    verdicts = [report['verdict'] for report in reports]
    overall = (
        'FAIL' if 'FAIL' in verdicts
        else 'PASS' if verdicts and all(value == 'PASS' for value in verdicts)
        else 'INCOMPLETE'
    )
    return {
        'summary': {
            'trial_count': len(reports),
            'pass_count': verdicts.count('PASS'),
            'fail_count': verdicts.count('FAIL'),
            'incomplete_count': verdicts.count('INCOMPLETE'),
            'verdict': overall,
        },
        'trials': reports,
    }
