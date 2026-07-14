import argparse
from bisect import bisect_right
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import yaml

from .throw_analysis import (
    ObservationSample,
    PredictionSample,
    TrialGroundTruth,
    analyze_session,
)


def default_config_path():
    source = Path(__file__).resolve().parents[1] / 'config' / 'd435_throw_analysis.yaml'
    if source.exists():
        return source
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory('volleyball_catch_controller')) / 'config' / source.name
    except Exception:
        return source


def message_time_s(message, fallback_ns):
    if hasattr(message, 'header') and hasattr(message.header, 'stamp'):
        stamp = message.header.stamp
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if value > 0.0:
            return value
    return float(fallback_ns) * 1e-9


def nearest_sample(samples, timestamps, timestamp_s, max_delta_s):
    insertion = bisect_right(timestamps, timestamp_s)
    candidates = [
        index for index in (insertion - 1, insertion)
        if 0 <= index < len(samples)
    ]
    if not candidates:
        return None
    index = min(candidates, key=lambda value: abs(timestamps[value] - timestamp_s))
    if abs(timestamps[index] - timestamp_s) > max_delta_s:
        return None
    return samples[index]


def matched_state_tti(landing_timestamp_s, state):
    if state is None or not state[2] or state[3] != 2 or not math.isfinite(state[1]):
        return None
    # CatchState may be published just before or after /ball/landing. Convert
    # its remaining time to the landing message timestamp instead of treating
    # both messages as simultaneous.
    return float(state[1]) + float(state[0]) - float(landing_timestamp_s)


def same_held_prediction(previous, current, position_epsilon_m=1e-9, impact_epsilon_s=0.005):
    if previous is None:
        return False
    if math.hypot(previous.x_m - current.x_m, previous.y_m - current.y_m) > position_epsilon_m:
        return False
    if previous.predicted_tti_s is None or current.predicted_tti_s is None:
        return True
    previous_impact = previous.timestamp_s + previous.predicted_tti_s
    current_impact = current.timestamp_s + current.predicted_tti_s
    return abs(previous_impact - current_impact) <= impact_epsilon_s


def read_bag(bag_path, storage_id, config):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            'ROS2 Python bag support is unavailable; source /opt/ros/humble/setup.bash first'
        ) from exc

    topics = config['topics']
    selected = {
        topics['observation'],
        topics['landing'],
        topics['state'],
        topics['goal'],
    }
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr'
        ),
    )
    topic_types = {info.name: info.type for info in reader.get_all_topics_and_types()}
    missing = sorted(topic for topic in selected if topic not in topic_types)
    if topics['observation'] in missing:
        raise RuntimeError(f"bag does not contain required topic {topics['observation']}")
    message_types = {
        topic: get_message(topic_types[topic]) for topic in selected if topic in topic_types
    }

    observations = []
    landings = []
    states = []
    goals = []
    bag_start_s = None
    bag_end_s = None
    timing = config['timing']

    while reader.has_next():
        topic, serialized, record_ns = reader.read_next()
        record_s = float(record_ns) * 1e-9
        bag_start_s = record_s if bag_start_s is None else min(bag_start_s, record_s)
        bag_end_s = record_s if bag_end_s is None else max(bag_end_s, record_s)
        if topic not in message_types:
            continue
        message = deserialize_message(serialized, message_types[topic])
        timestamp_s = message_time_s(message, record_ns)
        if topic == topics['observation']:
            timing_valid = bool(message.timestamp_mapping_valid) and (
                abs(int(message.rgb_depth_timestamp_delta_ns))
                <= int(timing['max_rgb_depth_delta_ns'])
                and int(message.timestamp_uncertainty_ns)
                <= int(timing['max_timestamp_uncertainty_ns'])
            )
            observations.append(
                ObservationSample(
                    timestamp_s=timestamp_s,
                    receive_time_s=record_s,
                    detection_valid=bool(message.detection_valid),
                    rgbd_valid=bool(message.rgbd_valid),
                    timing_valid=timing_valid,
                    confidence=float(message.detection_confidence),
                    valid_depth_samples=int(message.valid_depth_samples),
                    depth_spread_m=float(message.depth_spread_m),
                    source_epoch=int(message.source_epoch),
                    frame_id=int(message.frame_id),
                    position_x_m=float(message.position.x),
                    position_y_m=float(message.position.y),
                    position_z_m=float(message.position.z),
                    bbox_width_px=float(message.bbox_xyxy[2] - message.bbox_xyxy[0]),
                    bbox_height_px=float(message.bbox_xyxy[3] - message.bbox_xyxy[1]),
                )
            )
        elif topic == topics['landing']:
            landings.append(
                (timestamp_s, float(message.point.x), float(message.point.y))
            )
        elif topic == topics['state']:
            states.append(
                (
                    timestamp_s,
                    float(message.time_to_impact_s),
                    bool(message.goal_valid),
                    int(message.active_source),
                )
            )
        elif topic == topics['goal']:
            goals.append(
                (
                    timestamp_s,
                    math.hypot(float(message.pose.position.x), float(message.pose.position.y)),
                )
            )

    if bag_start_s is None or bag_end_s is None:
        raise RuntimeError('bag contains no messages')
    states.sort(key=lambda item: item[0])
    goals.sort(key=lambda item: item[0])
    state_times = [item[0] for item in states]
    goal_times = [item[0] for item in goals]
    match_age = float(config['analysis']['state_goal_match_max_age_s'])
    predictions = []
    for timestamp_s, x_m, y_m in sorted(landings):
        state = nearest_sample(states, state_times, timestamp_s, match_age)
        goal = nearest_sample(goals, goal_times, timestamp_s, match_age)
        predicted_tti = matched_state_tti(timestamp_s, state)
        candidate = PredictionSample(
            timestamp_s=timestamp_s,
            x_m=x_m,
            y_m=y_m,
            predicted_tti_s=predicted_tti,
            goal_distance_m=goal[1] if goal is not None else None,
        )
        if not same_held_prediction(predictions[-1] if predictions else None, candidate):
            predictions.append(candidate)
    return observations, predictions, bag_start_s, bag_end_s, sorted(topic_types), missing


def optional_float(value):
    text = str(value or '').strip()
    if not text:
        return None
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f'ground truth value must be finite, got {text!r}')
    return number


def read_ground_truth(path, reference, bag_start_s):
    trials = []
    with Path(path).open('r', encoding='utf-8', newline='') as stream:
        rows = csv.DictReader(line for line in stream if not line.lstrip().startswith('#'))
        required = {'trial_id', 'start_s', 'impact_s', 'impact_x_m', 'impact_y_m'}
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError('ground truth CSV columns must be: ' + ','.join(sorted(required)))
        for row in rows:
            start_s = float(row['start_s'])
            impact_s = float(row['impact_s'])
            if reference == 'relative':
                start_s += bag_start_s
                impact_s += bag_start_s
            if impact_s <= start_s:
                raise ValueError(f"trial {row['trial_id']}: impact_s must be after start_s")
            trials.append(
                TrialGroundTruth(
                    trial_id=str(row['trial_id']).strip(),
                    start_s=start_s,
                    end_s=impact_s,
                    impact_s=impact_s,
                    impact_x_m=optional_float(row['impact_x_m']),
                    impact_y_m=optional_float(row['impact_y_m']),
                )
            )
    if not trials:
        raise ValueError('ground truth CSV contains no trials')
    return trials


def write_csv_exports(directory, observations, predictions, bag_start_s):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'd435_observations.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = list(asdict(observations[0]).keys()) if observations else [
            'timestamp_s', 'receive_time_s', 'detection_valid', 'rgbd_valid',
            'timing_valid', 'confidence', 'valid_depth_samples', 'depth_spread_m',
            'source_epoch', 'frame_id', 'position_x_m', 'position_y_m',
            'position_z_m', 'bbox_width_px', 'bbox_height_px',
        ]
        writer = csv.DictWriter(stream, fieldnames=['relative_time_s', *fields])
        writer.writeheader()
        for sample in observations:
            writer.writerow({'relative_time_s': sample.timestamp_s - bag_start_s, **asdict(sample)})
    with (directory / 'landing_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = list(asdict(predictions[0]).keys()) if predictions else [
            'timestamp_s', 'x_m', 'y_m', 'predicted_tti_s', 'goal_distance_m',
        ]
        writer = csv.DictWriter(stream, fieldnames=['relative_time_s', *fields])
        writer.writeheader()
        for sample in predictions:
            writer.writerow({'relative_time_s': sample.timestamp_s - bag_start_s, **asdict(sample)})


def format_value(value, digits=3):
    return 'n/a' if value is None or not math.isfinite(float(value)) else f'{float(value):.{digits}f}'


def print_report(report):
    summary = report['summary']
    print('\nD435 throw analysis')
    print('=' * 72)
    for item in report['trials']:
        perception = item['perception']
        prediction = item['prediction']
        print(
            f"{item['trial']['trial_id']}: {item['verdict']} "
            f"obs={item['counts']['observations']} pred={item['counts']['predictions']} "
            f"det={format_value(perception['detection_rate'])} "
            f"rgbd={format_value(perception['rgbd_rate'])} "
            f"lead={format_value(prediction['first_prediction_lead_s'])}s "
            f"landing_p90={format_value(prediction['landing_error_p90_m'])}m"
        )
        failed = [check['name'] for check in item['checks'] if check['status'] == 'FAIL']
        if failed:
            print('  failed: ' + ', '.join(failed))
    print('-' * 72)
    print(
        f"overall={summary['verdict']} trials={summary['trial_count']} "
        f"pass={summary['pass_count']} fail={summary['fail_count']} "
        f"incomplete={summary['incomplete_count']}"
    )


def main():
    parser = argparse.ArgumentParser(description='Analyze D435 throw rosbag data')
    parser.add_argument('--bag', '-b', required=True, help='rosbag2 directory')
    parser.add_argument('--config', '-c', default=str(default_config_path()))
    parser.add_argument('--ground-truth', '-g', help='CSV containing trial impact truth')
    parser.add_argument(
        '--time-reference', choices=('relative', 'absolute'), default='relative',
        help='ground-truth timestamps relative to bag start or absolute ROS time',
    )
    parser.add_argument('--target-radius', type=float, help='effective ball-center tolerance in meters')
    parser.add_argument('--storage', default='sqlite3')
    parser.add_argument('--output', '-o', help='write JSON report')
    parser.add_argument('--csv-dir', help='export decoded observation/prediction CSV files')
    args = parser.parse_args()

    try:
        with Path(args.config).open('r', encoding='utf-8') as stream:
            config = yaml.safe_load(stream) or {}
        observations, predictions, bag_start, bag_end, bag_topics, missing = read_bag(
            Path(args.bag), args.storage, config
        )
        trials = (
            read_ground_truth(args.ground_truth, args.time_reference, bag_start)
            if args.ground_truth
            else [TrialGroundTruth('session', bag_start, bag_end)]
        )
        thresholds = dict(config['thresholds'])
        if args.target_radius is not None:
            if args.target_radius <= 0.0:
                raise ValueError('--target-radius must be positive')
            thresholds['effective_target_radius_m'] = args.target_radius
        report = analyze_session(trials, observations, predictions, thresholds)
        report['bag'] = {
            'path': str(Path(args.bag).resolve()),
            'start_s': bag_start,
            'end_s': bag_end,
            'duration_s': bag_end - bag_start,
            'topics': bag_topics,
            'configured_topics_missing': missing,
        }
        report['thresholds'] = thresholds
        print_report(report)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'JSON report: {output}')
        if args.csv_dir:
            write_csv_exports(args.csv_dir, observations, predictions, bag_start)
            print(f'CSV export: {args.csv_dir}')
        return 1 if report['summary']['verdict'] == 'FAIL' else 0
    except Exception as exc:
        print(f'analyze_d435_throw_bag: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
