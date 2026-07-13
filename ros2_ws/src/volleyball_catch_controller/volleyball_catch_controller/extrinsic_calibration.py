import argparse
import json
import math

import numpy as np


def rotation_matrix_to_quaternion(rotation):
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=float,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ],
                dtype=float,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ],
                dtype=float,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=float,
            )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        return None
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def solve_rigid_transform(camera_points, base_points):
    camera = np.asarray(camera_points, dtype=float)
    base = np.asarray(base_points, dtype=float)
    if (
        camera.ndim != 2
        or camera.shape[1:] != (3,)
        or base.shape != camera.shape
        or camera.shape[0] < 3
        or not np.all(np.isfinite(camera))
        or not np.all(np.isfinite(base))
    ):
        raise ValueError('at least three finite camera/base 3D point pairs are required')
    camera_centered = camera - np.mean(camera, axis=0)
    base_centered = base - np.mean(base, axis=0)
    if np.linalg.matrix_rank(camera_centered, tol=1e-7) < 2:
        raise ValueError('calibration points must not be collinear')
    covariance = camera_centered.T @ base_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    translation = np.mean(base, axis=0) - rotation @ np.mean(camera, axis=0)
    residuals = base - (camera @ rotation.T + translation)
    errors = np.linalg.norm(residuals, axis=1)
    quaternion = rotation_matrix_to_quaternion(rotation)
    return {
        'rotation': rotation,
        'translation': translation,
        'quaternion': quaternion,
        'rmse_m': float(math.sqrt(np.mean(errors * errors))),
        'max_error_m': float(np.max(errors)),
        'errors_m': errors,
    }


def load_correspondences(path):
    samples = []
    with open(path, encoding='utf-8') as stream:
        content = stream.read().strip()
    if not content:
        raise ValueError('calibration sample file is empty')
    try:
        parsed = json.loads(content)
        samples = parsed['samples'] if isinstance(parsed, dict) else parsed
    except json.JSONDecodeError:
        samples = [json.loads(line) for line in content.splitlines() if line.strip()]
    camera = [sample['camera'] for sample in samples]
    base = [sample['base'] for sample in samples]
    return camera, base


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Solve T_base_d435 from measured volleyball-center point pairs.'
    )
    parser.add_argument('samples', help='JSON or JSONL correspondence file')
    parser.add_argument('--max-rmse-m', type=float, default=0.020)
    options = parser.parse_args(args=args)
    camera, base = load_correspondences(options.samples)
    result = solve_rigid_transform(camera, base)
    output = {
        'sample_count': len(camera),
        'd435_camera_translation': result['translation'].tolist(),
        'd435_camera_quaternion': result['quaternion'].tolist(),
        'rmse_m': result['rmse_m'],
        'max_error_m': result['max_error_m'],
        'accepted': result['rmse_m'] <= max(0.0, options.max_rmse_m),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output['accepted']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
