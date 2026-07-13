#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(image: np.ndarray, size: int = 640, padding: int = 127) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    top = (size - resized_height) // 2
    bottom = size - resized_height - top
    left = (size - resized_width) // 2
    right = size - resized_width - left
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(padding, padding, padding),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate S100 YOLO calibration tensors")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images = sorted(
        path for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit(f"No supported images found under {args.input_dir}")

    if len(images) > args.count:
        images = random.Random(args.seed).sample(images, args.count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in args.output_dir.glob("*.npy"):
        old_file.unlink()

    written = 0
    for image_path in images:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Skip unreadable image: {image_path}")
            continue
        rgb = cv2.cvtColor(letterbox(bgr), cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        np.save(args.output_dir / f"calibration_{written:04d}.npy", tensor)
        written += 1

    if written == 0:
        raise SystemExit("No calibration tensors were generated")
    sample = np.load(args.output_dir / "calibration_0000.npy")
    print(f"Generated {written} tensors in {args.output_dir}")
    print(f"Sample: shape={sample.shape}, dtype={sample.dtype}, range=[{sample.min():.6f}, {sample.max():.6f}]")


if __name__ == "__main__":
    main()
