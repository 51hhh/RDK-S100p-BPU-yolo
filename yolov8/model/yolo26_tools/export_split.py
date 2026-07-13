#!/usr/bin/env python3
import argparse

from ultralytics import YOLO
from ultralytics.nn.modules import Detect


def split_forward(self, features):
    outputs = []
    for index in range(self.nl):
        cls = self.cv3[index](features[index]).permute(0, 2, 3, 1).contiguous()
        box = self.cv2[index](features[index]).permute(0, 2, 3, 1).contiguous()
        outputs.extend((cls, box))
    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a six-output YOLO26 ONNX for RDK S100P")
    parser.add_argument("model")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    Detect.forward = split_forward
    model = YOLO(args.model)
    result = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=1,
        opset=19,
        dynamic=False,
        simplify=True,
        nms=False,
        device="cpu",
    )
    print(result)


if __name__ == "__main__":
    main()
