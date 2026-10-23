"""Exports every ONNX variant used by the benchmarks and the export-correctness checks.

FP32 static (several resolutions), FP32 dynamic (batch + rectangular input),
FP16 (needs a CUDA GPU), and INT8 static-quantized (QDQ) models calibrated on
200 frames of the MOT17 *half1* split, i.e. never on evaluation data.
"""

import argparse
import logging
from pathlib import Path

from visiontrack.bench.export import calibration_files, export_onnx, quantize_int8

CALIB_ROOTS = [Path("data/mot17/half1")]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gpu", action="store_true", help="skip the FP16 export")
    ap.add_argument("--calib-images", type=int, default=200)
    args = ap.parse_args()

    calib = calibration_files(CALIB_ROOTS, args.calib_images)
    for imgsz in (320, 480, 640, 960, 1280):
        export_onnx("yolo11s.pt", imgsz)
    export_onnx("yolo11s.pt", 640, dynamic=True)
    for weights in ("yolo11n.pt",):
        p = export_onnx(weights, 640)
        quantize_int8(p, calib, 640)
    s640 = Path("models/onnx/yolo11s_640_fp32.onnx")
    quantize_int8(s640, calib, 640, exclude_head=True)
    quantize_int8(s640, calib, 640, exclude_head=False)
    quantize_int8(s640, calib, 640, exclude_head=True, method="percentile")
    quantize_int8(Path("models/onnx/yolo11s_1280_fp32.onnx"), calib, 1280, exclude_head=True)
    if not args.no_gpu:
        export_onnx("yolo11s.pt", 640, half=True)
        export_onnx("yolo11s.pt", 1280, half=True)
    for p in sorted(Path("models/onnx").glob("*.onnx")):
        print(f"{p.stat().st_size / 2**20:8.1f} MB  {p}")
