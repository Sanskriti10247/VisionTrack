"""Checks that exported / quantized models still do the job.

1. Numerical: identical letterboxed tensors (held-out MOT17 frames) through the PyTorch
   model and each ONNX variant; max / mean absolute difference of the raw head output.
2. Detection agreement: after identical post-processing (conf 0.25, NMS 0.7), how many
   PyTorch boxes each variant reproduces (IoU >= 0.5, same class) and vice versa.
3. Task level (optional, --coco): COCO val2017 mAP for each variant via scripts/eval_coco.py logic.

Writes reports/benchmarks/export_verification.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from visiontrack.bench.export import raw_outputs_onnx, raw_outputs_torch
from visiontrack.eval.hardware import hardware_info
from visiontrack.preprocess import letterbox, postprocess, to_tensor
from visiontrack.tracking.matching import box_iou

log = logging.getLogger("verify")
ONNX = Path("models/onnx")
FRAMES = sorted(Path("data/mot17/half2/MOT17-02/img1").glob("*.jpg"))[::25] + sorted(
    Path("data/mot17/half2/MOT17-10/img1").glob("*.jpg")
)[::25]

VARIANTS = {
    "fp32": ("yolo11s_640_fp32.onnx", ["CPUExecutionProvider"]),
    "fp32_dynamic": ("yolo11s_640_fp32_dynamic.onnx", ["CPUExecutionProvider"]),
    "fp16_cuda": ("yolo11s_640_fp16.onnx", ["CUDAExecutionProvider"]),
    "int8_head_fp32": ("yolo11s_640_int8.onnx", ["CPUExecutionProvider"]),
    "int8_percentile": ("yolo11s_640_int8_percentile.onnx", ["CPUExecutionProvider"]),
    "int8_fullgraph": ("yolo11s_640_int8_fullgraph.onnx", ["CPUExecutionProvider"]),
}


def agreement(ref: np.ndarray, test: np.ndarray, infos) -> dict:
    """Box-level agreement between two raw outputs after the same post-processing."""
    n_ref = n_test = matched = 0
    ious, dconf = [], []
    for r, t, info in zip(ref, test, infos, strict=True):
        rb, rc, rk = postprocess(r, info, 0.25, 0.7, None)
        tb, tc, tk = postprocess(t, info, 0.25, 0.7, None)
        n_ref += len(rb)
        n_test += len(tb)
        if not len(rb) or not len(tb):
            continue
        iou = box_iou(rb, tb)
        iou[rk[:, None] != tk[None, :]] = 0
        used = set()
        for i in np.argsort(-rc):
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= 0.5 and j not in used:
                used.add(j)
                matched += 1
                ious.append(iou[i, j])
                dconf.append(abs(rc[i] - tc[j]))
    return {
        "ref_boxes": n_ref,
        "test_boxes": n_test,
        "matched": matched,
        "ref_recall": round(matched / max(n_ref, 1), 4),
        "test_precision": round(matched / max(n_test, 1), 4),
        "mean_iou_matched": round(float(np.mean(ious)), 4) if ious else None,
        "mean_abs_conf_diff": round(float(np.mean(dconf)), 4) if dconf else None,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", action="store_true", help="also run COCO val2017 mAP for each variant")
    args = ap.parse_args()

    boxed, infos = zip(*(letterbox(cv2.imread(str(f)), (640, 640)) for f in FRAMES), strict=True)
    x = to_tensor(list(boxed))
    ref = np.concatenate([raw_outputs_torch("yolo11s.pt", x[i : i + 1]) for i in range(len(x))])
    report: dict = {"frames": [str(f) for f in FRAMES], "reference": "PyTorch FP32 yolo11s, same input tensors",
                    "variants": {}, "hardware": hardware_info()}
    for name, (fname, providers) in VARIANTS.items():
        path = ONNX / fname
        if not path.exists():
            log.warning("skip %s (missing %s)", name, path)
            continue
        out = np.concatenate([raw_outputs_onnx(path, x[i : i + 1], providers) for i in range(len(x))])
        box_diff = np.abs(out[:, :4] - ref[:, :4])
        cls_diff = np.abs(out[:, 4:] - ref[:, 4:])
        report["variants"][name] = {
            "file": str(path),
            "size_mb": round(path.stat().st_size / 2**20, 2),
            "providers": providers,
            "raw_box_abs_diff_px": {"max": round(float(box_diff.max()), 4), "mean": round(float(box_diff.mean()), 5)},
            "raw_score_abs_diff": {"max": round(float(cls_diff.max()), 5), "mean": round(float(cls_diff.mean()), 7)},
            "detection_agreement_conf0.25": agreement(ref, out, infos),
        }
        log.info("%s: %s", name, json.dumps(report["variants"][name]["detection_agreement_conf0.25"]))

    if args.coco:
        from visiontrack.config import DetectorConfig
        from visiontrack.eval.coco_eval import run_coco_eval

        coco = {}
        for name, (fname, providers) in VARIANTS.items():
            path = ONNX / fname
            if not path.exists():
                continue
            device = "cuda:0" if "CUDAExecutionProvider" in providers else "cpu"
            cfg = DetectorConfig(backend="onnx", weights=str(path), imgsz=640, device=device)
            res = run_coco_eval(cfg, Path("runs/coco") / f"onnx_{name}")
            coco[name] = {"street_classes": {k: res["street_classes"][k] for k in ("mAP50_95", "mAP50", "operating_point")},
                          "all_80_mAP50_95": res["all_80_classes"]["mAP50_95"],
                          "detector_ms_per_image": res["detector_ms_per_image"]}
            log.info("coco %s: %s", name, coco[name]["street_classes"]["mAP50_95"])
        report["coco_val2017"] = coco

    out = Path("reports/benchmarks/export_verification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
