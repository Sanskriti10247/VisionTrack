"""Detection metrics (P/R, mAP@0.5, mAP@0.5:0.95) on COCO val2017.

Example:
    python scripts/eval_coco.py --weights yolo11s.pt --imgsz 640
    python scripts/eval_coco.py --weights models/onnx/yolo11s_640_fp32.onnx --backend onnx
"""

import argparse
import json
import logging
from pathlib import Path

from visiontrack.config import DetectorConfig
from visiontrack.eval.coco_eval import run_coco_eval

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--backend", choices=["ultralytics", "onnx"], default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--name", help="output folder name under reports/detection/coco")
    args = ap.parse_args()

    backend = args.backend or ("onnx" if args.weights.endswith(".onnx") else "ultralytics")
    cfg = DetectorConfig(
        backend=backend, weights=args.weights, imgsz=args.imgsz, device=args.device, half=args.half
    )
    name = args.name or f"{Path(args.weights).stem}_{backend}_{args.imgsz}" + ("_fp16" if args.half else "")
    res = run_coco_eval(cfg, Path("runs/coco") / name, max_images=args.max_images)
    out = Path("reports/detection/coco") / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    s, a = res["street_classes"], res["all_80_classes"]
    print(
        f"{name}: street mAP50-95 {s['mAP50_95']:.4f} mAP50 {s['mAP50']:.4f} "
        f"P {s['operating_point']['precision']:.3f} R {s['operating_point']['recall']:.3f} | "
        f"all-80 mAP50-95 {a['mAP50_95']:.4f} | det p50 {res['detector_ms_per_image']['p50']} ms"
    )
