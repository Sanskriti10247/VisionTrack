"""Detector evaluation on COCO val2017 with pycocotools.

One pass at conf=0.001 over all 5,000 val images; metrics are reported for

* the street classes VisionTrack uses (person, bicycle, car, motorcycle, bus, truck), and
* all 80 classes, as a check of this harness against the published model numbers.

Note: val2017 is not used to train the ultralytics checkpoints, but it was
used for model selection, so this is held-out *data*, not a pristine test set.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2

from visiontrack.config import DetectorConfig
from visiontrack.detection import build_detector
from visiontrack.eval.detection import coco_category_map, coco_evaluate, detections_to_coco
from visiontrack.eval.hardware import hardware_info
from visiontrack.timing import summarize
from visiontrack.types import DEFAULT_CLASSES

log = logging.getLogger(__name__)

COCO_DIR = Path("data/coco")


def run_coco_eval(
    det_cfg: DetectorConfig,
    out_dir: Path,
    images_dir: Path = COCO_DIR / "val2017",
    ann_file: Path = COCO_DIR / "annotations" / "instances_val2017.json",
    max_images: int | None = None,
    operating_conf: float = 0.25,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = json.loads(Path(ann_file).read_text())
    index_to_cat, cat_names = coco_category_map(ann_file)
    images = sorted(gt["images"], key=lambda im: im["id"])
    if max_images:
        images = images[:max_images]

    # Evaluate on all classes in one pass; subsets are selected at scoring time.
    cfg = det_cfg.model_copy(update={"conf_threshold": 0.001, "classes": None, "batch_size": 1})
    detector = build_detector(cfg)
    detector.warmup()
    dets: list[dict] = []
    times: list[float] = []
    for k, im in enumerate(images):
        img = cv2.imread(str(images_dir / im["file_name"]))
        if img is None:
            raise FileNotFoundError(images_dir / im["file_name"])
        t0 = time.perf_counter()
        d = detector.detect(img)
        times.append((time.perf_counter() - t0) * 1000)
        dets.extend(detections_to_coco(im["id"], d, index_to_cat))
        if (k + 1) % 500 == 0:
            log.info("%d/%d images", k + 1, len(images))
    (out_dir / "detections.json").write_text(json.dumps(dets))

    img_ids = [im["id"] for im in images]
    street_ids = [index_to_cat[i] for i in DEFAULT_CLASSES]
    result = {
        "dataset": {"name": "COCO val2017", "num_images": len(images), "ann_file": str(ann_file)},
        "detector": det_cfg.model_dump(mode="json"),
        "eval_conf_threshold": 0.001,
        "street_classes": coco_evaluate(gt, dets, street_ids, 100, img_ids, operating_conf, cat_names),
        "all_80_classes": coco_evaluate(gt, dets, list(cat_names), 100, img_ids, operating_conf, cat_names),
        "detector_ms_per_image": summarize(times),
        "hardware": hardware_info(),
    }
    result["all_80_classes"].pop("per_class", None)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result
