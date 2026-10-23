"""Detection metrics with pycocotools (COCO's reference implementation).

* mAP@0.5:0.95 and mAP@0.5 are COCO's area-under-PR-curve averages
  (101-point interpolation), computed with COCOeval unchanged.
* Precision / recall are additionally reported at a fixed *operating point*
  (a deployment confidence threshold, IoU 0.5), because mAP integrates over
  all thresholds and does not say how the detector behaves at the threshold
  actually used in the pipeline. They are derived from COCOeval's own
  per-image matching, so they use exactly the same matches as the mAP.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from visiontrack.eval.mot17 import DISTRACTOR_CLASSES, PEDESTRIAN, load_gt
from visiontrack.types import Detections


def _coco_objects(gt: dict, dets: list[dict]):
    from pycocotools.coco import COCO

    coco_gt = COCO()
    coco_gt.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets) if dets else None
    return coco_gt, coco_dt


def coco_evaluate(
    gt: dict,
    dets: list[dict],
    cat_ids: list[int],
    max_dets: int = 100,
    img_ids: list[int] | None = None,
    operating_conf: float = 0.25,
    cat_names: dict[int, str] | None = None,
) -> dict:
    from pycocotools.cocoeval import COCOeval

    coco_gt, coco_dt = _coco_objects(gt, dets)
    if coco_dt is None:
        return {"mAP50_95": 0.0, "mAP50": 0.0, "note": "no detections"}
    E = COCOeval(coco_gt, coco_dt, "bbox")
    E.params.catIds = list(cat_ids)
    if img_ids is not None:
        E.params.imgIds = list(img_ids)
    E.params.maxDets = [1, 10, max_dets]
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        E.evaluate()
        E.accumulate()
        E.summarize()
    # Read AP/AR straight from the accumulated arrays rather than E.stats:
    # summarize() hard-codes maxDets=100 for some entries and returns -1
    # when a different max_dets is configured.
    prec = E.eval["precision"]  # [T(iou), R(recall pts), K(class), A(area), M(maxdets)]
    rec = E.eval["recall"]  # [T, K, A, M]

    def ap(t=None, area=0) -> float:
        p = prec[:, :, :, area, -1] if t is None else prec[t, :, :, area, -1]
        p = p[p > -1]
        return round(float(p.mean()), 4) if p.size else -1.0

    r_all = rec[:, :, 0, -1]
    r_all = r_all[r_all > -1]
    out = {
        "mAP50_95": ap(),
        "mAP50": ap(0),
        "mAP75": ap(5),
        "mAP_small": ap(area=1),
        "mAP_medium": ap(area=2),
        "mAP_large": ap(area=3),
        f"AR@{max_dets}": round(float(r_all.mean()), 4) if r_all.size else -1.0,
        "max_dets": max_dets,
        "num_images": len(E.params.imgIds),
        "pycocotools_summary": buf.getvalue().strip().splitlines(),
    }
    per_class = {}
    for k, cid in enumerate(E.params.catIds):
        p = prec[:, :, k, 0, -1]
        p50 = prec[0, :, k, 0, -1]
        name = (cat_names or {}).get(cid, str(cid))
        per_class[name] = {
            "AP50_95": round(float(p[p > -1].mean()), 4) if (p > -1).any() else None,
            "AP50": round(float(p50[p50 > -1].mean()), 4) if (p50 > -1).any() else None,
        }
    out["per_class"] = per_class
    out["operating_point"] = operating_point(E, operating_conf)
    out["best_f1_point"] = best_f1(E)
    return out


def _match_arrays(E) -> tuple[np.ndarray, np.ndarray, int]:
    """Scores and TP flags (IoU 0.5, area 'all') of all non-ignored detections, plus #non-ignored GT."""
    all_rng = E.params.areaRng[0]
    scores, tps, n_gt = [], [], 0
    for e in E.evalImgs:
        if e is None or e["aRng"] != all_rng:
            continue
        dt_ignore = np.asarray(e["dtIgnore"])[0].astype(bool) if len(e["dtIds"]) else np.zeros(0, bool)
        matched = np.asarray(e["dtMatches"])[0] > 0 if len(e["dtIds"]) else np.zeros(0, bool)
        sc = np.asarray(e["dtScores"])
        scores.append(sc[~dt_ignore])
        tps.append(matched[~dt_ignore])
        n_gt += int((~np.asarray(e["gtIgnore"]).astype(bool)).sum())
    if not scores:
        return np.zeros(0), np.zeros(0, bool), n_gt
    return np.concatenate(scores), np.concatenate(tps), n_gt


def operating_point(E, conf: float) -> dict:
    scores, tp, n_gt = _match_arrays(E)
    keep = scores >= conf
    n_tp = int(tp[keep].sum())
    n_fp = int((~tp[keep]).sum())
    p = n_tp / (n_tp + n_fp) if n_tp + n_fp else 0.0
    r = n_tp / n_gt if n_gt else 0.0
    return {
        "conf_threshold": conf,
        "iou_threshold": 0.5,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "tp": n_tp,
        "fp": n_fp,
        "fn": n_gt - n_tp,
    }


def best_f1(E) -> dict:
    scores, tp, n_gt = _match_arrays(E)
    if not len(scores) or not n_gt:
        return {}
    order = np.argsort(-scores, kind="mergesort")
    tp_c = np.cumsum(tp[order])
    fp_c = np.cumsum(~tp[order])
    prec = tp_c / (tp_c + fp_c)
    rec = tp_c / n_gt
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return {
        "conf_threshold": round(float(scores[order][i]), 4),
        "precision": round(float(prec[i]), 4),
        "recall": round(float(rec[i]), 4),
        "f1": round(float(f1[i]), 4),
    }


# ------------------------------------------------------------------- COCO val


def coco_category_map(ann_file: Path) -> tuple[list[int], dict[int, str]]:
    """YOLO's contiguous 80-class index -> COCO category id (categories sorted by id)."""
    cats = sorted(json.loads(Path(ann_file).read_text())["categories"], key=lambda c: c["id"])
    return [c["id"] for c in cats], {c["id"]: c["name"] for c in cats}


def detections_to_coco(image_id: int, dets: Detections, index_to_cat: list[int]) -> list[dict]:
    out = []
    for (x1, y1, x2, y2), s, c in zip(dets.xyxy, dets.confidence, dets.class_id, strict=True):
        out.append(
            {
                "image_id": int(image_id),
                "category_id": int(index_to_cat[int(c)]),
                "bbox": [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2 - x1), 2),
                    round(float(y2 - y1), 2),
                ],
                "score": round(float(s), 5),
            }
        )
    return out


# ---------------------------------------------------------------------- MOT17


def mot17_gt_to_coco(
    gt_path: Path, n_frames: int, min_visibility: float = 0.25, image_id_offset: int = 0
) -> dict:
    """Pedestrian boxes as COCO GT for one sequence.

    Distractor classes (static person, person on vehicle, reflection, ...)
    and pedestrians below ``min_visibility`` become ``iscrowd`` regions: a
    detection on them is neither a TP nor an FP (pycocotools ignores
    detections matched to crowd regions). Missing a barely-visible person is
    a tracking concern (measured there), not a detector error.
    """
    gt = load_gt(gt_path)
    images = [{"id": image_id_offset + f, "file_name": f"{f:06d}.jpg"} for f in range(1, n_frames + 1)]
    anns = []
    for i, (frame, _tid, x, y, w, h, consider, cls, vis) in enumerate(gt):
        is_ped = int(cls) == PEDESTRIAN and int(consider) == 1
        is_ignore = int(cls) in DISTRACTOR_CLASSES or (is_ped and vis < min_visibility)
        if not (is_ped or is_ignore):
            continue  # vehicles, non-person objects
        anns.append(
            {
                "id": image_id_offset * 1000 + i + 1,
                "image_id": image_id_offset + int(frame),
                "category_id": 1,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": int(is_ignore),
            }
        )
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}


def merge_coco(parts: list[dict]) -> dict:
    merged = {"images": [], "annotations": [], "categories": parts[0]["categories"] if parts else []}
    next_id = 1
    for p in parts:
        merged["images"].extend(p["images"])
        for a in p["annotations"]:
            merged["annotations"].append({**a, "id": next_id})
            next_id += 1
    return merged
