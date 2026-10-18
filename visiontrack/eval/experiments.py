"""Reproducible detection + tracking experiments on MOT17 splits.

An experiment = (detector config, optional perturbation, tracker config,
split). Detections come from the cache (visiontrack.eval.detcache), so every
tracker variant sees byte-identical detector output; each run writes a JSON
record with the full configuration, hardware, and metrics.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

from visiontrack.config import DetectorConfig, TrackerConfig
from visiontrack.eval.detcache import build_cache, cache_name, load_cache
from visiontrack.eval.detection import coco_evaluate, detections_to_coco, merge_coco, mot17_gt_to_coco
from visiontrack.eval.hardware import hardware_info
from visiontrack.eval.mot17 import SeqInfo, list_split
from visiontrack.eval.tracking import track_sequence, trackeval_mot17, write_results

log = logging.getLogger(__name__)

DATA_ROOT = Path("data/mot17")
CACHE_ROOT = Path("runs/detcache")
PERSON = [0]


def ensure_cache(
    det_cfg: DetectorConfig, split: str, perturbation: str | None = None, batch_size: int = 4
) -> tuple[Path, list[SeqInfo]]:
    seqs = list_split(DATA_ROOT, split)
    path = CACHE_ROOT / cache_name(det_cfg, perturbation) / split
    build_cache(det_cfg, seqs, path, perturbation=perturbation, batch_size=batch_size)
    return path, seqs


def detection_metrics(cache: Path, seqs: list[SeqInfo], operating_conf: float = 0.5) -> dict:
    """Person-detection AP on MOT17 GT (visibility < 0.25 and distractors are ignore regions)."""
    parts, dets = [], []
    for k, s in enumerate(seqs):
        offset = (k + 1) * 100000
        parts.append(mot17_gt_to_coco(s.gt_path, s.length, image_id_offset=offset))
        for f, d in enumerate(load_cache(cache, s.name, classes=PERSON), start=1):
            dets.extend(detections_to_coco(offset + f, d, index_to_cat=[1]))
    gt = merge_coco(parts)
    # MOT17 frames hold up to ~100 people, so allow 300 detections per image (COCO default is 100).
    r = coco_evaluate(gt, dets, [1], max_dets=300, operating_conf=operating_conf, cat_names={1: "person"})
    r.pop("per_class", None)
    return r


def tracking_metrics(
    cache: Path,
    seqs: list[SeqInfo],
    tracker_cfg: TrackerConfig,
    det_conf: float,
    split: str,
    keep_results: Path | None = None,
) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="vt_track_"))
    try:
        name = "tracker"
        stats = {}
        t0 = time.perf_counter()
        n_frames = 0
        for s in seqs:
            dets = load_cache(cache, s.name, conf=det_conf, classes=PERSON)
            rows, st = track_sequence(dets, tracker_cfg, s.fps)
            write_results(rows, tmp / name / "data" / f"{s.name}.txt")
            stats[s.name] = st
            n_frames += len(dets)
        track_s = time.perf_counter() - t0
        metrics = trackeval_mot17(DATA_ROOT / split, tmp, name, seqs)
        if keep_results is not None:
            keep_results.mkdir(parents=True, exist_ok=True)
            for f in (tmp / name / "data").glob("*.txt"):
                shutil.copy(f, keep_results / f.name)
        metrics["tracker_ms_per_frame"] = round(1000 * track_s / max(n_frames, 1), 3)
        metrics["ids_per_sequence"] = {k: v["num_ids"] for k, v in stats.items()}
        return metrics
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_experiment(
    name: str,
    det_cfg: DetectorConfig,
    tracker_cfg: TrackerConfig,
    split: str,
    det_conf: float | None = None,
    perturbation: str | None = None,
    with_detection_metrics: bool = True,
    out_dir: Path = Path("reports/experiments"),
    keep_results: bool = False,
) -> dict:
    det_conf = tracker_cfg.track_low_thresh if det_conf is None else det_conf
    cache, seqs = ensure_cache(det_cfg, split, perturbation)
    meta = json.loads((cache / "meta.json").read_text())
    record = {
        "name": name,
        "dataset": {
            "name": "MOT17 train (FRCNN copy)",
            "split": split,
            "sequences": [s.name for s in seqs],
            "frames": sum(s.length for s in seqs),
        },
        "detector": det_cfg.model_dump(mode="json"),
        "detector_conf_into_tracker": det_conf,
        "perturbation": perturbation,
        "tracker": tracker_cfg.model_dump(mode="json"),
        "detector_timing_during_caching": {
            "detector_ms_per_frame": meta.get("detector_ms_per_frame"),
            "batch_size": meta.get("batch_size"),
            "note": "measured while caching (batched); see reports/benchmarks for controlled latency numbers",
        },
        "hardware": hardware_info(),
    }
    if with_detection_metrics:
        record["detection"] = detection_metrics(cache, seqs)
    record["tracking"] = tracking_metrics(
        cache,
        seqs,
        tracker_cfg,
        det_conf,
        split,
        keep_results=(out_dir / "mot_results" / name) if keep_results else None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=2))
    c = record["tracking"]["combined"]
    log.info("%s: HOTA %.2f MOTA %.2f IDF1 %.2f IDSW %d", name, c["HOTA"], c["MOTA"], c["IDF1"], c["IDSW"])
    return record
