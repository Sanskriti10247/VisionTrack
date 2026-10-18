"""Caches raw detector output per sequence so tracker experiments reuse identical detections.

Detections are stored at a very low confidence (0.001) and filtered later:
greedy NMS keeps or suppresses a box based only on higher-scoring boxes, so
"NMS at 0.001 then drop < t" equals "drop < t then NMS" for any t (up to the
max_det cap, which MOT17 never reaches). One detector pass therefore serves
every confidence threshold, tracker setting, and the detection mAP.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from visiontrack.config import DetectorConfig
from visiontrack.eval.hardware import hardware_info
from visiontrack.eval.mot17 import SeqInfo
from visiontrack.eval.perturb import PERTURBATIONS
from visiontrack.io.video import VideoSource
from visiontrack.timing import summarize
from visiontrack.types import Detections

log = logging.getLogger(__name__)

CACHE_CONF = 0.001


def cache_name(cfg: DetectorConfig, perturbation: str | None = None) -> str:
    stem = Path(cfg.weights).stem
    key = json.dumps(
        {
            k: getattr(cfg, k)
            for k in ("backend", "weights", "imgsz", "iou_threshold", "classes", "half", "max_det")
        },
        sort_keys=True,
    )
    h = hashlib.sha1(key.encode()).hexdigest()[:8]
    parts = [stem, cfg.backend, str(cfg.imgsz)]
    if cfg.half:
        parts.append("fp16")
    if perturbation:
        parts.append(perturbation)
    return "_".join(parts) + f"_{h}"


def build_cache(
    cfg: DetectorConfig,
    seqs: list[SeqInfo],
    out_dir: Path,
    perturbation: str | None = None,
    batch_size: int = 4,
    force: bool = False,
) -> Path:
    """Runs the detector over every frame of ``seqs`` and stores one .npz per sequence."""
    from visiontrack.detection import build_detector

    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seqs if force or not (out_dir / f"{s.name}.npz").exists()]
    if not todo:
        return out_dir
    det_cfg = cfg.model_copy(update={"conf_threshold": CACHE_CONF, "batch_size": batch_size})
    detector = build_detector(det_cfg)
    warm_ms = detector.warmup(shape=(todo[0].height, todo[0].width))
    perturb = PERTURBATIONS[perturbation] if perturbation else None
    per_frame_ms: list[float] = []

    for seq in todo:
        t_seq = time.perf_counter()
        boxes, scores, classes, counts = [], [], [], []
        batch, idxs = [], []

        def flush() -> None:
            t0 = time.perf_counter()
            res = detector.detect_batch(batch)
            per_frame_ms.extend([(time.perf_counter() - t0) * 1000 / len(batch)] * len(batch))
            for d in res:
                boxes.append(d.xyxy)
                scores.append(d.confidence)
                classes.append(d.class_id)
                counts.append(len(d))
            batch.clear()
            idxs.clear()

        for frame in VideoSource(seq.img_dir):
            img = perturb(frame.image, frame.index) if perturb else frame.image
            batch.append(img)
            idxs.append(frame.index)
            if len(batch) == batch_size:
                flush()
        if batch:
            flush()
        if len(counts) != seq.length:
            raise RuntimeError(f"{seq.name}: expected {seq.length} frames, detected {len(counts)}")
        np.savez_compressed(
            out_dir / f"{seq.name}.npz",
            boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
            scores=np.concatenate(scores) if scores else np.zeros(0, np.float32),
            classes=np.concatenate(classes) if classes else np.zeros(0, np.int64),
            counts=np.asarray(counts, np.int64),
        )
        log.info("cached %s (%d frames) in %.1fs", seq.name, seq.length, time.perf_counter() - t_seq)

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        {
            "detector": detector.info(),
            "cache_conf": CACHE_CONF,
            "perturbation": perturbation,
            "batch_size": batch_size,
            "warmup_ms": round(warm_ms, 1),
            "detector_ms_per_frame": summarize(per_frame_ms),
            "hardware": hardware_info(),
            "sequences": sorted({*meta.get("sequences", []), *(s.name for s in todo)}),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    return out_dir


def load_cache(
    path: Path, seq_name: str, conf: float = 0.0, classes: list[int] | None = None
) -> list[Detections]:
    """Returns per-frame detections (index 0 = frame 1) with score >= conf."""
    z = np.load(path / f"{seq_name}.npz")
    boxes, scores, cls, counts = z["boxes"], z["scores"], z["classes"], z["counts"]
    out = []
    start = 0
    for n in counts:
        sl = slice(start, start + int(n))
        start += int(n)
        keep = scores[sl] >= conf
        if classes is not None:
            keep &= np.isin(cls[sl], classes)
        out.append(Detections(boxes[sl][keep], scores[sl][keep], cls[sl][keep]))
    return out
