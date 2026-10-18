"""Tracking evaluation on MOT17 with TrackEval (HOTA, CLEAR/MOTA, Identity/IDF1).

Uses TrackEval's MotChallenge2DBox dataset class with BENCHMARK=MOT17 and
DO_PREPROC=True, i.e. the official MOTChallenge protocol:

* only class 1 (pedestrian) GT with consider-flag 1 is evaluated;
* tracker boxes that match (IoU >= 0.5) distractor GT (classes 2, 7, 8, 12)
  are removed before scoring rather than counted as false positives;
* CLEAR/Identity use an IoU threshold of 0.5; HOTA averages over
  localisation thresholds alpha = 0.05 .. 0.95.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np

from visiontrack.config import TrackerConfig
from visiontrack.eval.mot17 import SeqInfo
from visiontrack.tracking import build_tracker
from visiontrack.types import Detections


def _patch_numpy_aliases() -> None:
    # TrackEval (last release 2022) still uses aliases removed in NumPy 1.24.
    # They were plain aliases of the builtins, so restoring them is exact.
    for name, typ in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, typ)


def track_sequence(
    dets_per_frame: list[Detections], tracker_cfg: TrackerConfig, fps: float
) -> tuple[list[str], dict]:
    """Runs the tracker over one sequence; returns MOT rows and simple stats."""
    tracker = build_tracker(tracker_cfg, frame_rate=fps)
    rows: list[str] = []
    ids: set[int] = set()
    for f, dets in enumerate(dets_per_frame, start=1):
        for t in tracker.update(dets):
            x1, y1, x2, y2 = (float(v) for v in t.xyxy)
            rows.append(
                f"{f},{t.track_id},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{t.score:.4f},-1,-1,-1"
            )
            ids.add(t.track_id)
    return rows, {"num_ids": len(ids), "num_boxes": len(rows)}


def write_results(rows: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""))


def trackeval_mot17(gt_root: Path, trackers_root: Path, tracker_name: str, seqs: list[SeqInfo]) -> dict:
    """Evaluates ``trackers_root/<tracker_name>/data/<seq>.txt`` against ``gt_root/<seq>/gt/gt.txt``."""
    _patch_numpy_aliases()
    import trackeval

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(
        {
            "USE_PARALLEL": False,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
            "DISPLAY_LESS_PROGRESS": True,
        }
    )
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(trackers_root),
            "OUTPUT_FOLDER": str(trackers_root / "_trackeval"),
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "DO_PREPROC": True,
            "TRACKER_SUB_FOLDER": "data",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {s.name: s.length for s in seqs},
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
            "PRINT_CONFIG": False,
        }
    )
    metric_cfg = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    metrics = [
        trackeval.metrics.HOTA(metric_cfg),
        trackeval.metrics.CLEAR(metric_cfg),
        trackeval.metrics.Identity(metric_cfg),
    ]
    evaluator = trackeval.Evaluator(eval_cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        res, msg = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)
    status = msg["MotChallenge2DBox"][tracker_name]
    if status != "Success":
        raise RuntimeError(f"TrackEval failed: {status}")
    per_seq = res["MotChallenge2DBox"][tracker_name]
    out = {"combined": _flatten(per_seq["COMBINED_SEQ"]["pedestrian"])}
    out["per_sequence"] = {
        name: _flatten(v["pedestrian"]) for name, v in per_seq.items() if name != "COMBINED_SEQ"
    }
    return out


def _flatten(r: dict) -> dict:
    h, c, i = r["HOTA"], r["CLEAR"], r["Identity"]
    pct = lambda x: round(100 * float(np.mean(x)), 2)  # noqa: E731
    return {
        "HOTA": pct(h["HOTA"]),
        "DetA": pct(h["DetA"]),
        "AssA": pct(h["AssA"]),
        "LocA": pct(h["LocA"]),
        "MOTA": pct(c["MOTA"]),
        "MOTP": pct(c["MOTP"]),
        "IDF1": pct(i["IDF1"]),
        "IDP": pct(i["IDP"]),
        "IDR": pct(i["IDR"]),
        "IDSW": int(c["IDSW"]),
        "Frag": int(c["Frag"]),
        "MT": int(c["MT"]),
        "ML": int(c["ML"]),
        "FP": int(c["CLR_FP"]),
        "FN": int(c["CLR_FN"]),
        "TP": int(c["CLR_TP"]),
        "GT_IDs": int(c["MT"] + c["PT"] + c["ML"]),
    }
