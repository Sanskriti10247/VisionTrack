"""Controlled detection + tracking experiments on MOT17.

Protocol (see docs/EVALUATION.md):
  1. `detectors`  – model size x input resolution, default ByteTrack, on half1 AND half2
                    (no tuning involved, so held-out numbers are fine to report).
  2. `tracker`    – one-parameter-at-a-time sweeps of ByteTrack on half1 only, using the
                    chosen detector; a "tuned" config is assembled from the per-parameter
                    winners and checked on half1.
  3. `final`      – default vs tuned tracker on held-out half2 (+ ultralytics' ByteTrack on
                    the same detections as an implementation cross-check).
  4. `robustness` – synthetic corruptions of half2 frames, default + tuned tracker.

Usage:
    python scripts/run_mot17_experiments.py --suite all --detector yolo11s.pt:1280
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from visiontrack.config import DetectorConfig, TrackerConfig
from visiontrack.eval.experiments import run_experiment
from visiontrack.eval.perturb import PERTURBATIONS

log = logging.getLogger("experiments")
OUT = Path("reports/experiments")

MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
RESOLUTIONS = [640, 1280]

TRACKER_SWEEPS: dict[str, list] = {
    "track_high_thresh": [0.4, 0.5, 0.6, 0.7],
    "track_low_thresh": [0.05, 0.1, 0.2, 0.3],
    "new_track_thresh": [0.5, 0.6, 0.7, 0.8],
    "match_thresh": [0.7, 0.8, 0.9],
    "track_buffer": [15, 30, 60, 90],
    "fuse_score": [True, False],
    "use_low_score_association": [True, False],
}


def det_cfg(spec: str) -> DetectorConfig:
    weights, imgsz = spec.split(":")
    return DetectorConfig(weights=weights, imgsz=int(imgsz), classes=[0], iou_threshold=0.7)


def _tag(spec: str) -> str:
    w, s = spec.split(":")
    return f"{Path(w).stem}_{s}"


def suite_detectors(splits: list[str]) -> None:
    for split in splits:
        for m in MODELS:
            for r in RESOLUTIONS:
                spec = f"{m}:{r}"
                run_experiment(
                    f"{_tag(spec)}_{split}", det_cfg(spec), TrackerConfig(), split, out_dir=OUT / "detectors"
                )


def suite_tracker(spec: str) -> TrackerConfig:
    d = det_cfg(spec)
    base = TrackerConfig()
    best: dict = {}
    results: dict = {}
    for param, values in TRACKER_SWEEPS.items():
        scores = {}
        for v in values:
            cfg = base.model_copy(update={param: v})
            if cfg.track_low_thresh > cfg.track_high_thresh:
                continue
            name = f"{_tag(spec)}_half1_{param}={v}"
            rec = run_experiment(name, d, cfg, "half1", with_detection_metrics=False, out_dir=OUT / "tracker")
            scores[str(v)] = rec["tracking"]["combined"]
        results[param] = scores
        winner = max(scores, key=lambda k: scores[k]["HOTA"])
        default = str(getattr(base, param))
        # Only move away from the default when the gain is larger than run-to-run noise (+0.2 HOTA).
        if scores[winner]["HOTA"] - scores[default]["HOTA"] > 0.2:
            best[param] = {str(v): v for v in values}[winner]
    tuned = base.model_copy(update=best)
    rec = run_experiment(
        f"{_tag(spec)}_half1_tuned", d, tuned, "half1", with_detection_metrics=False, out_dir=OUT / "tracker"
    )
    default_rec = json.loads((OUT / "tracker" / f"{_tag(spec)}_half1_track_high_thresh=0.5.json").read_text())
    summary = {
        "detector": spec,
        "selection_rule": "per-parameter argmax of HOTA on half1, adopted only if > default + 0.2",
        "sweeps": results,
        "chosen_changes": best,
        "tuned_config": tuned.model_dump(mode="json"),
        "tuned_half1": rec["tracking"]["combined"],
        "default_half1": default_rec["tracking"]["combined"],
    }
    if rec["tracking"]["combined"]["HOTA"] < default_rec["tracking"]["combined"]["HOTA"]:
        summary["note"] = "combined changes did not beat the default on half1; keeping default"
        tuned = base
    summary["final_tracker_config"] = tuned.model_dump(mode="json")
    (OUT / "tracker" / "selection.json").write_text(json.dumps(summary, indent=2))
    return tuned


def load_tuned() -> TrackerConfig:
    sel = json.loads((OUT / "tracker" / "selection.json").read_text())
    return TrackerConfig.model_validate(sel["final_tracker_config"])


def suite_final(spec: str, tuned: TrackerConfig) -> None:
    d = det_cfg(spec)
    out = OUT / "final"
    run_experiment(f"{_tag(spec)}_half2_default", d, TrackerConfig(), "half2", out_dir=out, keep_results=True)
    run_experiment(
        f"{_tag(spec)}_half2_tuned",
        d,
        tuned,
        "half2",
        with_detection_metrics=False,
        out_dir=out,
        keep_results=True,
    )
    run_experiment(
        f"{_tag(spec)}_half2_ultralytics_bytetrack",
        d,
        TrackerConfig(type="ultralytics_bytetrack", class_aware=False),
        "half2",
        with_detection_metrics=False,
        out_dir=out,
    )
    run_experiment(
        f"{_tag(spec)}_half2_no_low_score_assoc",
        d,
        TrackerConfig(use_low_score_association=False),
        "half2",
        with_detection_metrics=False,
        out_dir=out,
    )


def suite_robustness(spec: str, tuned: TrackerConfig) -> None:
    d = det_cfg(spec)
    for p in PERTURBATIONS:
        run_experiment(
            f"{_tag(spec)}_half2_{p}",
            d,
            tuned,
            "half2",
            perturbation=p,
            out_dir=OUT / "robustness",
            keep_results=True,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["detectors", "tracker", "final", "robustness", "all"], default="all")
    ap.add_argument(
        "--detector", default="yolo11s.pt:1280", help="weights:imgsz used for tracker/final/robustness"
    )
    ap.add_argument("--splits", nargs="+", default=["half1", "half2"])
    args = ap.parse_args()

    if args.suite in ("detectors", "all"):
        suite_detectors(args.splits)
    if args.suite in ("tracker", "all"):
        suite_tracker(args.detector)
    if args.suite in ("final", "all"):
        suite_final(args.detector, load_tuned())
    if args.suite in ("robustness", "all"):
        suite_robustness(args.detector, load_tuned())
