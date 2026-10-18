"""PyTorch YOLO detector via the ultralytics package."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from visiontrack.config import DetectorConfig
from visiontrack.detection.base import Detector, resolve_device
from visiontrack.types import Detections

log = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def resolve_weights(weights: str) -> str:
    """Keeps downloaded checkpoints in ./models instead of the working directory."""
    p = Path(weights)
    if p.exists():
        return str(p)
    local = MODELS_DIR / p.name
    if local.exists():
        return str(local)
    if p.suffix == ".pt" and p.parent == Path("."):
        from ultralytics.utils.downloads import attempt_download_asset

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return str(attempt_download_asset(local))
    raise FileNotFoundError(f"weights not found: {weights}")


class UltralyticsDetector(Detector):
    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        from ultralytics import YOLO

        self.device = resolve_device(cfg.device)
        self.half = cfg.half and self.device.startswith("cuda")
        if cfg.half and not self.half:
            log.warning("FP16 requested but running on %s; using FP32", self.device)
        self.weights_path = resolve_weights(cfg.weights)
        self.model = YOLO(self.weights_path, task="detect")
        names = self.model.names
        self.names = tuple(names[i] for i in range(len(names)))

    def detect_batch(self, images: list[np.ndarray]) -> list[Detections]:
        if not images:
            return []
        t0 = time.perf_counter()
        results = self.model.predict(
            images,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            classes=self.cfg.classes,
            device=self.device,
            half=self.half,
            max_det=self.cfg.max_det,
            verbose=False,
        )
        out = []
        for r in results:
            b = r.boxes
            out.append(
                Detections(
                    b.xyxy.cpu().numpy(),
                    b.conf.cpu().numpy(),
                    b.cls.cpu().numpy().astype(np.int64),
                )
            )
        wall = (time.perf_counter() - t0) * 1000 / len(images)
        speed = results[0].speed  # ultralytics reports per-image averages for the batch
        self.last_timings = {
            "preprocess_ms": float(speed.get("preprocess") or 0.0),
            "inference_ms": float(speed.get("inference") or 0.0),
            "postprocess_ms": float(speed.get("postprocess") or 0.0),
            "detector_total_ms": wall,
        }
        return out

    def info(self) -> dict:
        return {**super().info(), "device": self.device, "half": self.half, "weights_path": self.weights_path}
