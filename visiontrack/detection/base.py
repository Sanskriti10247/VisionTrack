"""Detector interface. Concrete backends: PyTorch (ultralytics) and ONNX Runtime."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from visiontrack.config import DetectorConfig
from visiontrack.types import COCO_NAMES, Detections


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Detector(ABC):
    """Runs object detection on BGR uint8 frames.

    ``last_timings`` holds a per-image breakdown (milliseconds) of the most
    recent call so the pipeline can separate detector latency from the rest.
    """

    names: tuple[str, ...] = COCO_NAMES

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.last_timings: dict[str, float] = {}

    @abstractmethod
    def detect_batch(self, images: list[np.ndarray]) -> list[Detections]: ...

    def detect(self, image: np.ndarray) -> Detections:
        return self.detect_batch([image])[0]

    def warmup(self, iterations: int = 3, shape: tuple[int, int] = (720, 1280)) -> float:
        """Runs dummy inferences (CUDA context, cuDNN autotune, ORT allocations); returns elapsed ms."""
        dummy = np.full((*shape, 3), 114, dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(iterations):
            self.detect_batch([dummy] * self.cfg.batch_size)
        return (time.perf_counter() - t0) * 1000

    def info(self) -> dict:
        return {
            "backend": self.cfg.backend,
            "weights": self.cfg.weights,
            "imgsz": self.cfg.imgsz,
            "conf_threshold": self.cfg.conf_threshold,
            "iou_threshold": self.cfg.iou_threshold,
            "classes": self.cfg.classes,
            "half": self.cfg.half,
        }

    def _class_filter(self, dets: Detections) -> Detections:
        if self.cfg.classes is None or len(dets) == 0:
            return dets
        return dets.filter(np.isin(dets.class_id, np.asarray(self.cfg.classes)))
