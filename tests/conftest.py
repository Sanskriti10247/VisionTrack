from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from visiontrack.config import DetectorConfig
from visiontrack.detection.base import Detector
from visiontrack.types import Detections


def make_video(path: Path, n_frames: int = 30, size=(160, 120), fps: float = 10.0) -> Path:
    """Writes a tiny MJPG/AVI clip whose frame k has a white square at x = 4 * k."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert writer.isOpened()
    for k in range(n_frames):
        img = np.zeros((h, w, 3), np.uint8)
        x = 4 * k % (w - 20)
        img[40:60, x : x + 20] = 255
        cv2.putText(img, str(k), (2, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        writer.write(img)
    writer.release()
    return path


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    return make_video(tmp_path / "tiny.avi")


class ScriptedDetector(Detector):
    """Finds the bright square drawn by make_video, so pipeline tests need no model weights."""

    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        super().__init__(cfg or DetectorConfig(conf_threshold=0.1))

    def detect_batch(self, images):
        out = []
        for img in images:
            mask = img[..., 0] > 200
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                out.append(Detections.empty())
                continue
            box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            out.append(Detections([box], [0.9], [0]))
        self.last_timings = {"inference_ms": 0.0}
        return out


@pytest.fixture
def scripted_detector() -> ScriptedDetector:
    return ScriptedDetector()
