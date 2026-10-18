from visiontrack.config import DetectorConfig
from visiontrack.detection.base import Detector


def build_detector(cfg: DetectorConfig) -> Detector:
    """Instantiates the configured backend. Imports are lazy so the ONNX path does not need torch."""
    if cfg.backend == "ultralytics":
        from visiontrack.detection.ultralytics_detector import UltralyticsDetector

        return UltralyticsDetector(cfg)
    if cfg.backend == "onnx":
        from visiontrack.detection.onnx_detector import OnnxDetector

        return OnnxDetector(cfg)
    raise ValueError(f"unknown detector backend: {cfg.backend}")


__all__ = ["Detector", "build_detector"]
