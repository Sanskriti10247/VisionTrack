from visiontrack.config import TrackerConfig
from visiontrack.tracking.base import Tracker
from visiontrack.tracking.bytetrack import ByteTracker


def build_tracker(cfg: TrackerConfig, frame_rate: float = 30.0) -> Tracker:
    if cfg.type == "bytetrack":
        return ByteTracker(cfg, frame_rate=frame_rate)
    if cfg.type == "ultralytics_bytetrack":
        from visiontrack.tracking.ultralytics_adapter import UltralyticsByteTracker

        return UltralyticsByteTracker(cfg, frame_rate=frame_rate)
    raise ValueError(f"unknown tracker type: {cfg.type}")


__all__ = ["ByteTracker", "Tracker", "build_tracker"]
