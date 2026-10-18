"""Tracker interface so detectors and trackers can be evaluated independently."""

from __future__ import annotations

from abc import ABC, abstractmethod

from visiontrack.types import Detections, TrackedObject


class Tracker(ABC):
    def __init__(self) -> None:
        # Confirmed tracks that were terminated during the most recent update()
        # (lost for longer than the track buffer). Analytics use this to close
        # zone visits and trajectories.
        self.ended_track_ids: list[int] = []

    @abstractmethod
    def update(self, detections: Detections) -> list[TrackedObject]:
        """Consumes one frame of detections and returns the confirmed, currently visible tracks."""

    @abstractmethod
    def reset(self) -> None: ...

    def active_track_ids(self) -> list[int]:
        """IDs of tracks still alive (tracked or temporarily lost)."""
        return []
