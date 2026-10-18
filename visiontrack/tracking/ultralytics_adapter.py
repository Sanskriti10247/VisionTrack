"""Adapter around ultralytics' BYTETracker.

Used as an independent reference implementation: running both trackers on
identical cached detections checks our ByteTrack port and shows how much
small implementation details (>= vs >, class gating) move the metrics.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from visiontrack.config import TrackerConfig
from visiontrack.tracking.base import Tracker
from visiontrack.types import Detections, TrackedObject


class _Boxes:
    """The minimal subset of ultralytics' Boxes interface that BYTETracker reads."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy.astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> _Boxes:
        return _Boxes(self.xyxy[idx], self.conf[idx], self.cls[idx])


class UltralyticsByteTracker(Tracker):
    def __init__(self, cfg: TrackerConfig | None = None, frame_rate: float = 30.0) -> None:
        super().__init__()
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.cfg = cfg or TrackerConfig()
        args = SimpleNamespace(
            track_high_thresh=self.cfg.track_high_thresh,
            track_low_thresh=self.cfg.track_low_thresh if self.cfg.use_low_score_association else 1.0,
            new_track_thresh=self.cfg.new_track_thresh,
            track_buffer=self.cfg.track_buffer,
            match_thresh=self.cfg.match_thresh,
            fuse_score=self.cfg.fuse_score,
        )
        self._tracker = BYTETracker(args, frame_rate=int(round(frame_rate)))
        self._ages: dict[int, tuple[int, int]] = {}  # id -> (start_frame, hits)
        self._frame = 0
        self._alive: set[int] = set()

    def reset(self) -> None:
        self._tracker.reset()
        self._ages.clear()
        self._frame = 0
        self._alive = set()
        self.ended_track_ids = []

    def active_track_ids(self) -> list[int]:
        t = self._tracker
        return [s.track_id for s in t.tracked_stracks + t.lost_stracks if s.is_activated]

    def update(self, detections: Detections) -> list[TrackedObject]:
        self._frame += 1
        rows = self._tracker.update(_Boxes(detections.xyxy, detections.confidence, detections.class_id))
        out = []
        for r in rows.reshape(-1, 8):
            tid = int(r[4])
            start, hits = self._ages.get(tid, (self._frame, 0))
            self._ages[tid] = (start, hits + 1)
            out.append(
                TrackedObject(
                    track_id=tid,
                    xyxy=r[:4].astype(np.float32),
                    score=float(r[5]),
                    class_id=int(r[6]),
                    age=self._frame - start + 1,
                    hits=hits + 1,
                )
            )
        alive = set(self.active_track_ids())
        self.ended_track_ids = sorted(self._alive - alive)
        self._alive = alive
        for tid in self.ended_track_ids:
            self._ages.pop(tid, None)
        return out
