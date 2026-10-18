"""ByteTrack (Zhang et al., ECCV 2022) re-implemented for a class-aware, multi-instance service.

The association logic follows the reference implementation
(github.com/ifzhang/ByteTrack, yolox/tracker/byte_tracker.py) step by step:

1. Kalman-predict all tracked and lost tracks.
2. Associate high-score detections (score > track_high_thresh) with them
   using IoU (optionally fused with the detection score).
3. Associate the *remaining tracked* tracks with low-score detections
   (track_low_thresh < score <= track_high_thresh). This is the key idea of
   ByteTrack: occluded or blurred objects often still produce a low-score box.
4. Match unconfirmed (single-frame) tracks with leftover high-score boxes;
   unmatched unconfirmed tracks are discarded.
5. Start new tracks from unmatched detections with score >= new_track_thresh.
6. Remove tracks that have been lost for more than the track buffer.

Differences from the reference code, all deliberate:
* track IDs are per tracker instance (the reference uses a global counter),
  so concurrent jobs in the API do not share an ID space;
* optional class gating so a "car" track never absorbs a "person" box;
* the class of a track is a score-weighted vote over its detections;
* removed tracks are not accumulated forever (the reference keeps them all).
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from visiontrack.config import TrackerConfig
from visiontrack.tracking.base import Tracker
from visiontrack.tracking.kalman import KalmanFilterXYAH, xyah_to_xyxy, xyxy_to_xyah
from visiontrack.tracking.matching import fuse_score, gate_classes, iou_distance, linear_assignment
from visiontrack.types import Detections, TrackedObject


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    def __init__(self, xyxy: np.ndarray, score: float, cls: int) -> None:
        self._det_xyxy = np.asarray(xyxy, dtype=np.float64)
        self.score = float(score)
        self.det_cls = int(cls)
        self.class_votes: dict[int, float] = {int(cls): float(score)}
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.state = TrackState.NEW
        self.is_activated = False
        self.track_id = 0
        self.start_frame = 0
        self.frame_id = 0
        self.tracklet_len = 0
        self.hits = 0

    # ------------------------------------------------------------- properties
    @property
    def xyxy(self) -> np.ndarray:
        if self.mean is None:
            return self._det_xyxy.copy()
        return xyah_to_xyxy(self.mean)

    @property
    def cls(self) -> int:
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def end_frame(self) -> int:
        return self.frame_id

    # -------------------------------------------------------------- lifecycle
    def activate(self, kf: KalmanFilterXYAH, frame_id: int, track_id: int) -> None:
        self.track_id = track_id
        self.mean, self.covariance = kf.initiate(xyxy_to_xyah(self._det_xyxy))
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        # Tracks born on the very first frame are trusted immediately; others
        # must be matched again on the next frame before they are reported.
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.hits = 1

    def _absorb(self, det: STrack, kf: KalmanFilterXYAH, frame_id: int) -> None:
        assert self.mean is not None and self.covariance is not None
        self.mean, self.covariance = kf.update(self.mean, self.covariance, xyxy_to_xyah(det._det_xyxy))
        self.frame_id = frame_id
        self.score = det.score
        self.class_votes[det.det_cls] = self.class_votes.get(det.det_cls, 0.0) + det.score
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.hits += 1

    def update(self, det: STrack, kf: KalmanFilterXYAH, frame_id: int) -> None:
        self.tracklet_len += 1
        self._absorb(det, kf, frame_id)

    def re_activate(self, det: STrack, kf: KalmanFilterXYAH, frame_id: int) -> None:
        self.tracklet_len = 0
        self._absorb(det, kf, frame_id)

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED

    def to_tracked_object(self) -> TrackedObject:
        return TrackedObject(
            track_id=self.track_id,
            xyxy=self.xyxy.astype(np.float32),
            score=self.score,
            class_id=self.cls,
            age=self.frame_id - self.start_frame + 1,
            hits=self.hits,
        )


def _boxes(tracks: list[STrack]) -> np.ndarray:
    return np.array([t.xyxy for t in tracks], dtype=np.float64).reshape(-1, 4)


def _joint(a: list[STrack], b: list[STrack]) -> list[STrack]:
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen]


def _sub(a: list[STrack], b: list[STrack]) -> list[STrack]:
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _remove_duplicates(
    a: list[STrack], b: list[STrack], class_aware: bool = False
) -> tuple[list[STrack], list[STrack]]:
    """When a tracked and a lost track overlap almost perfectly, keep the older one."""
    dist = iou_distance(_boxes(a), _boxes(b))
    if class_aware and dist.size:
        dist = gate_classes(dist, np.array([t.cls for t in a]), np.array([t.cls for t in b]))
    dup_a, dup_b = set(), set()
    for p, q in zip(*np.where(dist < 0.15), strict=True):
        age_p = a[p].frame_id - a[p].start_frame
        age_q = b[q].frame_id - b[q].start_frame
        if age_p > age_q:
            dup_b.add(q)
        else:
            dup_a.add(p)
    return [t for i, t in enumerate(a) if i not in dup_a], [t for i, t in enumerate(b) if i not in dup_b]


class ByteTracker(Tracker):
    def __init__(self, cfg: TrackerConfig | None = None, frame_rate: float = 30.0) -> None:
        super().__init__()
        self.cfg = cfg or TrackerConfig()
        self.frame_rate = frame_rate
        self.max_time_lost = int(frame_rate / 30.0 * self.cfg.track_buffer)
        self.kf = KalmanFilterXYAH()
        self.reset()

    def reset(self) -> None:
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self.frame_id = 0
        self._next_id = 1
        self.ended_track_ids = []
        self._alive_confirmed: set[int] = set()

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def active_track_ids(self) -> list[int]:
        return [t.track_id for t in self.tracked + self.lost if t.is_activated]

    def _cost(self, tracks: list[STrack], dets: list[STrack], fuse: bool) -> np.ndarray:
        cost = iou_distance(_boxes(tracks), _boxes(dets))
        if fuse and self.cfg.fuse_score and cost.size:
            cost = fuse_score(cost, np.array([d.score for d in dets]))
        if self.cfg.class_aware and cost.size:
            cost = gate_classes(cost, np.array([t.cls for t in tracks]), np.array([d.det_cls for d in dets]))
        return cost

    def _predict(self, tracks: list[STrack]) -> None:
        if not tracks:
            return
        mean = np.stack([t.mean for t in tracks]).copy()
        cov = np.stack([t.covariance for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.TRACKED:
                mean[i, 7] = 0.0  # freeze height velocity of lost tracks (as in the reference)
        mean, cov = self.kf.multi_predict(mean, cov)
        for i, t in enumerate(tracks):
            t.mean, t.covariance = mean[i], cov[i]

    def update(self, detections: Detections) -> list[TrackedObject]:
        cfg = self.cfg
        self.frame_id += 1
        activated, refound, newly_lost, removed = [], [], [], []

        scores = detections.confidence
        high = scores > cfg.track_high_thresh
        low = (
            (scores > cfg.track_low_thresh) & ~high if cfg.use_low_score_association else np.zeros_like(high)
        )
        dets_high = [
            STrack(b, s, c)
            for b, s, c in zip(detections.xyxy[high], scores[high], detections.class_id[high], strict=True)
        ]
        dets_low = [
            STrack(b, s, c)
            for b, s, c in zip(detections.xyxy[low], scores[low], detections.class_id[low], strict=True)
        ]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]

        # Step 1: high-score detections vs. all confirmed + lost tracks.
        pool = _joint(confirmed, self.lost)
        self._predict(pool)
        matches, u_track, u_det = linear_assignment(self._cost(pool, dets_high, fuse=True), cfg.match_thresh)
        for it, idet in matches:
            track = pool[it]
            if track.state == TrackState.TRACKED:
                track.update(dets_high[idet], self.kf, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(dets_high[idet], self.kf, self.frame_id)
                refound.append(track)

        # Step 2: low-score detections vs. tracks that were visible last frame.
        remaining = [pool[i] for i in u_track if pool[i].state == TrackState.TRACKED]
        matches, u_track2, _ = linear_assignment(
            self._cost(remaining, dets_low, fuse=False), cfg.second_match_thresh
        )
        for it, idet in matches:
            track = remaining[it]
            if track.state == TrackState.TRACKED:
                track.update(dets_low[idet], self.kf, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(dets_low[idet], self.kf, self.frame_id)
                refound.append(track)
        for it in u_track2:
            track = remaining[it]
            if track.state != TrackState.LOST:
                track.mark_lost()
                newly_lost.append(track)

        # Step 3: unconfirmed tracks vs. leftover high-score detections.
        leftover = [dets_high[i] for i in u_det]
        matches, u_unconf, u_det = linear_assignment(
            self._cost(unconfirmed, leftover, fuse=True), cfg.unconfirmed_match_thresh
        )
        for it, idet in matches:
            unconfirmed[it].update(leftover[idet], self.kf, self.frame_id)
            activated.append(unconfirmed[it])
        for it in u_unconf:
            unconfirmed[it].mark_removed()
            removed.append(unconfirmed[it])

        # Step 4: start new tracks.
        for idet in u_det:
            det = leftover[idet]
            if det.score < cfg.new_track_thresh:
                continue
            det.activate(self.kf, self.frame_id, self._new_id())
            activated.append(det)

        # Step 5: expire tracks lost for too long.
        for track in self.lost:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked = [t for t in self.tracked if t.state == TrackState.TRACKED]
        self.tracked = _joint(self.tracked, activated)
        self.tracked = _joint(self.tracked, refound)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(newly_lost)
        self.lost = _sub(self.lost, removed)
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost, cfg.class_aware)

        # A confirmed track ends when it leaves both lists (expired, or dropped as a duplicate).
        alive = {t.track_id for t in self.tracked + self.lost if t.is_activated}
        self.ended_track_ids = sorted(self._alive_confirmed - alive)
        self._alive_confirmed = alive

        out = []
        for t in self.tracked:
            if not t.is_activated:
                continue
            obj = t.to_tracked_object()
            x1, y1, x2, y2 = obj.xyxy
            if cfg.min_box_area and (x2 - x1) * (y2 - y1) < cfg.min_box_area:
                continue
            out.append(obj)
        return out
