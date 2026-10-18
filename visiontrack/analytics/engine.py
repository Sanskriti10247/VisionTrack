"""Track-based video analytics: unique counts, line crossings, zone visits, dwell and trajectories.

Everything here consumes *tracks*, never raw detections, so an object that
stays in view for 300 frames is counted once. The engine is only as good
as the tracker's identities, though:

* an identity switch or a track that is lost and restarted with a new ID
  shows up as an extra unique object / an extra zone visit;
* two people merged into one track during an occlusion are counted once.

The summary output carries these caveats explicitly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from visiontrack.analytics.geometry import (
    point_in_polygon,
    scale_point,
    segments_intersect,
    signed_distance_to_line,
)
from visiontrack.config import AnalyticsConfig, LineConfig, ZoneConfig
from visiontrack.types import COCO_NAMES, TrackedObject

Point = tuple[float, float]

LIMITATIONS = [
    "Counts are per track ID. Identity switches and tracks that are lost and re-acquired under a new ID "
    "inflate unique counts and zone visits; merged identities during occlusion deflate them.",
    "Distances and speeds are in image pixels. No camera calibration is applied, so they are not physical "
    "units and depend on the object's distance from the camera.",
    "Dwell time is measured from confirmed entry to confirmed exit. Visits closed because the track was lost "
    "or the video ended are marked and are lower bounds.",
]


@dataclass
class _LineState:
    side: int = 0  # last confirmed side: +1, -1, or 0 if not yet known
    side_point: Point | None = None  # anchor position when that side was confirmed
    counted: set[str] = field(default_factory=set)


@dataclass
class _ZoneState:
    inside: bool = False
    streak: int = 0  # consecutive observations disagreeing with `inside`
    streak_start: tuple[int, float] | None = None  # (frame, ts) of the first disagreeing observation
    entered_at: tuple[int, float] | None = None
    started_inside: bool = False
    seen: bool = False


@dataclass
class _TrackRecord:
    track_id: int
    first_frame: int
    first_ts: float
    last_frame: int
    last_ts: float
    hits: int = 0
    class_frames: Counter = field(default_factory=Counter)
    score_sum: float = 0.0
    points: list[tuple[int, float, float, float]] = field(default_factory=list)  # (frame, ts, x, y)
    path_length: float = 0.0
    lines: dict[str, _LineState] = field(default_factory=dict)
    zones: dict[str, _ZoneState] = field(default_factory=dict)
    ended: bool = False

    @property
    def class_id(self) -> int:
        return self.class_frames.most_common(1)[0][0]


@dataclass
class _Line:
    cfg: LineConfig
    p1: Point
    p2: Point


@dataclass
class _Zone:
    cfg: ZoneConfig
    polygon: list[Point]


class AnalyticsEngine:
    def __init__(
        self,
        cfg: AnalyticsConfig,
        fps: float,
        frame_size: tuple[int, int],
        names: tuple[str, ...] | list[str] = COCO_NAMES,
    ) -> None:
        self.cfg = cfg
        self.fps = fps
        self.width, self.height = frame_size
        self.names = tuple(names)
        self.lines = [self._resolve_line(line) for line in cfg.lines]
        self.zones = [self._resolve_zone(zone) for zone in cfg.zones]
        self.tracks: dict[int, _TrackRecord] = {}
        self.events: list[dict] = []
        self.frames_processed = 0
        self.last_frame = -1
        self.last_ts = 0.0

    # ---------------------------------------------------------------- set-up
    def _pt(self, p: Point) -> Point:
        if self.cfg.normalized_coordinates:
            return scale_point(p, self.width, self.height)
        return (float(p[0]), float(p[1]))

    def _resolve_line(self, line: LineConfig) -> _Line:
        p1, p2 = self._pt(line.p1), self._pt(line.p2)
        if p1 == p2:
            raise ValueError(f"line '{line.name}' has identical endpoints")
        return _Line(line, p1, p2)

    def _resolve_zone(self, zone: ZoneConfig) -> _Zone:
        return _Zone(zone, [self._pt(p) for p in zone.polygon])

    def _class_name(self, cls: int) -> str:
        return self.names[cls] if 0 <= cls < len(self.names) else str(cls)

    def _anchor(self, t: TrackedObject) -> Point:
        return t.anchor if self.cfg.anchor == "bottom_center" else t.center

    # ---------------------------------------------------------------- update
    def update(
        self,
        frame_index: int,
        timestamp_s: float,
        tracks: list[TrackedObject],
        ended_ids: list[int] | tuple = (),
    ) -> list[dict]:
        """Ingests one frame of confirmed tracks and returns the events it produced."""
        new_events: list[dict] = []
        self.frames_processed += 1
        self.last_frame, self.last_ts = frame_index, timestamp_s

        for t in tracks:
            rec = self.tracks.get(t.track_id)
            if rec is None:
                rec = _TrackRecord(t.track_id, frame_index, timestamp_s, frame_index, timestamp_s)
                self.tracks[t.track_id] = rec
            rec.last_frame, rec.last_ts = frame_index, timestamp_s
            rec.hits += 1
            rec.class_frames[t.class_id] += 1
            rec.score_sum += t.score
            p = self._anchor(t)
            self._add_point(rec, frame_index, timestamp_s, p)
            for line in self.lines:
                ev = self._update_line(rec, line, p, frame_index, timestamp_s)
                if ev:
                    new_events.append(ev)
            for zone in self.zones:
                ev = self._update_zone(rec, zone, p, frame_index, timestamp_s)
                if ev:
                    new_events.append(ev)

        for tid in ended_ids:
            rec = self.tracks.get(tid)
            if rec is not None and not rec.ended:
                rec.ended = True
                new_events.extend(self._close_visits(rec, "track_lost"))

        self.events.extend(new_events)
        return new_events

    def finalize(self) -> list[dict]:
        """Closes zone visits that are still open when the video ends."""
        closing: list[dict] = []
        for rec in self.tracks.values():
            if not rec.ended:
                closing.extend(self._close_visits(rec, "video_end"))
        self.events.extend(closing)
        return closing

    def _add_point(self, rec: _TrackRecord, frame: int, ts: float, p: Point) -> None:
        if rec.points:
            _, _, px, py = rec.points[-1]
            rec.path_length += float(np.hypot(p[0] - px, p[1] - py))
        rec.points.append((frame, ts, p[0], p[1]))
        if len(rec.points) > self.cfg.trajectory_max_points:
            # Decimate rather than truncate so the whole path keeps its shape.
            rec.points = rec.points[::2] if len(rec.points) % 2 else rec.points[::2] + [rec.points[-1]]

    def _counts_toward_totals(self, rec: _TrackRecord) -> bool:
        return rec.hits >= self.cfg.min_track_hits

    def _event(self, kind: str, rec: _TrackRecord, frame: int, ts: float, **extra) -> dict:
        return {
            "type": kind,
            "frame_index": int(frame),
            "timestamp_s": round(float(ts), 3),
            "track_id": int(rec.track_id),
            "class_name": self._class_name(rec.class_id),
            **extra,
        }

    def _update_line(self, rec: _TrackRecord, line: _Line, p: Point, frame: int, ts: float) -> dict | None:
        st = rec.lines.setdefault(line.cfg.name, _LineState())
        d = signed_distance_to_line(p, line.p1, line.p2)
        if abs(d) < line.cfg.hysteresis_px:
            return None  # inside the dead band around the line: side unchanged
        side = 1 if d > 0 else -1
        if st.side == 0:
            st.side, st.side_point = side, p
            return None
        if side == st.side:
            st.side_point = p
            return None
        # Side flipped: it is a crossing only if the movement passed through the segment itself.
        prev = st.side_point
        st.side, st.side_point = side, p
        if prev is None or not segments_intersect(prev, p, line.p1, line.p2):
            return None
        # "in" = moving from the positive to the negative side, i.e. along the direction obtained by
        # rotating p1->p2 90 degrees clockwise on screen (downwards for a left-to-right line).
        direction = "in" if side < 0 else "out"
        # Repeated crossings are always logged; with count_once_per_track only
        # the first one per direction counts. min_track_hits is applied in summary().
        counted = direction not in st.counted if self.cfg.count_once_per_track else True
        if counted:
            st.counted.add(direction)
        return self._event(
            "line_crossing", rec, frame, ts, line=line.cfg.name, direction=direction, counted=counted
        )

    def _update_zone(self, rec: _TrackRecord, zone: _Zone, p: Point, frame: int, ts: float) -> dict | None:
        st = rec.zones.setdefault(zone.cfg.name, _ZoneState())
        inside = point_in_polygon(p, zone.polygon)
        if not st.seen:
            st.seen = True
            st.started_inside = inside
        if inside == st.inside:
            st.streak, st.streak_start = 0, None
            return None
        if st.streak == 0:
            st.streak_start = (frame, ts)
        st.streak += 1
        needed = zone.cfg.min_frames_inside if inside else zone.cfg.min_frames_outside
        if st.streak < needed:
            return None
        start_frame, start_ts = st.streak_start or (frame, ts)
        st.inside, st.streak, st.streak_start = inside, 0, None
        if inside:
            st.entered_at = (start_frame, start_ts)
            return self._event(
                "zone_enter",
                rec,
                start_frame,
                start_ts,
                zone=zone.cfg.name,
                first_seen_inside=st.started_inside,
            )
        return self._exit_event(rec, zone.cfg.name, st, start_frame, start_ts, "exited")

    def _exit_event(
        self, rec: _TrackRecord, zone: str, st: _ZoneState, frame: int, ts: float, reason: str
    ) -> dict:
        entered = st.entered_at
        st.inside, st.entered_at = False, None
        dwell = (ts - entered[1]) if entered else None
        started_inside = st.started_inside
        st.started_inside = False  # later visits by the same track start outside
        return self._event(
            "zone_exit",
            rec,
            frame,
            ts,
            zone=zone,
            reason=reason,
            dwell_s=round(dwell, 3) if dwell is not None else None,
            dwell_is_lower_bound=reason != "exited" or started_inside,
        )

    def _close_visits(self, rec: _TrackRecord, reason: str) -> list[dict]:
        out = []
        for name, st in rec.zones.items():
            if st.inside:
                out.append(self._exit_event(rec, name, st, rec.last_frame, rec.last_ts, reason))
        return out

    # --------------------------------------------------------------- summary
    def live_counts(self) -> dict:
        """Cheap running totals for overlays and streaming."""
        unique = sum(1 for r in self.tracks.values() if self._counts_toward_totals(r))
        lines = {}
        for line in self.lines:
            c = {"in": 0, "out": 0}
            for e in self.events:
                if (
                    e["type"] == "line_crossing"
                    and e["line"] == line.cfg.name
                    and e["counted"]
                    and self._counts_toward_totals(self.tracks[e["track_id"]])
                ):
                    c[e["direction"]] += 1
            lines[line.cfg.name] = c
        zones = {
            z.cfg.name: sum(1 for r in self.tracks.values() if r.zones.get(z.cfg.name, _ZoneState()).inside)
            for z in self.zones
        }
        return {"unique_objects": unique, "lines": lines, "zone_occupancy": zones}

    def summary(self) -> dict:
        counted = [r for r in self.tracks.values() if self._counts_toward_totals(r)]
        counted_ids = {r.track_id for r in counted}
        by_class = Counter(self._class_name(r.class_id) for r in counted)

        per_class: dict[str, dict] = {}
        groups: dict[str, list[_TrackRecord]] = defaultdict(list)
        for r in counted:
            groups[self._class_name(r.class_id)].append(r)
        for name, recs in sorted(groups.items()):
            durations = [r.last_ts - r.first_ts for r in recs]
            speeds = [self._speed(r) for r in recs if self._speed(r) is not None]
            per_class[name] = {
                "unique_tracks": len(recs),
                "mean_track_duration_s": round(float(np.mean(durations)), 3),
                "mean_speed_px_per_s": round(float(np.mean(speeds)), 2) if speeds else None,
            }

        lines_out = []
        for line in self.lines:
            evs = [e for e in self.events if e["type"] == "line_crossing" and e["line"] == line.cfg.name]
            # Only tracks that pass the min_track_hits bar count, so a one-frame
            # false positive that "crosses" because of jitter is excluded.
            valid = [e for e in evs if e["counted"] and e["track_id"] in counted_ids]
            counts = {}
            for direction in ("in", "out"):
                d = [e for e in valid if e["direction"] == direction]
                counts[direction] = {
                    "total": len(d),
                    "by_class": dict(
                        Counter(self._class_name(self.tracks[e["track_id"]].class_id) for e in d)
                    ),
                }
            lines_out.append(
                {
                    "name": line.cfg.name,
                    "p1": list(line.p1),
                    "p2": list(line.p2),
                    "counts": counts,
                    "net_in_minus_out": counts["in"]["total"] - counts["out"]["total"],
                    "raw_crossing_events": len(evs),
                }
            )

        zones_out = []
        for zone in self.zones:
            enters = [e for e in self.events if e["type"] == "zone_enter" and e["zone"] == zone.cfg.name]
            exits = [e for e in self.events if e["type"] == "zone_exit" and e["zone"] == zone.cfg.name]
            visitors = {e["track_id"] for e in enters if e["track_id"] in counted_ids}
            complete = [
                e["dwell_s"] for e in exits if e["dwell_s"] is not None and not e["dwell_is_lower_bound"]
            ]
            all_dwell = [e["dwell_s"] for e in exits if e["dwell_s"] is not None]
            zones_out.append(
                {
                    "name": zone.cfg.name,
                    "polygon": [list(p) for p in zone.polygon],
                    "unique_visitors": {
                        "total": len(visitors),
                        "by_class": dict(
                            Counter(self._class_name(self.tracks[t].class_id) for t in visitors)
                        ),
                    },
                    "entries": len(enters),
                    "exits": len(exits),
                    "dwell_s": {
                        "complete_visits": len(complete),
                        "mean_complete": round(float(np.mean(complete)), 3) if complete else None,
                        "max_any": round(float(np.max(all_dwell)), 3) if all_dwell else None,
                    },
                }
            )

        return {
            "frames_processed": self.frames_processed,
            "unique_objects": {"total": len(counted), "by_class": dict(by_class)},
            "per_class": per_class,
            "lines": lines_out,
            "zones": zones_out,
            "counting_rules": {
                "min_track_hits": self.cfg.min_track_hits,
                "count_once_per_track": self.cfg.count_once_per_track,
                "anchor": self.cfg.anchor,
            },
            "limitations": LIMITATIONS,
        }

    def _speed(self, rec: _TrackRecord) -> float | None:
        dt = rec.last_ts - rec.first_ts
        return rec.path_length / dt if dt > 0 else None

    def track_table(self) -> list[dict]:
        rows = []
        for r in sorted(self.tracks.values(), key=lambda r: r.track_id):
            speed = self._speed(r)
            rows.append(
                {
                    "track_id": r.track_id,
                    "class_name": self._class_name(r.class_id),
                    "first_frame": r.first_frame,
                    "last_frame": r.last_frame,
                    "first_ts": round(r.first_ts, 3),
                    "last_ts": round(r.last_ts, 3),
                    "duration_s": round(r.last_ts - r.first_ts, 3),
                    "hits": r.hits,
                    "mean_score": round(r.score_sum / max(r.hits, 1), 4),
                    "path_length_px": round(r.path_length, 1),
                    "mean_speed_px_per_s": round(speed, 2) if speed is not None else None,
                    "counted": self._counts_toward_totals(r),
                }
            )
        return rows

    def trajectories(self, min_points: int = 2) -> dict[int, list[list[float]]]:
        """{track_id: [[frame, ts, x, y], ...]} for tracks with at least ``min_points`` points."""
        return {
            r.track_id: [[f, round(t, 3), round(x, 1), round(y, 1)] for f, t, x, y in r.points]
            for r in self.tracks.values()
            if len(r.points) >= min_points
        }

    def recent_trail(self, track_id: int, n: int) -> list[Point]:
        rec = self.tracks.get(track_id)
        if rec is None:
            return []
        return [(x, y) for _, _, x, y in rec.points[-n:]]
