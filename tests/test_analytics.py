import numpy as np
import pytest

from visiontrack.analytics.engine import AnalyticsEngine
from visiontrack.analytics.geometry import point_in_polygon, segments_intersect, signed_distance_to_line
from visiontrack.config import AnalyticsConfig, LineConfig, ZoneConfig
from visiontrack.types import TrackedObject

FPS = 10.0


def obj(tid, cx, by, cls=0, w=20, h=40, score=0.9):
    """Track whose bottom-centre anchor is at (cx, by)."""
    return TrackedObject(tid, np.array([cx - w / 2, by - h, cx + w / 2, by], np.float32), score, cls, 1, 1)


def engine(lines=(), zones=(), **kw):
    cfg = AnalyticsConfig(lines=list(lines), zones=list(zones), **kw)
    return AnalyticsEngine(cfg, FPS, (640, 480))


def feed(eng, positions, tid=1, start=0, cls=0):
    """positions: list of (x, y) anchors, one per frame; None = track not visible."""
    for k, p in enumerate(positions):
        f = start + k
        eng.update(f, f / FPS, [] if p is None else [obj(tid, *p, cls=cls)])


H_LINE = LineConfig(name="L", p1=(100, 200), p2=(500, 200), hysteresis_px=4)
ZONE = ZoneConfig(
    name="Z",
    polygon=[(200, 100), (400, 100), (400, 300), (200, 300)],
    min_frames_inside=3,
    min_frames_outside=3,
)


# ---------------------------------------------------------------- geometry


def test_geometry_helpers():
    assert signed_distance_to_line((300, 150), (100, 200), (500, 200)) == pytest.approx(50)
    assert signed_distance_to_line((300, 260), (100, 200), (500, 200)) == pytest.approx(-60)
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
    assert not segments_intersect((0, 0), (1, 1), (5, 5), (6, 0))
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq) and not point_in_polygon((15, 5), sq)


# ------------------------------------------------------------ line crossing


def test_downward_crossing_counts_in_once():
    eng = engine([H_LINE])
    feed(eng, [(300, y) for y in range(150, 260, 10)])
    lc = [e for e in eng.events if e["type"] == "line_crossing"]
    assert len(lc) == 1 and lc[0]["direction"] == "in" and lc[0]["counted"]
    s = eng.summary()["lines"][0]
    assert s["counts"]["in"]["total"] == 1 and s["counts"]["out"]["total"] == 0


def test_jitter_inside_dead_band_is_ignored():
    eng = engine([H_LINE])
    ys = [180, 190] + [198, 202, 197, 203, 199, 201] * 5  # hovers within +/- 3 px of the line
    feed(eng, [(300, y) for y in ys])
    assert not [e for e in eng.events if e["type"] == "line_crossing"]


def test_repeated_crossings_are_logged_but_counted_once_per_direction():
    eng = engine([H_LINE])
    ys = [150, 250, 150, 250, 150, 250]
    feed(eng, [(300, y) for y in ys])
    lc = [e for e in eng.events if e["type"] == "line_crossing"]
    assert [e["direction"] for e in lc] == ["in", "out", "in", "out", "in"]
    assert [e["counted"] for e in lc] == [True, True, False, False, False]
    counts = eng.summary()["lines"][0]["counts"]
    assert counts["in"]["total"] == 1 and counts["out"]["total"] == 1


def test_counting_every_crossing_when_configured():
    eng = engine([H_LINE], count_once_per_track=False)
    feed(eng, [(300, y) for y in (150, 250, 150, 250)])
    counts = eng.summary()["lines"][0]["counts"]
    assert counts["in"]["total"] == 2 and counts["out"]["total"] == 1


def test_crossing_beyond_segment_end_is_not_counted():
    eng = engine([H_LINE])
    feed(eng, [(600, y) for y in range(150, 260, 10)])  # passes x=600, line spans x 100..500
    assert not [e for e in eng.events if e["type"] == "line_crossing"]


def test_short_lived_tracks_are_excluded_from_counts():
    eng = engine([H_LINE], min_track_hits=5)
    feed(eng, [(300, 190), (300, 210)])  # 2-frame blip crossing the line
    assert eng.summary()["lines"][0]["counts"]["in"]["total"] == 0
    assert eng.summary()["unique_objects"]["total"] == 0


def test_crossing_detected_across_a_tracking_gap():
    eng = engine([H_LINE])
    feed(eng, [(300, 150), (300, 160), None, None, (300, 240), (300, 250)])
    assert [e["direction"] for e in eng.events if e["type"] == "line_crossing"] == ["in"]


def test_normalized_line_coordinates():
    line = LineConfig(name="N", p1=(0.0, 0.5), p2=(1.0, 0.5))
    eng = engine([line], normalized_coordinates=True)
    assert eng.lines[0].p1 == (0.0, 240.0) and eng.lines[0].p2 == (640.0, 240.0)


# --------------------------------------------------------------------- zones


def test_zone_entry_exit_and_dwell():
    eng = engine(zones=[ZONE])
    path = [(100, 200)] * 3 + [(300, 200)] * 20 + [(500, 200)] * 5  # outside, inside 2 s, outside
    feed(eng, path)
    enters = [e for e in eng.events if e["type"] == "zone_enter"]
    exits = [e for e in eng.events if e["type"] == "zone_exit"]
    assert len(enters) == 1 and len(exits) == 1
    assert enters[0]["frame_index"] == 3  # timestamped at the first inside frame, not at confirmation
    assert exits[0]["dwell_s"] == pytest.approx(2.0)
    assert exits[0]["reason"] == "exited" and not exits[0]["dwell_is_lower_bound"]
    z = eng.summary()["zones"][0]
    assert z["unique_visitors"]["total"] == 1 and z["dwell_s"]["mean_complete"] == pytest.approx(2.0)


def test_boundary_jitter_does_not_create_extra_visits():
    eng = engine(zones=[ZONE])
    inside, outside = (300, 200), (100, 200)
    path = [outside] * 3 + [inside] * 5 + [outside, inside] * 6 + [inside] * 3
    feed(eng, path)
    assert len([e for e in eng.events if e["type"] == "zone_enter"]) == 1
    assert not [e for e in eng.events if e["type"] == "zone_exit"]


def test_track_lost_inside_zone_closes_visit_as_lower_bound():
    eng = engine(zones=[ZONE])
    feed(eng, [(100, 200)] * 2 + [(300, 200)] * 10)
    eng.update(12, 1.2, [], ended_ids=[1])
    exits = [e for e in eng.events if e["type"] == "zone_exit"]
    assert len(exits) == 1 and exits[0]["reason"] == "track_lost" and exits[0]["dwell_is_lower_bound"]


def test_video_end_closes_open_visits():
    eng = engine(zones=[ZONE])
    feed(eng, [(300, 200)] * 10)  # first seen inside
    eng.finalize()
    enters = [e for e in eng.events if e["type"] == "zone_enter"]
    exits = [e for e in eng.events if e["type"] == "zone_exit"]
    assert enters[0]["first_seen_inside"] is True
    assert exits[0]["reason"] == "video_end" and exits[0]["dwell_is_lower_bound"]


# ------------------------------------------------------------- unique counts


def test_unique_counts_per_class_and_track_table():
    eng = engine()
    for f in range(10):
        eng.update(f, f / FPS, [obj(1, 100 + f, 200, cls=0), obj(2, 300, 200, cls=2)])
    eng.update(10, 1.0, [obj(3, 50, 50, cls=0)])  # one-frame blip: below min_track_hits
    s = eng.summary()
    assert s["unique_objects"] == {"total": 2, "by_class": {"person": 1, "car": 1}}
    rows = {r["track_id"]: r for r in eng.track_table()}
    assert rows[1]["mean_speed_px_per_s"] == pytest.approx(10.0)
    assert rows[3]["counted"] is False


def test_trajectory_decimation_keeps_endpoints():
    eng = engine(trajectory_max_points=10)
    feed(eng, [(x, 100) for x in range(100)])
    traj = eng.trajectories()[1]
    assert len(traj) <= 10
    assert traj[0][2] == 0 and traj[-1][2] == 99
