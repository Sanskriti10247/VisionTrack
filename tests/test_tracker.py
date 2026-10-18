import numpy as np
import pytest

from visiontrack.config import TrackerConfig
from visiontrack.tracking import build_tracker
from visiontrack.tracking.bytetrack import ByteTracker
from visiontrack.tracking.kalman import KalmanFilterXYAH, xyah_to_xyxy, xyxy_to_xyah
from visiontrack.tracking.matching import box_iou, linear_assignment
from visiontrack.types import Detections


def dets(*items):
    """items: (x1, y1, x2, y2, score[, cls])"""
    if not items:
        return Detections.empty()
    boxes = [i[:4] for i in items]
    scores = [i[4] for i in items]
    classes = [i[5] if len(i) > 5 else 0 for i in items]
    return Detections(boxes, scores, classes)


def moving_box(t, x0=100.0, y0=100.0, vx=5.0, vy=0.0, w=40.0, h=80.0, score=0.9, cls=0):
    x, y = x0 + vx * t, y0 + vy * t
    return (x, y, x + w, y + h, score, cls)


# ---------------------------------------------------------------- matching


def test_box_iou_basic():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 1 / 3, 0.0]])


def test_linear_assignment_respects_threshold():
    cost = np.array([[0.1, 0.9], [0.95, 0.85]])
    matches, ur, uc = linear_assignment(cost, thresh=0.8)
    assert matches.tolist() == [[0, 0]]
    assert ur.tolist() == [1] and uc.tolist() == [1]


def test_linear_assignment_prefers_global_optimum():
    # Greedy would take (0,0)=0.1 and then be forced into (1,1)=0.9 (> thresh);
    # the optimal thresholded matching takes both cheap off-diagonal pairs.
    cost = np.array([[0.1, 0.2], [0.3, 0.9]])
    matches, _, _ = linear_assignment(cost, thresh=0.5)
    assert sorted(map(tuple, matches.tolist())) == [(0, 1), (1, 0)]


def test_linear_assignment_empty():
    m, ur, uc = linear_assignment(np.zeros((0, 3)), 0.5)
    assert m.shape == (0, 2) and len(ur) == 0 and uc.tolist() == [0, 1, 2]


# ------------------------------------------------------------------ kalman


def test_box_conversion_roundtrip():
    box = np.array([10.0, 20.0, 50.0, 100.0])
    np.testing.assert_allclose(xyah_to_xyxy(xyxy_to_xyah(box)), box)


def test_kalman_learns_constant_velocity():
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(xyxy_to_xyah(np.array([0, 0, 40, 80.0])))
    for t in range(1, 30):
        mean, cov = kf.multi_predict(mean[None], cov[None])
        mean, cov = mean[0], cov[0]
        mean, cov = kf.update(mean, cov, xyxy_to_xyah(np.array([5.0 * t, 0, 5.0 * t + 40, 80.0])))
    assert mean[4] == pytest.approx(5.0, abs=0.2)  # learned vx
    pred, _ = kf.multi_predict(mean[None], cov[None])
    assert xyah_to_xyxy(pred[0])[0] == pytest.approx(150.0, abs=1.0)


# --------------------------------------------------------------- lifecycle


def test_single_object_keeps_one_id():
    tr = ByteTracker(TrackerConfig())
    ids = set()
    for t in range(50):
        out = tr.update(dets(moving_box(t)))
        assert len(out) == 1
        ids.add(out[0].track_id)
    assert ids == {1}
    assert out[0].hits == 50 and out[0].age == 50


def test_track_born_after_first_frame_needs_confirmation():
    tr = ByteTracker(TrackerConfig())
    assert tr.update(dets()) == []
    assert tr.update(dets(moving_box(0))) == []  # unconfirmed
    out = tr.update(dets(moving_box(1)))
    assert [o.track_id for o in out] == [1]


def test_single_frame_false_positive_never_reported():
    tr = ByteTracker(TrackerConfig())
    tr.update(dets())
    assert tr.update(dets((10, 10, 50, 90, 0.95))) == []
    for _ in range(5):
        assert tr.update(dets()) == []
    assert tr.active_track_ids() == []


def test_low_confidence_detection_does_not_start_a_track():
    tr = ByteTracker(TrackerConfig(new_track_thresh=0.6))
    for t in range(5):
        assert tr.update(dets(moving_box(t, score=0.55))) == []


def test_short_occlusion_keeps_id_and_long_one_ends_track():
    tr = ByteTracker(TrackerConfig(track_buffer=10), frame_rate=30)
    for t in range(10):
        tr.update(dets(moving_box(t)))
    for _ in range(5):  # occluded for 5 frames (< buffer)
        assert tr.update(dets()) == []
    out = tr.update(dets(moving_box(15)))  # reappears where the Kalman filter expects it
    assert [o.track_id for o in out] == [1]

    ended_at = None
    for k in range(20):
        tr.update(dets())
        if tr.ended_track_ids:
            ended_at = k
            assert tr.ended_track_ids == [1]
            break
    assert ended_at is not None and ended_at >= 9
    out = []
    for t in range(40, 43):
        out = tr.update(dets(moving_box(t)))
    assert out and out[0].track_id != 1


def test_low_score_association_bridges_weak_detections():
    """ByteTrack's second stage keeps a track alive through low-score (e.g. occluded) boxes."""

    def run(use_low: bool) -> list[int]:
        tr = ByteTracker(TrackerConfig(use_low_score_association=use_low))
        visible = []
        for t in range(30):
            score = 0.3 if 10 <= t < 20 else 0.9
            visible.append(len(tr.update(dets(moving_box(t, score=score)))))
        return visible

    with_low = run(True)
    without = run(False)
    assert all(v == 1 for v in with_low)
    assert all(v == 0 for v in without[10:20])


def test_class_gating_prevents_cross_class_matches():
    tr = ByteTracker(TrackerConfig(class_aware=True))
    for t in range(5):
        tr.update(dets(moving_box(t, cls=0)))
    # Same place, different class: must not inherit the person's ID.
    out = []
    for t in range(5, 8):
        out = tr.update(dets(moving_box(t, cls=2)))
    assert out and all(o.track_id != 1 for o in out)


def test_crossing_objects_keep_identities():
    tr = ByteTracker(TrackerConfig())
    direction = {}
    out = []
    for t in range(40):
        right = moving_box(t, x0=0, vx=8)
        left = moving_box(t, x0=320, vx=-8, y0=110)
        out = tr.update(dets(right, left))
        if t == 0:
            direction = {o.track_id: "right" if o.xyxy[0] < 100 else "left" for o in out}
    final_x = {o.track_id: float(o.xyxy[0]) for o in out}
    assert set(final_x) == set(direction)  # no new IDs after the boxes overlapped
    for tid, d in direction.items():
        assert (final_x[tid] > 250) if d == "right" else (final_x[tid] < 70)


def test_ids_are_per_instance():
    a, b = ByteTracker(), ByteTracker()
    a.update(dets(moving_box(0)))
    out = b.update(dets(moving_box(0)))
    assert out[0].track_id == 1


def test_ultralytics_adapter_agrees_on_simple_scene():
    cfg = TrackerConfig(type="ultralytics_bytetrack")
    tr = build_tracker(cfg)
    ids = set()
    for t in range(30):
        for o in tr.update(dets(moving_box(t), moving_box(t, x0=400, vx=-3))):
            ids.add(o.track_id)
    assert len(ids) == 2
