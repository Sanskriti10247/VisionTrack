"""Association helpers: IoU cost matrices and thresholded linear assignment."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def iou_distance(track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    return 1.0 - box_iou(track_boxes, det_boxes)


def fuse_score(cost: np.ndarray, det_scores: np.ndarray) -> np.ndarray:
    """ByteTrack's score fusion: similarity = IoU * detection score."""
    if cost.size == 0:
        return cost
    sim = (1.0 - cost) * np.asarray(det_scores)[None, :]
    return 1.0 - sim


def gate_classes(cost: np.ndarray, track_classes: np.ndarray, det_classes: np.ndarray) -> np.ndarray:
    """Makes cross-class pairs unassignable."""
    if cost.size == 0:
        return cost
    mismatch = np.asarray(track_classes)[:, None] != np.asarray(det_classes)[None, :]
    out = cost.copy()
    out[mismatch] = np.inf
    return out


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-cost matching where any pair costing more than ``thresh`` stays unmatched.

    Mirrors ``lap.lapjv(cost, extend_cost=True, cost_limit=thresh)`` used by
    the reference ByteTrack: the matrix is padded with dummy rows/columns of
    cost ``thresh / 2`` so leaving a track *and* a detection unmatched
    (thresh in total) is preferred over any pair costing more than ``thresh``.
    Returns (matches (K, 2), unmatched_rows, unmatched_cols).
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=int), np.arange(n), np.arange(m)
    big = 1e6
    c = np.where(np.isfinite(cost), cost, big)
    ext = np.full((n + m, n + m), thresh / 2.0)
    ext[n:, m:] = 0.0
    ext[:n, :m] = c
    rows, cols = linear_sum_assignment(ext)
    matches = [
        (r, c_) for r, c_ in zip(rows, cols, strict=True) if r < n and c_ < m and cost[r, c_] <= thresh
    ]
    matched_r = {r for r, _ in matches}
    matched_c = {c_ for _, c_ in matches}
    um_r = np.array([i for i in range(n) if i not in matched_r], dtype=int)
    um_c = np.array([j for j in range(m) if j not in matched_c], dtype=int)
    return np.array(matches, dtype=int).reshape(-1, 2), um_r, um_c
