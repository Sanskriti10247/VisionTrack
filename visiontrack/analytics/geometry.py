"""Small 2D geometry helpers for line crossings and polygon zones (pixel coordinates)."""

from __future__ import annotations

import numpy as np

Point = tuple[float, float]


def signed_distance_to_line(p: Point, a: Point, b: Point) -> float:
    """Signed perpendicular distance of p from the infinite line a->b.

    Positive on the left of a->b in image coordinates (y grows downwards),
    i.e. the side the line's normal (dy, -dx) points to.
    """
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    norm = float(np.hypot(dx, dy))
    if norm == 0:
        raise ValueError("line endpoints must differ")
    return ((p[0] - ax) * dy - (p[1] - ay) * dx) / norm


def projection_parameter(p: Point, a: Point, b: Point) -> float:
    """Position of p's projection along a->b: 0 at a, 1 at b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    return ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if segment p1-p2 intersects segment q1-q2 (touching counts)."""

    def orient(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: Point, b: Point, c: Point) -> bool:
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    return (
        (d1 == 0 and on_segment(q1, q2, p1))
        or (d2 == 0 and on_segment(q1, q2, p2))
        or (d3 == 0 and on_segment(p1, p2, q1))
        or (d4 == 0 and on_segment(p1, p2, q2))
    )


def point_in_polygon(p: Point, polygon: list[Point]) -> bool:
    """Even-odd ray casting; points exactly on an edge may fall either way."""
    x, y = p
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def scale_point(p: Point, width: int, height: int) -> Point:
    return (float(p[0]) * width, float(p[1]) * height)
