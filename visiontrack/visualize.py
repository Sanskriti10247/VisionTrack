"""Frame annotation: boxes, persistent IDs, trails, counting lines, zones and a stats overlay."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from visiontrack.types import COCO_NAMES, Detections, TrackedObject

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR colour per track ID (golden-ratio hue walk)."""
    h = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _label(
    img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float
) -> None:
    thickness = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, thickness)
    x, y = org
    y = max(y, th + base + 2)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y), color, -1)
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    cv2.putText(
        img,
        text,
        (x + 2, y - base - 1),
        _FONT,
        scale,
        (0, 0, 0) if lum > 140 else (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(img: np.ndarray, dets: Detections, names=COCO_NAMES, color=(160, 160, 160)) -> None:
    for box, conf, cls in zip(dets.xyxy, dets.confidence, dets.class_id, strict=True):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        _label(img, f"{names[int(cls)]} {conf:.2f}", (x1, y1), color, 0.4)


def draw_frame(
    img: np.ndarray,
    tracks: list[TrackedObject],
    names=COCO_NAMES,
    trails: dict[int, list[tuple[float, float]]] | None = None,
    lines: list[tuple[str, tuple[float, float], tuple[float, float], dict]] | None = None,
    zones: list[tuple[str, list[tuple[float, float]], int]] | None = None,
    hud: list[str] | None = None,
) -> np.ndarray:
    """Draws everything in place and returns the image.

    lines: (name, p1, p2, {"in": n, "out": m}); zones: (name, polygon, occupancy).
    """
    scale = max(0.45, min(img.shape[:2]) / 1400)
    thick = max(1, int(round(scale * 3)))

    if zones:
        overlay = img.copy()
        for _, poly, occ in zones:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 160, 255) if occ else (255, 200, 0))
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, dst=img)
        for name, poly, occ in zones:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(img, [pts], True, (255, 200, 0), thick, cv2.LINE_AA)
            _label(img, f"{name}: {occ} inside", tuple(pts.min(axis=0)), (255, 200, 0), scale)

    if lines:
        for name, p1, p2, counts in lines:
            a = (int(p1[0]), int(p1[1]))
            b = (int(p2[0]), int(p2[1]))
            cv2.line(img, a, b, (0, 0, 255), thick + 1, cv2.LINE_AA)
            for p in (a, b):
                cv2.circle(img, p, thick + 3, (0, 0, 255), -1, cv2.LINE_AA)
            _label(
                img, f"{name}  in {counts.get('in', 0)} | out {counts.get('out', 0)}", a, (0, 0, 255), scale
            )

    if trails:
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue
            color = track_color(tid)
            arr = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [arr], False, color, max(1, thick - 1), cv2.LINE_AA)

    for t in tracks:
        color = track_color(t.track_id)
        x1, y1, x2, y2 = (int(round(v)) for v in t.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
        name = names[t.class_id] if 0 <= t.class_id < len(names) else str(t.class_id)
        _label(img, f"#{t.track_id} {name} {t.score:.2f}", (x1, y1), color, scale * 0.9)

    if hud:
        y = int(28 * scale / 0.6)
        for text in hud:
            _label(img, text, (8, y), (30, 30, 30), scale)
            y += int(28 * scale / 0.6)
    return img
