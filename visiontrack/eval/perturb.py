"""Deterministic image corruptions for controlled robustness experiments.

These are applied to the *same* frames with the *same* ground truth, so any
metric change is caused by the corruption alone. They approximate, not
replace, real footage with those conditions (MOT17-10 is real low light +
blur, MOT17-13 real fast motion; see SEQUENCE_CONDITIONS).
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


def motion_blur(img: np.ndarray, length: int = 15, angle_deg: float = 0.0) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= k.sum()
    return cv2.filter2D(img, -1, k)


def low_light(
    img: np.ndarray, gain: float = 0.3, gamma: float = 1.5, noise_sigma: float = 6.0, seed: int = 0
) -> np.ndarray:
    """Darkens (gamma + gain) and adds sensor-like Gaussian noise."""
    x = (img.astype(np.float32) / 255.0) ** gamma * gain * 255.0
    rng = np.random.default_rng(seed)
    x += rng.normal(0.0, noise_sigma, size=x.shape).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


def downscale(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Loses detail as a lower-resolution camera would, keeping the frame size (and GT) unchanged."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def jpeg(img: np.ndarray, quality: int = 15) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# name -> f(img, frame_index)
PERTURBATIONS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "motion_blur_15": lambda img, i: motion_blur(img, 15),
    "motion_blur_31": lambda img, i: motion_blur(img, 31),
    "low_light": lambda img, i: low_light(img, seed=i),
    "downscale_50": lambda img, i: downscale(img, 0.5),
    "downscale_25": lambda img, i: downscale(img, 0.25),
    "jpeg_q15": lambda img, i: jpeg(img, 15),
}
