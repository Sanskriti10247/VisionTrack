"""Constant-velocity Kalman filter in (cx, cy, aspect, height) space.

This is the filter used by SORT/DeepSORT and the official ByteTrack code:
state = [cx, cy, a, h, vcx, vcy, va, vh]; process and measurement noise
scale with the box height so that large and small boxes behave alike.
"""

from __future__ import annotations

import numpy as np

_STD_POSITION = 1.0 / 20
_STD_VELOCITY = 1.0 / 160


class KalmanFilterXYAH:
    ndim = 4

    def __init__(self) -> None:
        dt = 1.0
        self._motion_mat = np.eye(2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = dt
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros_like(measurement)]
        h = measurement[3]
        std = [
            2 * _STD_POSITION * h,
            2 * _STD_POSITION * h,
            1e-2,
            2 * _STD_POSITION * h,
            10 * _STD_VELOCITY * h,
            10 * _STD_VELOCITY * h,
            1e-5,
            10 * _STD_VELOCITY * h,
        ]
        return mean, np.diag(np.square(std))

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised predict step for N tracks: mean (N, 8), covariance (N, 8, 8)."""
        h = mean[:, 3]
        std_pos = np.stack([_STD_POSITION * h, _STD_POSITION * h, np.full_like(h, 1e-2), _STD_POSITION * h])
        std_vel = np.stack([_STD_VELOCITY * h, _STD_VELOCITY * h, np.full_like(h, 1e-5), _STD_VELOCITY * h])
        sqr = np.square(np.r_[std_pos, std_vel]).T  # (N, 8)
        motion_cov = np.zeros((len(mean), 8, 8))
        idx = np.arange(8)
        motion_cov[:, idx, idx] = sqr
        mean = mean @ self._motion_mat.T
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [_STD_POSITION * h, _STD_POSITION * h, 1e-1, _STD_POSITION * h]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        # Solve via Cholesky instead of an explicit inverse for numerical stability.
        chol = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(chol.T, np.linalg.solve(chol, (covariance @ self._update_mat.T).T)).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = xyah[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)
