"""Constant-velocity Kalman filter for bounding-box tracking (DeepSORT/ByteTrack style).

State: [cx, cy, a, h, vcx, vcy, va, vh] where a = w/h aspect ratio.
Measurement: [cx, cy, a, h]. Noise scales with box height, per DeepSORT.
"""
from __future__ import annotations

import numpy as np


class KalmanFilterCV:
    """8-dim constant-velocity KF over box center/aspect/height."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat: np.ndarray = np.eye(2 * ndim)
        self._motion_mat[:ndim, ndim:] = dt * np.eye(ndim)
        self._update_mat: np.ndarray = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create state mean/covariance from an initial measurement [cx, cy, a, h]."""
        mean = np.r_[measurement, np.zeros_like(measurement)]
        h = float(measurement[3])
        std = np.array(
            [
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                1e-2,
                2 * self._std_weight_position * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                1e-5,
                10 * self._std_weight_velocity * h,
            ]
        )
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One-step time update."""
        h = float(mean[3])
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-2,
                self._std_weight_position * h,
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                1e-5,
                self._std_weight_velocity * h,
            ]
        )
        motion_cov = np.diag(np.square(std))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state to measurement space."""
        h = float(mean[3])
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ]
        )
        innovation_cov = np.diag(np.square(std))
        mean_proj = self._update_mat @ mean
        cov_proj = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return mean_proj, cov_proj

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Measurement update with [cx, cy, a, h]."""
        proj_mean, proj_cov = self.project(mean, covariance)
        chol = np.linalg.cholesky(proj_cov)
        kalman_gain = np.linalg.solve(
            chol.T, np.linalg.solve(chol, (covariance @ self._update_mat.T).T)
        ).T
        innovation = measurement - proj_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov
