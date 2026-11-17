"""Kalman filter sanity: tracks constant-velocity motion, uncertainty grows without updates."""
from __future__ import annotations

import numpy as np

from omot.track.bytetrack import tlwh_to_xyah, xyah_to_tlwh
from omot.track.kalman import KalmanFilterCV


def test_tracks_constant_velocity() -> None:
    kf = KalmanFilterCV()
    pos = np.array([100.0, 200.0])
    vel = np.array([5.0, -2.0])
    box = np.array([pos[0], pos[1], 50.0, 100.0])
    mean, cov = kf.initiate(tlwh_to_xyah(box))
    for t in range(1, 31):
        mean, cov = kf.predict(mean, cov)
        true_box = np.array([pos[0] + vel[0] * t, pos[1] + vel[1] * t, 50.0, 100.0])
        mean, cov = kf.update(mean, cov, tlwh_to_xyah(true_box))
    predicted = xyah_to_tlwh(mean[:4])
    final_true = np.array([pos[0] + vel[0] * 30, pos[1] + vel[1] * 30, 50.0, 100.0])
    assert np.abs(predicted - final_true).max() < 2.0
    # velocity estimate converged towards the true center velocity
    assert np.abs(mean[4] - vel[0]) < 0.5
    assert np.abs(mean[5] - vel[1]) < 0.5


def test_uncertainty_grows_during_coasting() -> None:
    kf = KalmanFilterCV()
    mean, cov = kf.initiate(tlwh_to_xyah(np.array([100.0, 100.0, 50.0, 100.0])))
    _, proj0 = kf.project(mean, cov)
    for _ in range(10):
        mean, cov = kf.predict(mean, cov)
    _, proj10 = kf.project(mean, cov)
    assert np.trace(proj10) > np.trace(proj0)


def test_update_shrinks_uncertainty() -> None:
    kf = KalmanFilterCV()
    mean, cov = kf.initiate(tlwh_to_xyah(np.array([100.0, 100.0, 50.0, 100.0])))
    for _ in range(5):
        mean, cov = kf.predict(mean, cov)
    trace_before = np.trace(cov[:2, :2])
    mean, cov = kf.update(mean, cov, tlwh_to_xyah(np.array([110.0, 100.0, 50.0, 100.0])))
    assert np.trace(cov[:2, :2]) < trace_before
