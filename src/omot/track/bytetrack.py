"""Our ByteTrack implementation: two-stage IoU association over Kalman-predicted tracks.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
(ECCV 2022). Re-implemented for full control over tracker internals (context.md D3); the
hidden-state module (phase 4) extends `Track` while it is LOST.

Detections are (N, 5) float arrays: [x, y, w, h, score] (tlwh). Output rows are
[x, y, w, h, score, track_id].
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from scipy.optimize import linear_sum_assignment

from omot.track.kalman import KalmanFilterCV

logger = logging.getLogger(__name__)

_INFEASIBLE = 1e5


class TrackState(Enum):
    NEW = auto()
    TRACKED = auto()
    LOST = auto()
    REMOVED = auto()


@dataclass(frozen=True)
class TrackerConfig:
    """ByteTrack hyperparameters (defaults follow the official MOT17 settings)."""

    high_thresh: float = 0.6  # score split: >= high -> first association
    low_thresh: float = 0.1  # score >= low -> second association
    new_track_thresh: float = 0.7  # unmatched high det starts a track if score >= this
    match_thresh_first: float = 0.8  # max cost (1 - IoU) accepted, stage 1
    match_thresh_second: float = 0.5  # max cost accepted, stage 2 (low dets)
    match_thresh_unconfirmed: float = 0.7  # max cost accepted for 1-frame-old tracks
    track_buffer: int = 30  # frames a lost track survives
    min_box_area: float = 10.0
    fuse_score: bool = False  # stage 1 cost = 1 - IoU * det_score; measured WORSE with
    # COCO-detector scores on MOT17 val-half (D15) — enable only with calibrated detectors


def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
    """[x, y, w, h] -> [cx, cy, a, h] with a = w/h."""
    x, y, w, h = float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3])
    return np.array([x + w / 2, y + h / 2, w / max(h, 1e-6), h])


def xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
    """[cx, cy, a, h] -> [x, y, w, h]."""
    cx, cy, a, h = float(xyah[0]), float(xyah[1]), float(xyah[2]), float(xyah[3])
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, w, h])


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N, 4+) tlwh box arrays -> (Na, Nb)."""
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]))
    a = boxes_a[:, None, :4].astype(np.float64)
    b = boxes_b[None, :, :4].astype(np.float64)
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2])
    y2 = np.minimum(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter
    return inter / np.clip(union, 1e-9, None)


def linear_assignment(
    cost: np.ndarray, thresh: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment; pairs with cost > thresh are rejected.

    Returns (matches, unmatched_rows, unmatched_cols).
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    gated = np.where(cost > thresh, _INFEASIBLE, cost)
    rows, cols = linear_sum_assignment(gated)
    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(rows, cols, strict=True):
        if gated[r, c] < _INFEASIBLE:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))
    unmatched_rows = [i for i in range(cost.shape[0]) if i not in matched_rows]
    unmatched_cols = [j for j in range(cost.shape[1]) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


class Track:
    """Single tracked object with Kalman state."""

    def __init__(self, tlwh: np.ndarray, score: float, kf: KalmanFilterCV) -> None:
        self._kf = kf
        self.mean, self.covariance = kf.initiate(tlwh_to_xyah(tlwh))
        self.score = score
        self.state = TrackState.NEW
        self.is_activated = False
        self.track_id = 0
        self.start_frame = 0
        self.end_frame = 0  # last frame with a matched detection

    @property
    def tlwh(self) -> np.ndarray:
        return xyah_to_tlwh(self.mean[:4])

    def predict(self) -> None:
        if self.state is not TrackState.TRACKED:
            self.mean[7] = 0.0  # freeze height velocity while not actively tracked
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)

    def activate(self, frame_id: int, track_id: int) -> None:
        self.track_id = track_id
        self.state = TrackState.TRACKED
        self.is_activated = frame_id == 1  # frame-1 tracks activate immediately
        self.start_frame = frame_id
        self.end_frame = frame_id

    def re_activate(self, tlwh: np.ndarray, score: float, frame_id: int) -> None:
        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, tlwh_to_xyah(tlwh)
        )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = score
        self.end_frame = frame_id

    def update(self, tlwh: np.ndarray, score: float, frame_id: int) -> None:
        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, tlwh_to_xyah(tlwh)
        )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = score
        self.end_frame = frame_id

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


class ByteTracker:
    """Two-stage (high/low score) IoU association tracker."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.cfg = config or TrackerConfig()
        self.kf = KalmanFilterCV()
        self.tracked: list[Track] = []
        self.lost: list[Track] = []
        self.frame_id = 0
        self._next_id = 0
        # Kalman-coasted predictions of currently-lost tracks for the frame last updated:
        # (M, 6) [x, y, w, h, 0.0, track_id]. The G2 baseline evaluates these (hidden_eval).
        self.coasting: np.ndarray = np.zeros((0, 6))

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def update(self, detections: np.ndarray, frame_id: int | None = None) -> np.ndarray:
        """Advance one frame. detections: (N, 5) [x, y, w, h, score].

        Returns (M, 6) [x, y, w, h, score, track_id] for activated tracks.
        """
        self.frame_id = self.frame_id + 1 if frame_id is None else frame_id
        cfg = self.cfg
        detections = np.asarray(detections, dtype=np.float64).reshape(-1, 5)

        scores = detections[:, 4]
        high = detections[scores >= cfg.high_thresh]
        low = detections[(scores >= cfg.low_thresh) & (scores < cfg.high_thresh)]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]

        # Stage 1: confirmed + lost tracks vs high-score detections.
        pool = confirmed + self.lost
        for t in pool:
            t.predict()
        sim = iou_matrix(np.array([t.tlwh for t in pool]).reshape(-1, 4), high)
        if cfg.fuse_score and sim.size:
            sim = sim * high[:, 4][None, :]
        cost = 1.0 - sim
        matches, un_track, un_high = linear_assignment(cost, cfg.match_thresh_first)

        activated: list[Track] = []
        refound: list[Track] = []
        for ti, di in matches:
            track, det = pool[ti], high[di]
            if track.state is TrackState.TRACKED:
                track.update(det[:4], float(det[4]), self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det[:4], float(det[4]), self.frame_id)
                refound.append(track)

        # Stage 2: remaining *tracked* tracks vs low-score detections.
        remain_tracked = [pool[i] for i in un_track if pool[i].state is TrackState.TRACKED]
        cost = 1.0 - iou_matrix(np.array([t.tlwh for t in remain_tracked]).reshape(-1, 4), low)
        matches, un_track2, _ = linear_assignment(cost, cfg.match_thresh_second)
        for ti, di in matches:
            track, det = remain_tracked[ti], low[di]
            track.update(det[:4], float(det[4]), self.frame_id)
            activated.append(track)

        lost_now: list[Track] = []
        for i in un_track2:
            remain_tracked[i].mark_lost()
            lost_now.append(remain_tracked[i])

        # Unconfirmed (1-frame-old) tracks vs remaining high detections.
        for t in unconfirmed:
            t.predict()
        remaining_high = high[un_high] if len(un_high) else high[:0]
        cost = 1.0 - iou_matrix(
            np.array([t.tlwh for t in unconfirmed]).reshape(-1, 4), remaining_high
        )
        matches, un_unconf, un_high2 = linear_assignment(cost, cfg.match_thresh_unconfirmed)
        for ti, di in matches:
            det = remaining_high[di]
            unconfirmed[ti].update(det[:4], float(det[4]), self.frame_id)
            activated.append(unconfirmed[ti])
        removed: list[Track] = []
        for i in un_unconf:
            unconfirmed[i].mark_removed()
            removed.append(unconfirmed[i])

        # New tracks from leftover high-score detections.
        new_tracks: list[Track] = []
        for di in un_high2:
            det = remaining_high[di]
            if float(det[4]) >= cfg.new_track_thresh:
                track = Track(det[:4], float(det[4]), self.kf)
                track.activate(self.frame_id, self._new_id())
                new_tracks.append(track)

        # Expire lost tracks beyond the buffer.
        surviving_lost: list[Track] = []
        for t in self.lost:
            if t.state is TrackState.TRACKED:
                continue  # re-activated this frame; rehomed below
            if self.frame_id - t.end_frame > cfg.track_buffer:
                t.mark_removed()
            else:
                surviving_lost.append(t)

        self.tracked = [
            t
            for t in confirmed + unconfirmed + refound + new_tracks
            if t.state is TrackState.TRACKED
        ]
        # De-dup while preserving order (a track can appear via multiple paths).
        seen: set[int] = set()
        self.tracked = [t for t in self.tracked if not (id(t) in seen or seen.add(id(t)))]
        self.lost = surviving_lost + lost_now

        self.coasting = np.array(
            [np.r_[t.tlwh, 0.0, float(t.track_id)] for t in self.lost]
        ).reshape(-1, 6)

        out = [
            np.r_[t.tlwh, t.score, float(t.track_id)]
            for t in self.tracked
            if t.is_activated and t.tlwh[2] * t.tlwh[3] >= cfg.min_box_area
        ]
        return np.array(out).reshape(-1, 6)
