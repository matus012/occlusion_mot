"""Occlusion-aware tracker: ByteTracker + hidden-state handling (context.md phase 4 design).

Four extensions over the baseline, all geometric (v1, no learned parts):
  1. loss-time occlusion classification (overlap with an active track = inter-object occlusion)
  2. extended buffer for occluded tracks (gap p90 on MOT17 is 77 frames vs 30 baseline buffer)
  3. velocity damping while lost — interpolates CV coasting toward stay-put, matching the
     measured displacement distribution of occluded pedestrians (median 0.02 diag)
  4. recovery association: unclaimed high dets vs occluded lost tracks on box-scale-normalized
     center distance — catches re-emergences where IoU is zero after coasting drift

Baseline behavior is bit-identical when this subclass is not used (hook no-ops; verified).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from omot.track.bytetrack import (
    ByteTracker,
    Track,
    TrackerConfig,
    TrackState,
    iou_matrix,
    linear_assignment,
)

logger = logging.getLogger(__name__)


def _default_base() -> TrackerConfig:
    # Frozen operating point of the baseline (D15): activation 0.25.
    return TrackerConfig(high_thresh=0.25, new_track_thresh=0.35)


@dataclass(frozen=True)
class HiddenConfig:
    base: TrackerConfig = field(default_factory=_default_base)
    occl_buffer: int = 90  # frames an occlusion-classified lost track survives
    vel_damping: float = 0.95  # per-frame velocity decay while lost (1.0 = pure CV)
    occl_overlap_thresh: float = 0.25  # IoU with an active track to classify loss as occlusion
    recover_gate: float = 1.5  # max center distance in units of predicted box scale
    recover_min_score: float = 0.35  # min det score eligible for recovery


class OcclusionAwareTracker(ByteTracker):
    """ByteTracker with occlusion-aware lost-state handling and recovery association."""

    def __init__(self, config: HiddenConfig | None = None) -> None:
        self.hcfg = config or HiddenConfig()
        super().__init__(self.hcfg.base)

    def _predict_pool(self, pool: list[Track]) -> None:
        for t in pool:
            if t.state is TrackState.LOST:
                t.mean[4:6] *= self.hcfg.vel_damping
            t.predict()

    def _classify_lost(self, lost_now: list[Track], active: list[Track]) -> None:
        if not lost_now:
            return
        if not active:
            for t in lost_now:
                t.occluded = False
            return
        act_boxes = np.array([t.tlwh for t in active]).reshape(-1, 4)
        for t in lost_now:
            ious = iou_matrix(t.tlwh.reshape(1, 4), act_boxes)
            t.occluded = bool(ious.max() >= self.hcfg.occl_overlap_thresh)

    def _buffer_for(self, track: Track) -> int:
        return self.hcfg.occl_buffer if track.occluded else self.cfg.track_buffer

    def _recover(
        self, remaining_high: np.ndarray, un_high2: list[int]
    ) -> tuple[list[int], list[Track]]:
        # Lost tracks still LOST here were not re-associated in stage 1 this frame.
        candidates = [t for t in self.lost if t.state is TrackState.LOST and t.occluded]
        det_idx = [
            di for di in un_high2
            if float(remaining_high[di][4]) >= self.hcfg.recover_min_score
        ]
        if not candidates or not det_idx:
            return un_high2, []

        track_boxes = np.array([t.tlwh for t in candidates]).reshape(-1, 4)
        det_boxes = remaining_high[det_idx][:, :4]
        t_centers = track_boxes[:, :2] + track_boxes[:, 2:4] / 2
        d_centers = det_boxes[:, :2] + det_boxes[:, 2:4] / 2
        scale = np.sqrt(np.clip(track_boxes[:, 2] * track_boxes[:, 3], 1.0, None))[:, None]
        cost = np.linalg.norm(t_centers[:, None, :] - d_centers[None, :, :], axis=2) / scale

        matches, _, un_det = linear_assignment(cost, self.hcfg.recover_gate)
        recovered: list[Track] = []
        for ti, dj in matches:
            det = remaining_high[det_idx[dj]]
            candidates[ti].re_activate(det[:4], float(det[4]), self.frame_id)
            recovered.append(candidates[ti])
        if recovered:
            logger.debug(
                "frame %d: recovered %d occluded tracks", self.frame_id, len(recovered)
            )
        unclaimed = {det_idx[k] for k in un_det}
        leftover = [di for di in un_high2 if di not in set(det_idx) or di in unclaimed]
        return leftover, recovered
