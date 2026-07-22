"""Occlusion-segment extraction from GT visibility (context.md D7 revised by D14, gates G2).

Segment definition (D14, calibrated on real MOT17 GT where visibility decays gradually):
for one GT identity, a run of frames between two solidly-visible annotations
(visibility >= vis_hi) in which every annotated frame stays below vis_hi AND visibility
dips below vis_lo at least once — frames with no annotation at all count as occluded
(a fully-unannotated gap satisfies the dip). The gap must span >= min_len frames.
Runs whose visibility never leaves [vis_lo, vis_hi) are partial occlusions, not segments.
By default only pedestrian-class rows with the GT consider flag set are used (MOT GT
carries distractor/static classes and consider=0 rows that must not create segments).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from omot.io.mot_format import COL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcclusionSegment:
    track_id: int
    last_visible_frame: int
    reemergence_frame: int
    last_visible_box: np.ndarray  # tlwh
    reemergence_box: np.ndarray  # tlwh

    @property
    def gap_length(self) -> int:
        """Number of frames the identity is occluded."""
        return self.reemergence_frame - self.last_visible_frame - 1


def extract_segments(
    gt: np.ndarray,
    vis_lo: float = 0.25,
    vis_hi: float = 0.5,
    min_len: int = 5,
    pedestrian_only: bool = True,
) -> list[OcclusionSegment]:
    """Extract occlusion segments from (N, 9) GT rows for all identities."""
    if gt.size == 0:
        return []
    assert vis_lo <= vis_hi, "vis_lo must not exceed vis_hi"
    if pedestrian_only:
        gt = gt[(gt[:, COL.CLS] == 1.0) & (gt[:, COL.CONF] == 1.0)]
    segments: list[OcclusionSegment] = []
    for tid in np.unique(gt[:, COL.ID]).astype(np.int64):
        rows = gt[gt[:, COL.ID] == tid]
        rows = rows[np.argsort(rows[:, COL.FRAME])]
        frames = rows[:, COL.FRAME].astype(np.int64)
        vis = rows[:, COL.VIS]
        visible_idx = np.flatnonzero(vis >= vis_hi)
        for a, b in zip(visible_idx[:-1], visible_idx[1:], strict=True):
            f_a, f_b = int(frames[a]), int(frames[b])
            gap = f_b - f_a - 1
            if gap < min_len:
                continue
            between = vis[a + 1 : b]
            if between.size and between.min() >= vis_lo:
                continue  # never dips below vis_lo -> partial occlusion, not a segment
            segments.append(
                OcclusionSegment(
                    track_id=int(tid),
                    last_visible_frame=f_a,
                    reemergence_frame=f_b,
                    last_visible_box=rows[a, COL.X : COL.H + 1].copy(),
                    reemergence_box=rows[b, COL.X : COL.H + 1].copy(),
                )
            )
    logger.info("extracted %d occlusion segments (min_len=%d)", len(segments), min_len)
    return segments


def segment_stats(segments: list[OcclusionSegment], diagonal: float) -> dict[str, float]:
    """Aggregate stats used to freeze G2 thresholds: gap lengths and displacement
    (center distance between disappearance and re-emergence, in diagonal units)."""
    if not segments:
        return {"n_segments": 0.0}
    gaps = np.array([s.gap_length for s in segments], dtype=np.float64)
    disp = np.array(
        [
            float(
                np.hypot(
                    (s.reemergence_box[0] + s.reemergence_box[2] / 2)
                    - (s.last_visible_box[0] + s.last_visible_box[2] / 2),
                    (s.reemergence_box[1] + s.reemergence_box[3] / 2)
                    - (s.last_visible_box[1] + s.last_visible_box[3] / 2),
                )
            )
            for s in segments
        ]
    )
    return {
        "n_segments": float(len(segments)),
        "gap_median": float(np.median(gaps)),
        "gap_p90": float(np.percentile(gaps, 90)),
        "displacement_median_diag": float(np.median(disp) / diagonal),
        "displacement_p90_diag": float(np.percentile(disp, 90) / diagonal),
    }
