"""G2 evaluator: hidden-state prediction quality on occlusion segments (gates.yaml G2).

Consumes, all in the SAME frame space as the GT the segments came from:
  - active rows:   (N, >=6) [frame, id, x, y, w, h, ...] tracker output
  - coasting rows: (M, >=6) [frame, id, x, y, w, h, ...] predicted boxes of LOST tracks
    (our ByteTracker exposes these per frame; the hidden-state module will emit its own)

Per segment (D14):
  pre_id        tracker id matched (IoU >= match_iou) to the GT box at last_visible_frame
  post_id       tracker id matched to the GT box at reemergence_frame
  id_retained   pre_id exists and pre_id == post_id
  center_err    | predicted center of pre_id at reemergence_frame - GT center | / diagonal,
                prediction from coasting rows; if the track re-associated exactly at the
                re-emergence frame, its active box is used (detector-informed, slight
                optimism — noted in reports)
  time_err      | first frame > last_visible_frame where pre_id is active again
                - reemergence_frame |   (only defined when pre_id reappears at all)

Aggregates match the gates.yaml G2 schema.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from omot.eval.occlusion import OcclusionSegment
from omot.track.bytetrack import iou_matrix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentResult:
    track_id: int
    pre_id: int | None
    post_id: int | None
    id_retained: bool
    center_err: float | None  # fraction of diagonal
    time_err: float | None  # frames


def _rows_at(rows: np.ndarray, frame: int) -> np.ndarray:
    return rows[rows[:, 0] == frame]


def _match_id(rows: np.ndarray, frame: int, gt_box: np.ndarray, match_iou: float) -> int | None:
    cand = _rows_at(rows, frame)
    if not len(cand):
        return None
    ious = iou_matrix(gt_box.reshape(1, 4), cand[:, 2:6]).ravel()
    best = int(np.argmax(ious))
    return int(cand[best, 1]) if ious[best] >= match_iou else None


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])


def evaluate_segments(
    segments: list[OcclusionSegment],
    active: np.ndarray,
    coasting: np.ndarray,
    diagonal: float,
    match_iou: float = 0.5,
) -> list[SegmentResult]:
    """Evaluate one sequence's segments against tracker outputs."""
    active = np.asarray(active, dtype=np.float64).reshape(
        -1, active.shape[-1] if active.size else 6
    )
    coasting = np.asarray(coasting, dtype=np.float64).reshape(
        -1, coasting.shape[-1] if coasting.size else 6
    )
    results: list[SegmentResult] = []
    for seg in segments:
        pre_id = _match_id(active, seg.last_visible_frame, seg.last_visible_box, match_iou)
        post_id = _match_id(active, seg.reemergence_frame, seg.reemergence_box, match_iou)
        retained = pre_id is not None and pre_id == post_id

        center_err: float | None = None
        if pre_id is not None:
            pred_box: np.ndarray | None = None
            coast = _rows_at(coasting, seg.reemergence_frame)
            hit = coast[coast[:, 1] == pre_id]
            if len(hit):
                pred_box = hit[0, 2:6]
            else:
                act = _rows_at(active, seg.reemergence_frame)
                hit = act[act[:, 1] == pre_id]
                if len(hit):
                    pred_box = hit[0, 2:6]
            if pred_box is not None:
                gt_c = _center(seg.reemergence_box)
                center_err = float(np.linalg.norm(_center(pred_box) - gt_c) / diagonal)

        time_err: float | None = None
        if pre_id is not None:
            later = active[(active[:, 1] == pre_id) & (active[:, 0] > seg.last_visible_frame)]
            if len(later):
                first_back = int(later[:, 0].min())
                time_err = float(abs(first_back - seg.reemergence_frame))

        results.append(
            SegmentResult(
                track_id=seg.track_id,
                pre_id=pre_id,
                post_id=post_id,
                id_retained=retained,
                center_err=center_err,
                time_err=time_err,
            )
        )
    return results


def aggregate(results: list[SegmentResult]) -> dict[str, float | int]:
    """Aggregate to the gates.yaml G2 schema (incl. D26 split-gate metrics).

    - id_retention          end-to-end over ALL segments (G2b; detector-capped)
    - id_retention_assoc    association scope: segments where the tracker had the target
                            pre-gap AND a detection matched GT at re-emergence (G2a)
    - cov_prematched        prediction coverage conditional on pre-matched segments
                            (D26 definitional repair: unconditional coverage is capped
                            by pre_match_rate)
    Taxonomy fields (pre_match_rate, oracle_ceiling) ship with every aggregate so the
    D26 val-time guard is automatic.
    """
    n = len(results)
    center = [r.center_err for r in results if r.center_err is not None]
    time_e = [r.time_err for r in results if r.time_err is not None]
    n_pre = sum(r.pre_id is not None for r in results)
    assoc = [r for r in results if r.pre_id is not None and r.post_id is not None]
    n_retained = sum(r.id_retained for r in results)
    out: dict[str, float | int] = {
        "n_segments": n,
        "id_retention": n_retained / n if n else 0.0,
        "id_retention_assoc": (
            sum(r.id_retained for r in assoc) / len(assoc) if assoc else 0.0
        ),
        "n_assoc_scope": len(assoc),
        "pre_match_rate": n_pre / n if n else 0.0,
        "oracle_ceiling": len(assoc) / n if n else 0.0,  # max achievable id_retention
        "center_err_coverage": len(center) / n if n else 0.0,
        "cov_prematched": len(center) / n_pre if n_pre else 0.0,
        "time_err_coverage": len(time_e) / n if n else 0.0,
    }
    out["reemergence_center_err_med"] = float(np.median(center)) if center else float("nan")
    out["reemergence_time_err_med"] = float(np.median(time_e)) if time_e else float("nan")
    logger.info("hidden_eval aggregate: %s", out)
    return out
