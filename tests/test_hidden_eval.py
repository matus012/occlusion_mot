"""G2 evaluator on synthetic occluder scenes: retention, center/time errors, honest coverage."""
from __future__ import annotations

import numpy as np

from omot.eval.hidden_eval import SegmentResult, aggregate, evaluate_segments
from omot.eval.occlusion import extract_segments
from omot.synth import SynthConfig, generate
from omot.track.bytetrack import ByteTracker, TrackerConfig

PINNED = dict(n_frames=120, n_agents=1, seed=3, p_low_score=0.0, p_false_positive=0.0,
              det_noise_px=0.5, occluder=(300.0, 0.0, 400.0, 720.0),
              x0_range=(0.0, 10.0), vx_range=(5.0, 6.0), vy_range=(0.0, 0.0))


def _track_with_coasting(
    dets: dict[int, np.ndarray], cfg: TrackerConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (active_rows, coasting_rows) as [frame, id, x, y, w, h]."""
    tracker = ByteTracker(cfg)
    active: list[np.ndarray] = []
    coasting: list[np.ndarray] = []
    for f in sorted(dets):
        out = tracker.update(dets[f], frame_id=f)
        for x, y, w, h, _s, tid in out:
            active.append(np.array([f, tid, x, y, w, h]))
        for x, y, w, h, _s, tid in tracker.coasting:
            coasting.append(np.array([f, tid, x, y, w, h]))
    to_arr = lambda rows: np.stack(rows) if rows else np.zeros((0, 6))  # noqa: E731
    return to_arr(active), to_arr(coasting)


def test_baseline_retains_id_and_predicts_reemergence() -> None:
    scene = generate(SynthConfig(**PINNED))
    segments = extract_segments(scene.gt, min_len=3)
    assert len(segments) == 1
    active, coasting = _track_with_coasting(scene.detections, TrackerConfig(track_buffer=60))
    diag = float(np.hypot(1280, 720))
    results = evaluate_segments(segments, active, coasting, diag)
    assert len(results) == 1
    r = results[0]
    assert r.pre_id is not None
    assert r.id_retained
    # constant-velocity coasting through a linear gap -> tight re-emergence prediction
    assert r.center_err is not None and r.center_err < 0.02
    assert r.time_err is not None and r.time_err <= 1
    agg = aggregate(results)
    assert agg["id_retention"] == 1.0
    assert agg["n_segments"] == 1


def test_buffer_expiry_loses_id_and_is_counted_honestly() -> None:
    scene = generate(SynthConfig(**PINNED))
    segments = extract_segments(scene.gt, min_len=3)
    gap = segments[0].gap_length
    # buffer far below the gap -> track removed mid-gap -> retention fails
    active, coasting = _track_with_coasting(
        scene.detections, TrackerConfig(track_buffer=max(2, gap // 4))
    )
    results = evaluate_segments(segments, active, coasting, float(np.hypot(1280, 720)))
    r = results[0]
    assert r.pre_id is not None
    assert not r.id_retained
    assert r.center_err is None  # no coasting alive at re-emergence; no silent fallback
    agg = aggregate(results)
    assert agg["id_retention"] == 0.0
    assert agg["center_err_coverage"] == 0.0


def test_aggregate_handles_empty_and_partial() -> None:
    assert aggregate([])["n_segments"] == 0
    partial = [
        SegmentResult(1, 1, 1, True, 0.01, 0.0),
        SegmentResult(2, None, None, False, None, None),
    ]
    agg = aggregate(partial)
    assert agg["id_retention"] == 0.5
    assert agg["pre_match_rate"] == 0.5
    assert agg["center_err_coverage"] == 0.5
    assert agg["reemergence_center_err_med"] == 0.01
    # D26 split-gate metrics: only the first segment is in association scope
    assert agg["id_retention_assoc"] == 1.0
    assert agg["oracle_ceiling"] == 0.5
    assert agg["cov_prematched"] == 1.0


def test_aggregate_assoc_scope_excludes_detector_failures() -> None:
    results = [
        SegmentResult(1, 1, 1, True, 0.01, 0.0),     # retained
        SegmentResult(2, 2, 9, False, 0.02, 4.0),    # assoc-scope failure (swapped)
        SegmentResult(3, 3, None, False, 0.03, None),  # no detection at re-emergence
        SegmentResult(4, None, None, False, None, None),  # never tracked pre-gap
    ]
    agg = aggregate(results)
    assert agg["id_retention"] == 0.25          # end-to-end (G2b view)
    assert agg["id_retention_assoc"] == 0.5     # module view (G2a): 1 of 2 in scope
    assert agg["n_assoc_scope"] == 2
    assert agg["oracle_ceiling"] == 0.5
    assert agg["pre_match_rate"] == 0.75
    assert agg["cov_prematched"] == 1.0         # all 3 pre-matched got predictions
