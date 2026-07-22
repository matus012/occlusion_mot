"""Occlusion-segment extraction: hand-built GT cases + synthetic occluder scene."""
from __future__ import annotations

import numpy as np

from omot.eval.occlusion import extract_segments, segment_stats
from omot.io.mot_format import COL
from omot.synth import SynthConfig, generate


def _gt_row(frame: int, tid: int, x: float, vis: float) -> list[float]:
    return [frame, tid, x, 100.0, 50.0, 120.0, 1.0, 1.0, vis]


def test_basic_segment() -> None:
    rows = []
    for f in range(1, 11):
        rows.append(_gt_row(f, 1, 10.0 * f, 1.0))
    for f in range(11, 19):  # 8 occluded frames
        rows.append(_gt_row(f, 1, 10.0 * f, 0.1))
    for f in range(19, 25):
        rows.append(_gt_row(f, 1, 10.0 * f, 0.9))
    segs = extract_segments(np.array(rows), min_len=5)
    assert len(segs) == 1
    s = segs[0]
    assert s.track_id == 1
    assert s.last_visible_frame == 10
    assert s.reemergence_frame == 19
    assert s.gap_length == 8
    assert s.last_visible_box[0] == 100.0
    assert s.reemergence_box[0] == 190.0


def test_annotation_gap_counts_as_occlusion() -> None:
    rows = [_gt_row(f, 2, 5.0 * f, 1.0) for f in range(1, 6)]
    rows += [_gt_row(f, 2, 5.0 * f, 1.0) for f in range(20, 25)]  # 14-frame hole
    segs = extract_segments(np.array(rows), min_len=5)
    assert len(segs) == 1
    assert segs[0].gap_length == 14


def test_short_gap_ignored() -> None:
    rows = [_gt_row(f, 1, 10.0 * f, 1.0 if f not in (5, 6) else 0.0) for f in range(1, 12)]
    assert extract_segments(np.array(rows), min_len=5) == []


def test_partial_visibility_contaminates() -> None:
    rows = [_gt_row(f, 1, 10.0 * f, 1.0) for f in range(1, 6)]
    rows += [_gt_row(f, 1, 10.0 * f, 0.1) for f in range(6, 12)]
    rows[8] = _gt_row(9, 1, 90.0, 0.4)  # vis in [vis_lo, vis_hi) mid-gap
    rows += [_gt_row(f, 1, 10.0 * f, 1.0) for f in range(12, 15)]
    assert extract_segments(np.array(rows), min_len=5) == []


def test_multiple_ids_independent() -> None:
    rows = []
    for f in range(1, 30):
        rows.append(_gt_row(f, 1, 10.0 * f, 1.0))  # id 1 never occluded
        vis = 0.0 if 10 <= f < 20 else 1.0
        rows.append(_gt_row(f, 2, 400.0 + 5.0 * f, vis))
    segs = extract_segments(np.array(rows), min_len=5)
    assert [s.track_id for s in segs] == [2]


def test_synth_occluder_scene_yields_segments() -> None:
    scene = generate(SynthConfig(n_frames=120, n_agents=1, seed=3, p_false_positive=0.0,
                                 occluder=(300.0, 0.0, 400.0, 720.0),
                                 x0_range=(0.0, 10.0), vx_range=(5.0, 6.0),
                                 vy_range=(0.0, 0.0)))
    segs = extract_segments(scene.gt, min_len=3)
    assert len(segs) == 1
    s = segs[0]
    occluded_frames = scene.gt[scene.gt[:, COL.VIS] == 0.0][:, COL.FRAME].astype(int)
    assert s.last_visible_frame == occluded_frames.min() - 1
    assert s.reemergence_frame == occluded_frames.max() + 1
    stats = segment_stats(segs, diagonal=float(np.hypot(1280, 720)))
    assert stats["n_segments"] == 1.0
    assert stats["gap_median"] == s.gap_length
    assert 0.0 < stats["displacement_median_diag"] < 1.0
