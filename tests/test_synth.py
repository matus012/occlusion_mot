"""Synthetic scene generator invariants."""
from __future__ import annotations

import numpy as np

from omot.io.mot_format import COL
from omot.synth import SynthConfig, generate


def test_deterministic_given_seed() -> None:
    a = generate(SynthConfig(n_frames=30, seed=42))
    b = generate(SynthConfig(n_frames=30, seed=42))
    np.testing.assert_array_equal(a.gt, b.gt)
    assert a.detections.keys() == b.detections.keys()
    for f in a.detections:
        np.testing.assert_array_equal(a.detections[f], b.detections[f])


def test_occluder_suppresses_detections_and_marks_visibility() -> None:
    cfg = SynthConfig(n_frames=120, n_agents=1, seed=3, p_false_positive=0.0,
                      occluder=(300.0, 0.0, 400.0, 720.0),
                      x0_range=(0.0, 10.0), vx_range=(5.0, 6.0), vy_range=(0.0, 0.0))
    scene = generate(cfg)
    occluded_rows = scene.gt[scene.gt[:, COL.VIS] == 0.0]
    assert len(occluded_rows) >= 5, "occluder must create a real occlusion segment"
    for f in occluded_rows[:, COL.FRAME].astype(int):
        assert len(scene.detections[f]) == 0
    visible_rows = scene.gt[scene.gt[:, COL.VIS] == 1.0]
    assert len(visible_rows) > len(occluded_rows)


def test_gt_schema() -> None:
    scene = generate(SynthConfig(n_frames=20, n_agents=3, seed=0))
    gt = scene.gt
    assert gt.shape[1] == 9
    assert gt[:, COL.FRAME].min() >= 1
    assert set(np.unique(gt[:, COL.CLS])) == {1.0}
    assert gt[:, COL.CONF].min() == 1.0
    # every frame has detections dict entry (possibly empty)
    assert set(scene.detections.keys()) == set(range(1, 21))
