"""Mock feeder (P2): determinism, projection sanity, visibility GT, MOT-dir round-trip."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from omot.data.mot import load_sequence
from omot.eval.occlusion import extract_segments
from omot.io.mot_format import COL
from omot.sim.feeder import MockBackend, _Cam, _project_box, render_scenario
from omot.sim.scenario import (
    OcclusionScenario,
    WalkerSpec,
    generate_scenarios,
    make_scenario,
)


def test_scenario_generation_deterministic() -> None:
    a = generate_scenarios(6, seed=1)
    b = generate_scenarios(6, seed=1)
    assert [s.to_json() for s in a] == [s.to_json() for s in b]
    assert len({s.scenario_id for s in a}) == 6
    assert {s.template for s in a} == {"behind_static", "crossing_paths", "crowd_merge"}


def test_projection_geometry() -> None:
    s = make_scenario("behind_static", seed=0)
    cam = _Cam.from_scenario(s)
    near = _project_box(cam, 5.0, 0.0, 1.75)
    far = _project_box(cam, 20.0, 0.0, 1.75)
    assert near is not None and far is not None
    assert near[3] > far[3], "nearer person must project taller"
    behind = _project_box(cam, -1.0, 0.0, 1.75)
    assert behind is None
    left = _project_box(cam, 10.0, -3.0, 1.75)
    right = _project_box(cam, 10.0, 3.0, 1.75)
    assert left is not None and right is not None
    assert left[0] < right[0], "world -y must project left of +y"


def _occluded_walker_scenario() -> OcclusionScenario:
    # One walker crossing directly behind a wide occluder; camera default.
    return OcclusionScenario(
        scenario_id="unit_occl", template="crossing_paths", seed=7,
        duration_s=12.0, fps=10,
        walkers=[WalkerSpec(walker_id=1, waypoints=[(16.0, -7.0), (16.0, 7.0)], speed=1.2)],
        occluders=[(14.5, 0.0, 1.5, 0.5, 2.4)],
        cam_pos=(0.0, 0.0, 2.5), cam_yaw_deg=0.0,
    )


def test_visibility_drops_behind_occluder(tmp_path: Path) -> None:
    seq_dir = render_scenario(_occluded_walker_scenario(), tmp_path, MockBackend())
    seq = load_sequence(seq_dir)
    assert seq.gt is not None and len(seq.gt)
    vis = seq.gt[:, COL.VIS]
    assert vis.min() < 0.25, "walker must be deeply occluded at some point"
    assert vis.max() > 0.9, "walker must be fully visible away from the occluder"
    # visibility profile: high -> low -> high through the crossing
    frames = seq.gt[:, COL.FRAME]
    lowest = frames[int(np.argmin(vis))]
    assert frames.min() < lowest < frames.max()


def test_rendered_dir_feeds_existing_pipeline(tmp_path: Path) -> None:
    seq_dir = render_scenario(_occluded_walker_scenario(), tmp_path, MockBackend())
    seq = load_sequence(seq_dir)
    assert seq.seq_length == 120
    assert seq.frame_path(1).exists() and seq.frame_path(seq.seq_length).exists()
    segs = extract_segments(seq.gt, min_len=3)
    assert len(segs) >= 1, "D14 extractor must find the occlusion segment in sim GT"
    s = segs[0]
    assert s.gap_length >= 3
    assert s.reemergence_box[3] > 0
