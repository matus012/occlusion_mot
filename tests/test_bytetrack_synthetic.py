"""ByteTracker behavior on synthetic scenes: stable IDs, occlusion recovery, low-score stage."""
from __future__ import annotations

import numpy as np

from omot.synth import SynthConfig, generate
from omot.track.bytetrack import ByteTracker, TrackerConfig


def _run_tracker(
    scene_frames: dict[int, np.ndarray], cfg: TrackerConfig | None = None
) -> dict[int, np.ndarray]:
    tracker = ByteTracker(cfg)
    out: dict[int, np.ndarray] = {}
    for f in sorted(scene_frames):
        out[f] = tracker.update(scene_frames[f], frame_id=f)
    return out


def _ids_over_time(outputs: dict[int, np.ndarray]) -> dict[int, set[float]]:
    return {f: set(o[:, 5]) for f, o in outputs.items() if len(o)}


def test_stable_ids_clean_scene() -> None:
    scene = generate(SynthConfig(n_frames=60, n_agents=4, seed=1, p_low_score=0.0,
                                 p_false_positive=0.0, det_noise_px=1.0))
    outputs = _run_tracker(scene.detections)
    # After warmup, the same 4 ids persist every frame.
    ids_per_frame = _ids_over_time(outputs)
    stable = [ids for f, ids in ids_per_frame.items() if f > 5]
    assert all(len(ids) == 4 for ids in stable)
    assert set.union(*stable) == stable[0]  # no id churn


def test_id_survives_occlusion_gap() -> None:
    # Pinned spawn/velocity guarantee an occluder crossing with a ~17-20 frame gap,
    # comfortably below the 60-frame buffer.
    scene = generate(SynthConfig(n_frames=120, n_agents=1, seed=3, p_low_score=0.0,
                                 p_false_positive=0.0, det_noise_px=0.5,
                                 occluder=(300.0, 0.0, 400.0, 720.0),
                                 x0_range=(0.0, 10.0), vx_range=(5.0, 6.0),
                                 vy_range=(0.0, 0.0)))
    frames_with_det = [f for f, d in scene.detections.items() if len(d)]
    gap = [f for f, d in scene.detections.items() if not len(d)]
    assert gap, "scene must actually contain an occlusion gap"
    outputs = _run_tracker(scene.detections, TrackerConfig(track_buffer=60))
    ids_before = _ids_over_time({f: outputs[f] for f in frames_with_det if f < min(gap)})
    ids_after = _ids_over_time({f: outputs[f] for f in frames_with_det if f > max(gap)})
    assert ids_before and ids_after
    # ByteTrack keeps lost tracks alive within the buffer -> same id on re-emergence
    assert set.union(*ids_before.values()) == set.union(*ids_after.values())


def test_low_score_detections_rescued_in_second_stage() -> None:
    # One agent whose detections all get low scores after frame 20 — first-stage-only
    # trackers would drop it; ByteTrack's second stage must keep it alive.
    rng = np.random.default_rng(0)
    frames: dict[int, np.ndarray] = {}
    for f in range(1, 41):
        x = 100.0 + 5.0 * (f - 1)
        score = 0.9 if f <= 20 else 0.3
        jitter = rng.normal(0, 0.5, 2)
        frames[f] = np.array([[x + jitter[0], 200 + jitter[1], 50.0, 100.0, score]])
    outputs = _run_tracker(frames)
    late = [outputs[f] for f in range(25, 41)]
    assert all(len(o) == 1 for o in late), "low-score track was dropped"
    all_ids = {o[0, 5] for o in late}
    assert len(all_ids) == 1, "id switched on score drop"


def test_no_tracks_from_pure_noise() -> None:
    rng = np.random.default_rng(7)
    frames = {
        f: np.hstack([rng.uniform(0, 500, (3, 4)), rng.uniform(0.1, 0.4, (3, 1))])
        for f in range(1, 20)
    }
    outputs = _run_tracker(frames)
    assert all(len(o) == 0 for o in outputs.values()), "low-score noise must not spawn tracks"


def test_output_schema() -> None:
    scene = generate(SynthConfig(n_frames=10, n_agents=2, seed=5))
    outputs = _run_tracker(scene.detections)
    for out in outputs.values():
        assert out.ndim == 2 and out.shape[1] == 6
        if len(out):
            assert (out[:, 2] > 0).all() and (out[:, 3] > 0).all()  # w, h positive
            assert (out[:, 5] >= 1).all()  # ids start at 1
