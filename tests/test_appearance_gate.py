"""Appearance gate (phase 5a): identity-theft veto and correct-recovery acceptance,
driven by synthetic orthogonal embeddings."""
from __future__ import annotations

import numpy as np

from omot.hidden.occlusion_tracker import HiddenConfig, OcclusionAwareTracker
from omot.track.bytetrack import ByteTracker, TrackerConfig

BASE = TrackerConfig(high_thresh=0.25, new_track_thresh=0.35)
W, H = 60.0, 120.0

E_AGENT = np.zeros(8, dtype=np.float32)
E_AGENT[0] = 1.0
E_IMPOSTOR = np.zeros(8, dtype=np.float32)
E_IMPOSTOR[1] = 1.0  # orthogonal -> cos distance 1.0


def _det(x: float, y: float, score: float = 0.9) -> np.ndarray:
    return np.array([x, y, W, H, score])


def _theft_scenario(tracker: ByteTracker) -> tuple[int | None, dict[int, list[int]]]:
    """Agent (emb E_AGENT) walks right, disappears at f20 behind a static partner.
    From f26, an IMPOSTOR with orthogonal appearance walks exactly where the agent's
    coasted prediction is. The agent itself re-emerges on-trajectory at f46 offset +80px
    (forcing recovery rather than plain IoU). Returns (agent id, {id: frames seen})."""
    partner_x = 100.0 + 5.0 * 19 + W * 0.5
    agent_id: int | None = None
    seen: dict[int, list[int]] = {}
    for f in range(1, 61):
        dets = [np.array([partner_x, 400.0, W, H, 0.9])]  # partner on separate row
        embs = [np.zeros(8, dtype=np.float32)]
        if f <= 20:
            dets.append(_det(100.0 + 5.0 * (f - 1), 200.0))
            embs.append(E_AGENT)
        if 26 <= f <= 60:  # impostor rides the coasted prediction
            dets.append(_det(100.0 + 5.0 * (f - 1), 200.0))
            embs.append(E_IMPOSTOR)
        if f >= 46:  # true agent re-emerges, offset from prediction
            dets.append(_det(100.0 + 5.0 * (f - 1) + 80.0, 220.0))
            embs.append(E_AGENT)
        out = tracker.update(np.stack(dets), frame_id=f, embeddings=np.stack(embs))
        for _x, y, _w, _h, _s, tid in out:
            if abs(y - 400.0) > 50:  # ignore the partner row
                seen.setdefault(int(tid), []).append(f)
        if f == 20 and agent_id is None and seen:
            agent_id = max(seen, key=lambda k: len(seen[k]))
    return agent_id, seen


def test_gate_blocks_impostor_and_recovers_true_agent() -> None:
    cfg = HiddenConfig(base=BASE, vel_damping=1.0, occl_overlap_thresh=0.0,
                       recover_gate=3.0, app_gate_lost=0.35, app_gate_recover=0.35)
    tracker = OcclusionAwareTracker(cfg)
    agent_id, seen = _theft_scenario(tracker)
    assert agent_id is not None
    late = [f for f in seen.get(agent_id, []) if f >= 46]
    assert late, "agent id must be re-acquired on the TRUE re-emergence"
    impostor_frames = [f for f in seen.get(agent_id, []) if 26 <= f <= 44]
    assert not impostor_frames, "agent id must NOT ride the impostor during the gap"


def test_no_gate_allows_theft() -> None:
    cfg = HiddenConfig(base=BASE, vel_damping=1.0, occl_overlap_thresh=0.0,
                       recover_gate=3.0, app_gate_lost=None, app_gate_recover=None)
    tracker = OcclusionAwareTracker(cfg)
    agent_id, seen = _theft_scenario(tracker)
    assert agent_id is not None
    impostor_frames = [f for f in seen.get(agent_id, []) if 26 <= f <= 44]
    assert impostor_frames, "without the gate, the impostor should steal the id (control)"


def test_no_embeddings_means_no_behavior_change() -> None:
    cfg_gated = HiddenConfig(base=BASE, app_gate_lost=0.35, app_gate_recover=0.35)
    cfg_off = HiddenConfig(base=BASE, app_gate_lost=None, app_gate_recover=None)
    rng = np.random.default_rng(0)
    frames = {
        f: np.stack([_det(100.0 + 5.0 * (f - 1) + rng.normal(0, 0.5), 200.0)])
        for f in range(1, 40)
    }
    outs = []
    for cfg in (cfg_gated, cfg_off):
        tracker = OcclusionAwareTracker(cfg)
        rows = [tracker.update(frames[f], frame_id=f) for f in sorted(frames)]
        outs.append(np.vstack([r for r in rows if len(r)]))
    np.testing.assert_array_equal(outs[0], outs[1])
