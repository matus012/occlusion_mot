"""OcclusionAwareTracker: classification, extended buffer, damping, recovery — all
deterministic hand-built detection streams (agent + static occluder partner)."""
from __future__ import annotations

import numpy as np

from omot.hidden.occlusion_tracker import HiddenConfig, OcclusionAwareTracker
from omot.track.bytetrack import ByteTracker, TrackerConfig

BASE = TrackerConfig(high_thresh=0.25, new_track_thresh=0.35)
W, H = 60.0, 120.0


def _det(x: float, y: float, score: float = 0.9) -> np.ndarray:
    return np.array([x, y, W, H, score])


def _scene_with_gap(
    gap: int, vx: float = 5.0, reappear_offset: float = 0.0, stop_during_gap: bool = False
) -> dict[int, np.ndarray]:
    """Agent moves right at vx; a static partner box overlaps it at the loss frame.
    Agent detections vanish for `gap` frames, then reappear either on-trajectory,
    at the disappearance point (stop_during_gap), or offset by reappear_offset."""
    frames: dict[int, np.ndarray] = {}
    x0, y = 100.0, 200.0
    lose_at = 20  # last visible frame
    partner_x = x0 + vx * (lose_at - 1) + W * 0.5  # overlaps agent box at loss time
    for f in range(1, 20 + gap + 21):
        dets = [_det(partner_x, y)]  # static partner, always visible
        if f <= lose_at:
            dets.append(_det(x0 + vx * (f - 1), y))
        elif f > lose_at + gap:
            if stop_during_gap:
                x = x0 + vx * (lose_at - 1) + reappear_offset
            else:
                x = x0 + vx * (f - 1) + reappear_offset
            dets.append(_det(x, y))
        frames[f] = np.stack(dets)
    return frames


def _ids_after_gap(tracker: ByteTracker, frames: dict[int, np.ndarray], gap: int) -> tuple:
    """Returns (agent id before gap, set of non-partner ids after gap)."""
    before, after = None, set()
    partner_id = None
    for f in sorted(frames):
        out = tracker.update(frames[f], frame_id=f)
        for x, y, w, h, s, tid in out:
            if abs(y - 200.0) > 1:
                continue
            if f <= 20:
                if x < 150 + 5.0 * f:  # leftmost = agent (partner sits ahead)
                    before = tid
                else:
                    partner_id = tid
            elif f > 20 + gap and tid != partner_id:
                after.add(tid)
    return before, after


def test_baseline_loses_long_gap_occl_tracker_retains() -> None:
    gap = 45  # > baseline buffer 30, < occl_buffer 90
    frames = _scene_with_gap(gap, vx=5.0)
    b_before, b_after = _ids_after_gap(ByteTracker(BASE), frames, gap)
    assert b_before is not None and b_after, "baseline sanity"
    assert b_before not in b_after, "baseline should lose the id past its buffer"

    frames = _scene_with_gap(gap, vx=5.0)
    o_before, o_after = _ids_after_gap(
        OcclusionAwareTracker(HiddenConfig(base=BASE, vel_damping=1.0)), frames, gap
    )
    assert o_before is not None and o_before in o_after, "occl tracker must retain the id"


def test_unoccluded_loss_expires_at_base_buffer() -> None:
    # Same gap but NO partner overlap at loss time -> not classified occluded -> base buffer.
    gap = 45
    frames: dict[int, np.ndarray] = {}
    for f in range(1, 20 + gap + 21):
        dets = []
        if f <= 20:
            dets.append(_det(100.0 + 5.0 * (f - 1), 200.0))
        elif f > 20 + gap:
            dets.append(_det(100.0 + 5.0 * (f - 1), 200.0))
        dets.append(_det(900.0, 500.0))  # far-away agent keeps tracker busy
        frames[f] = np.stack(dets)
    tracker = OcclusionAwareTracker(HiddenConfig(base=BASE, vel_damping=1.0))
    seen: dict[int, list[float]] = {}
    for f in sorted(frames):
        for x, y, w, h, s, tid in tracker.update(frames[f], frame_id=f):
            if abs(y - 200.0) < 1:
                seen.setdefault(int(tid), []).append(f)
    assert len(seen) == 2, f"expected new id after unoccluded 45f gap, got {seen.keys()}"


def test_recovery_catches_offset_reemergence() -> None:
    # Reappear 100px off-trajectory: IoU with the coasted prediction is 0, but the
    # scale-normalized center distance ~1.18 < recover_gate 1.5 -> recovered.
    gap = 40
    frames = _scene_with_gap(gap, vx=5.0, reappear_offset=100.0)
    o_before, o_after = _ids_after_gap(
        OcclusionAwareTracker(HiddenConfig(base=BASE, vel_damping=1.0)), frames, gap
    )
    assert o_before is not None and o_before in o_after

    frames = _scene_with_gap(gap, vx=5.0, reappear_offset=100.0)
    tight = OcclusionAwareTracker(HiddenConfig(base=BASE, vel_damping=1.0, recover_gate=0.5))
    t_before, t_after = _ids_after_gap(tight, frames, gap)
    assert t_before not in t_after, "tight gate must reject the offset re-emergence"


def test_damping_tracks_stopping_agents() -> None:
    # Agent STOPS behind the occluder. Damped coasting stays near the disappearance
    # point; undamped CV coasts far ahead. Compare coasting x at the last gap frame.
    gap = 40
    final_coast_x: dict[float, float] = {}
    for damping in (1.0, 0.9):
        frames = _scene_with_gap(gap, vx=5.0, stop_during_gap=True)
        tracker = OcclusionAwareTracker(HiddenConfig(base=BASE, vel_damping=damping))
        for f in sorted(frames):
            tracker.update(frames[f], frame_id=f)
            if f == 20 + gap:
                coast = tracker.coasting
                agent_rows = coast[np.abs(coast[:, 1] - 200.0) < 1]
                assert len(agent_rows) == 1
                final_coast_x[damping] = float(agent_rows[0, 0])
    stop_x = 100.0 + 5.0 * 19  # where the agent actually is
    assert abs(final_coast_x[0.9] - stop_x) < abs(final_coast_x[1.0] - stop_x)
