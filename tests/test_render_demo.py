"""Synthetic-data tests for scripts/render_demo.py's pure segment-scoring/picking and
small drawing-helper logic (D39 demo package). No MOT17/CARLA data or ffmpeg needed --
these tests never call any of the frame-rendering functions."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omot.eval.occlusion import OcclusionSegment  # noqa: E402


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "render_demo", ROOT / "scripts" / "render_demo.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses need the module registered (see render_demo.py)
    spec.loader.exec_module(mod)
    return mod


rd = _load_module()


def _seg(track_id: int, lv_frame: int, re_frame: int, height: float) -> OcclusionSegment:
    return OcclusionSegment(
        track_id, lv_frame, re_frame,
        np.array([0.0, 0.0, 20.0, height]), np.array([5.0, 0.0, 20.0, height]),
    )


def _td(rows: list[tuple[int, int, float, float, float, float]]) -> object:
    active = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 6))
    return rd.dv.TrackerData(active=active, coasting=np.zeros((0, 6)))


# ---------------------------------------------------------------------------------
# _seg_key / _legibility_score
# ---------------------------------------------------------------------------------
def test_seg_key_format() -> None:
    seg = _seg(track_id=14, lv_frame=206, re_frame=238, height=99.0)
    assert rd._seg_key("MOT17-02-FRCNN", seg) == "MOT17-02-FRCNN:t14:f206-238"


def test_legibility_score_prefers_bigger_box() -> None:
    small = _seg(1, 10, 40, height=20.0)  # gap 29, in-range for both
    big = _seg(1, 10, 40, height=100.0)
    assert rd._legibility_score(big) > rd._legibility_score(small)


def test_legibility_score_prefers_gap_in_target_band() -> None:
    in_band = _seg(1, 10, 45, height=50.0)  # gap 34, in [15, 60]
    far_out = _seg(1, 10, 500, height=50.0)  # gap 489, way outside
    assert rd._legibility_score(in_band) > rd._legibility_score(far_out)


# ---------------------------------------------------------------------------------
# pick_segments
# ---------------------------------------------------------------------------------
def _make_ctx(seq_name: str, segs: list[OcclusionSegment], tds: dict[str, object]) -> dict:
    return {
        "tds": tds, "dets": {}, "gt": np.zeros((0, 9)), "segs": segs, "mid": 10_000,
    }


def test_pick_segments_selects_expected_patterns_and_hero_dedicated() -> None:
    # seg1: base fails (no post match -> POST_NONE), geom retains -> S1vS2 candidate.
    # in45 retains on seg1 too, so seg1 must NOT qualify as a hero candidate.
    seg1 = _seg(track_id=1, lv_frame=10, re_frame=45, height=80.0)
    base_td = _td([(10, 1, 0, 0, 20, 80)])  # no row at frame 45 -> post_id None -> POST_NONE
    geom_td = _td([(10, 1, 0, 0, 20, 80), (45, 1, 5, 0, 20, 80)])  # same id both ends -> RETAINED
    in45_td = _td([(10, 1, 0, 0, 20, 80), (45, 1, 5, 0, 20, 80)])  # RETAINED (not a hero segment)
    conv_td = _td([(10, 1, 0, 0, 20, 80), (45, 1, 5, 0, 20, 80)])  # RETAINED

    # seg2: base fails (switches), in45 switches, conv retains -> S3vS4 candidate AND hero.
    seg2 = _seg(track_id=2, lv_frame=10, re_frame=45, height=60.0)
    base2_td = _td([(10, 2, 0, 0, 20, 60), (45, 3, 5, 0, 20, 60)])  # different id -> SWITCHED
    geom2_td = _td([(10, 2, 0, 0, 20, 60), (45, 3, 5, 0, 20, 60)])  # doesn't matter for hero
    in452_td = _td([(10, 2, 0, 0, 20, 60), (45, 3, 5, 0, 20, 60)])  # SWITCHED
    conv2_td = _td([(10, 2, 0, 0, 20, 60), (45, 2, 5, 0, 20, 60)])  # same id -> RETAINED

    ctx1 = _make_ctx(
        "SEQ-A", [seg1],
        {"base": base_td, "geom": geom_td, "in45": in45_td, "conv": conv_td},
    )
    ctx2 = _make_ctx(
        "SEQ-B", [seg2],
        {"base": base2_td, "geom": geom2_td, "in45": in452_td, "conv": conv2_td},
    )

    picks = rd.pick_segments({"SEQ-A": ctx1, "SEQ-B": ctx2})

    assert picks["s1_vs_s2"].key == "SEQ-A:t1:f10-45"
    assert picks["s3_vs_s4"].key == "SEQ-B:t2:f10-45"
    assert picks["hero"].key == "SEQ-B:t2:f10-45"  # dedicated hero candidate, not the fallback


def test_pick_segments_hero_falls_back_to_s3_vs_s4_when_no_dedicated_candidate() -> None:
    # Only an S3vS4-shaped segment exists, but its base outcome RETAINS (does not fail),
    # so it cannot satisfy the hero pattern -> hero must fall back to the s3_vs_s4 pick.
    seg = _seg(track_id=5, lv_frame=1, re_frame=30, height=70.0)
    base_td = _td([(1, 5, 0, 0, 20, 70), (30, 5, 5, 0, 20, 70)])  # RETAINED -> base does NOT fail
    geom_td = _td([(1, 5, 0, 0, 20, 70), (30, 5, 5, 0, 20, 70)])
    in45_td = _td([(1, 5, 0, 0, 20, 70), (30, 6, 5, 0, 20, 70)])  # SWITCHED
    conv_td = _td([(1, 5, 0, 0, 20, 70), (30, 5, 5, 0, 20, 70)])  # RETAINED

    # We also need an S1vS2 candidate so pick_segments doesn't assert-fail. Distinct
    # frame numbers (100/130) so merging active arrays below can't cross-match rows
    # belonging to the other synthetic segment (both use the same box geometry).
    seg_s1s2 = _seg(track_id=9, lv_frame=100, re_frame=130, height=70.0)
    base9 = _td([(100, 9, 0, 0, 20, 70)])  # POST_NONE
    geom9 = _td([(100, 9, 0, 0, 20, 70), (130, 9, 5, 0, 20, 70)])  # RETAINED
    in459 = _td([(100, 9, 0, 0, 20, 70), (130, 9, 5, 0, 20, 70)])
    conv9 = _td([(100, 9, 0, 0, 20, 70), (130, 9, 5, 0, 20, 70)])

    ctx = _make_ctx("SEQ-C", [seg, seg_s1s2],
                     {"base": _merge(base_td, base9), "geom": _merge(geom_td, geom9),
                      "in45": _merge(in45_td, in459), "conv": _merge(conv_td, conv9)})

    picks = rd.pick_segments({"SEQ-C": ctx})
    assert picks["hero"].key == picks["s3_vs_s4"].key


def _merge(a: object, b: object) -> object:
    return rd.dv.TrackerData(
        active=np.vstack([a.active, b.active]) if len(a.active) and len(b.active)
        else (a.active if len(a.active) else b.active),
        coasting=np.zeros((0, 6)),
    )


# ---------------------------------------------------------------------------------
# small drawing helpers
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("v,expected", [
    (0.9, (0, 200, 0)),
    (0.5, (0, 200, 0)),
    (0.49, (0, 165, 255)),
    (0.25, (0, 165, 255)),
    (0.1, (0, 0, 220)),
])
def test_vis_color_thresholds(v: float, expected: tuple[int, int, int]) -> None:
    assert rd._vis_color(v) == expected


def test_pick_crowded_window_finds_peak_within_dev_bound() -> None:
    # counts: frame 5 has the most GT rows (peak), all within dev_end=20.
    rows = []
    for f, n in {1: 1, 5: 8, 10: 3, 25: 20}.items():  # frame 25 outside dev range, ignored
        rows += [[f, i, 0, 0, 10, 10, 1, 1, 1.0] for i in range(n)]
    gt = np.array(rows, dtype=np.float64)

    f0, f1 = rd._pick_crowded_window(gt, dev_end=20, window=10)
    assert 1 <= f0 <= 5 <= f1 <= 20
    assert f1 - f0 <= 10


def test_title_card_returns_requested_size() -> None:
    card = rd._title_card(["hello", "world"], (200, 100))
    assert card.shape == (100, 200, 3)
