"""Guards for the Stage-0 perfect-detector cache (perun_detector_v1.md, D54).

Synthetic GT only -- no dataset dependency.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omot.detect.cache import load_cached_detections  # noqa: E402
from omot.io.mot_format import COL, N_COLS  # noqa: E402


def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cgd = _load_module("cache_gt_detections_test", "scripts/cache_gt_detections.py")


def _row(frame: int, tid: int, vis: float, cls: float = 1.0, conf: float = 1.0):
    r = np.zeros(N_COLS)
    r[COL.FRAME], r[COL.ID] = frame, tid
    r[COL.X], r[COL.Y], r[COL.W], r[COL.H] = 10.0 * tid, 20.0, 30.0, 60.0
    r[COL.CONF], r[COL.CLS], r[COL.VIS] = conf, cls, vis
    return r


@pytest.fixture
def gt() -> np.ndarray:
    return np.array([
        _row(1, 1, 1.0),            # fully visible pedestrian
        _row(1, 2, 0.0),            # annotated but INVISIBLE -> only gtall keeps it
        _row(2, 1, 0.4),            # partially visible
        _row(2, 3, 1.0, cls=7.0),   # non-pedestrian class -> never kept
        _row(3, 4, 1.0, conf=0.0),  # consider=0 -> never kept
    ])


def test_gtvis_drops_zero_visibility_rows(gt, tmp_path):
    out = tmp_path / "S__gtvis.npz"
    n_frames, n_boxes = cgd.build_gt_cache("S", gt, out, cgd.VARIANTS["gtvis"], "gtvis")
    assert n_boxes == 2, "expected the visible pedestrian rows only"
    dets = load_cached_detections(out)
    assert sorted(dets) == [1, 2]
    assert len(dets[1]) == 1 and len(dets[2]) == 1


def test_gtall_keeps_invisible_but_still_drops_class_and_consider(gt, tmp_path):
    out = tmp_path / "S__gtall.npz"
    _, n_boxes = cgd.build_gt_cache("S", gt, out, cgd.VARIANTS["gtall"], "gtall")
    assert n_boxes == 3, "gtall keeps the vis=0 row, but never distractors or consider=0"


def test_gtall_is_a_superset_of_gtvis(gt, tmp_path):
    """The kill criterion reads gtvis; gtall is the upper-upper bound. If that
    containment ever broke, the two arms would not bracket the truth."""
    a = tmp_path / "S__gtvis.npz"
    b = tmp_path / "S__gtall.npz"
    cgd.build_gt_cache("S", gt, a, cgd.VARIANTS["gtvis"], "gtvis")
    cgd.build_gt_cache("S", gt, b, cgd.VARIANTS["gtall"], "gtall")
    va, vb = load_cached_detections(a), load_cached_detections(b)
    for f, boxes in va.items():
        assert f in vb
        for box in boxes:
            assert any(np.allclose(box, other) for other in vb[f]), (f, box)


def test_scores_are_one_and_boxes_are_tlwh(gt, tmp_path):
    out = tmp_path / "S__gtvis.npz"
    cgd.build_gt_cache("S", gt, out, cgd.VARIANTS["gtvis"], "gtvis")
    dets = load_cached_detections(out)
    box = dets[1][0]
    assert box[4] == pytest.approx(1.0), "a perfect detector is perfectly confident"
    assert box[2] == pytest.approx(30.0) and box[3] == pytest.approx(60.0), "w,h not x2,y2"


def test_cache_is_loadable_by_the_ordinary_detection_reader(gt, tmp_path):
    """Byte-compatibility with cache_detections is the whole design: downstream code
    must not be able to tell a GT cache from a detector cache."""
    out = tmp_path / "S__gtvis.npz"
    cgd.build_gt_cache("S", gt, out, cgd.VARIANTS["gtvis"], "gtvis")
    blob = np.load(out, allow_pickle=False)
    assert set(blob.files) == {"frames", "boxes", "model", "seed"}
    assert blob["boxes"].shape[1] == 5
    assert str(blob["model"]) == "gtvis"


def test_empty_gt_yields_empty_cache(tmp_path):
    out = tmp_path / "S__gtvis.npz"
    n_frames, n_boxes = cgd.build_gt_cache(
        "S", np.zeros((0, N_COLS)), out, cgd.VARIANTS["gtvis"], "gtvis"
    )
    assert (n_frames, n_boxes) == (0, 0)
    assert load_cached_detections(out) == {}
