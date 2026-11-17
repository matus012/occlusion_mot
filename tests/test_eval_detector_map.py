"""Guards for the MOT17-side detector mAP used as the dose-response x-axis (D60).

Synthetic boxes only. The properties that matter: a perfect detector scores 1.0, IoU
thresholding actually bites, duplicate detections are penalised, and ranking by score
behaves like AP (a good detector must outrank a bad one).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


em = _load("eval_detector_map_test", "scripts/eval_detector_map.py")


def test_iou_matrix_identical_boxes_is_one():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert em.iou_matrix(a, a)[0, 0] == pytest.approx(1.0)


def test_iou_matrix_disjoint_is_zero():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[100.0, 100.0, 10.0, 10.0]])
    assert em.iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


def test_iou_matrix_half_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 10.0, 10.0]])          # intersection 50, union 150
    assert em.iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150)


def test_iou_matrix_handles_empty_sides():
    a = np.zeros((0, 4))
    b = np.array([[0.0, 0.0, 1.0, 1.0]])
    assert em.iou_matrix(a, b).shape == (0, 1)
    assert em.iou_matrix(b, a).shape == (1, 0)


def test_perfect_detector_scores_one():
    tp = np.ones(5)
    conf = np.linspace(0.9, 0.5, 5)
    assert em.average_precision(tp, conf, n_gt=5) == pytest.approx(1.0, abs=1e-6)


def test_all_false_positives_score_zero():
    tp = np.zeros(5)
    conf = np.linspace(0.9, 0.5, 5)
    assert em.average_precision(tp, conf, n_gt=5) == pytest.approx(0.0)


def test_missing_half_the_objects_caps_ap():
    """5 GT, 5 correct detections but 10 GT total -> recall caps at 0.5, so AP ~ 0.5."""
    tp = np.ones(5)
    conf = np.linspace(0.9, 0.5, 5)
    ap = em.average_precision(tp, conf, n_gt=10)
    assert 0.45 < ap < 0.55, ap


def test_better_ranking_scores_higher():
    """AP is rank-sensitive: true positives ahead of false ones must score better."""
    conf = np.linspace(0.9, 0.1, 6)
    good = em.average_precision(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), conf, n_gt=3)
    bad = em.average_precision(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), conf, n_gt=3)
    assert good > bad


def test_no_ground_truth_is_nan_not_zero():
    """A sequence with no GT must not silently drag the mean down."""
    assert np.isnan(em.average_precision(np.ones(1), np.ones(1), n_gt=0))


def test_iou_thresholds_are_the_coco_sweep():
    assert len(em.IOU_THRESHOLDS) == 10
    assert em.IOU_THRESHOLDS[0] == pytest.approx(0.5)
    assert em.IOU_THRESHOLDS[-1] == pytest.approx(0.95)
