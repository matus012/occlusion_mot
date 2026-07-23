"""Synthetic tests for scripts/finetune_detector.py's pure dataset-prep logic (D18
dev-half-only invariant + GT-to-YOLO conversion), no MOT17 data or ultralytics needed."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omot.data.mot import MOTSequence  # noqa: E402
from omot.io.mot_format import COL  # noqa: E402


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "finetune_detector", ROOT / "scripts" / "finetune_detector.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ft = _load_module()


def _fake_seq(seq_length: int, gt: np.ndarray | None = None) -> MOTSequence:
    return MOTSequence(
        name="SEQ-01", root=Path("unused"), frame_rate=30.0, seq_length=seq_length,
        img_width=640, img_height=480, gt=gt,
    )


def test_split_dev_frames_never_exceeds_dev_half() -> None:
    seq = _fake_seq(seq_length=101)  # mid = 50
    train_frames, monitor_frames = ft._split_dev_frames(seq, val_frac=0.1)

    all_frames = train_frames + monitor_frames
    assert max(all_frames) <= 50
    assert min(all_frames) == 1
    assert set(train_frames).isdisjoint(monitor_frames)
    assert sorted(all_frames) == list(range(1, 51))
    # last ~10% held out for monitoring
    assert len(monitor_frames) == max(1, round(50 * 0.1))
    assert monitor_frames == list(range(51 - len(monitor_frames), 51))


def test_split_dev_frames_val_frac_zero_still_disjoint() -> None:
    seq = _fake_seq(seq_length=20)  # mid = 10
    train_frames, monitor_frames = ft._split_dev_frames(seq, val_frac=0.0)
    assert set(train_frames) | set(monitor_frames) == set(range(1, 11))
    assert set(train_frames).isdisjoint(monitor_frames)


def test_yolo_label_lines_filters_class_conf_and_visibility() -> None:
    # rows: frame, id, x, y, w, h, conf, cls, vis
    gt = np.array([
        [1, 1, 100, 100, 50, 100, 1, 1, 0.9],  # keep
        [1, 2, 0, 0, 20, 20, 1, 1, 0.05],  # below min_vis -> drop
        [1, 3, 10, 10, 20, 20, 0, 1, 0.9],  # conf flag 0 -> drop
        [1, 4, 10, 10, 20, 20, 1, 2, 0.9],  # class != pedestrian -> drop
        [2, 5, 10, 10, 20, 20, 1, 1, 0.9],  # different frame -> drop
    ], dtype=np.float64)

    lines = ft._yolo_label_lines(gt, frame=1, min_vis=0.1, img_w=640, img_h=480)

    assert len(lines) == 1
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0"
    assert 0.0 <= float(cx) <= 1.0
    assert 0.0 <= float(cy) <= 1.0
    assert 0.0 <= float(w) <= 1.0
    assert 0.0 <= float(h) <= 1.0
    # (100 + 25) / 640, (100 + 50) / 480
    assert float(cx) == pytest.approx(125 / 640, abs=1e-5)
    assert float(cy) == pytest.approx(150 / 480, abs=1e-5)


def test_yolo_label_lines_clips_out_of_bounds_boxes() -> None:
    gt = np.array([
        [1, 1, -50, -50, 100, 100, 1, 1, 1.0],  # box straddles the top-left edge
    ], dtype=np.float64)

    lines = ft._yolo_label_lines(gt, frame=1, min_vis=0.1, img_w=640, img_h=480)

    assert len(lines) == 1
    _, cx, cy, w, h = lines[0].split()
    for v in (cx, cy, w, h):
        assert 0.0 <= float(v) <= 1.0


def test_yolo_label_lines_no_rows_returns_empty() -> None:
    assert ft._yolo_label_lines(np.zeros((0, 9)), frame=1, min_vis=0.1, img_w=640, img_h=480) == []


def test_expected_counts_matches_manual_sum() -> None:
    seqs = [_fake_seq(101), _fake_seq(61)]
    n_train, n_val = ft._expected_counts(seqs, val_frac=0.1)
    manual = 0
    for seq in seqs:
        tr, va = ft._split_dev_frames(seq, val_frac=0.1)
        manual += len(tr) + len(va)
    assert n_train + n_val == manual


def test_col_frame_matches_mot_format() -> None:
    # sanity: the module under test indexes GT via the same COL contract as the rest
    # of the codebase (guards against a silent column-index drift).
    assert COL.FRAME == 0 and COL.CLS == 7 and COL.VIS == 8
