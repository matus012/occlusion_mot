"""MOT sequence loader on a fabricated MOT17-style directory (no dataset needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omot.data.mot import (
    export_half_gt,
    filter_gt_frames,
    half_split_frames,
    load_sequence,
    load_split,
)
from omot.io.mot_format import COL, write_mot

SEQINFO = """[Sequence]
name={name}
imDir=img1
frameRate=30
seqLength={length}
imWidth=1920
imHeight=1080
imExt=.jpg
"""


def _make_seq(root: Path, name: str, length: int = 20) -> Path:
    d = root / name
    (d / "gt").mkdir(parents=True)
    (d / "seqinfo.ini").write_text(SEQINFO.format(name=name, length=length), encoding="utf-8")
    rows = []
    for f in range(1, length + 1):
        rows.append([f, 1, 10.0 * f, 100.0, 50.0, 120.0, 1, 1, 1.0])
        if f > 5:
            rows.append([f, 2, 500.0, 10.0 * f, 60.0, 110.0, 1, 1, 0.8])
    write_mot(d / "gt" / "gt.txt", np.array(rows))
    return d


def test_load_sequence(tmp_path: Path) -> None:
    _make_seq(tmp_path / "train", "MOT17-99-FRCNN")
    seq = load_sequence(tmp_path / "train" / "MOT17-99-FRCNN")
    assert seq.name == "MOT17-99-FRCNN"
    assert seq.seq_length == 20
    assert seq.img_width == 1920
    assert seq.gt is not None and len(seq.gt) == 20 + 15
    assert seq.diagonal == pytest.approx(np.hypot(1920, 1080))


def test_load_split_filters_detector_variants(tmp_path: Path) -> None:
    for det in ("DPM", "FRCNN", "SDP"):
        _make_seq(tmp_path / "train", f"MOT17-02-{det}")
        _make_seq(tmp_path / "train", f"MOT17-04-{det}")
    seqs = load_split(tmp_path, "train", detector="FRCNN")
    assert [s.name for s in seqs] == ["MOT17-02-FRCNN", "MOT17-04-FRCNN"]


def test_load_split_no_suffix_fallback(tmp_path: Path) -> None:
    _make_seq(tmp_path / "train", "MOT20-01")
    seqs = load_split(tmp_path, "train")
    assert [s.name for s in seqs] == ["MOT20-01"]


def test_load_split_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "test")


def test_export_half_gt(tmp_path: Path) -> None:
    _make_seq(tmp_path / "train", "MOT17-99-FRCNN", length=20)
    seq = load_sequence(tmp_path / "train" / "MOT17-99-FRCNN")
    out_root = tmp_path / "val_gt"
    seq_info = export_half_gt([seq], out_root, half="val")
    assert seq_info == {"MOT17-99-FRCNN": 10}
    from omot.io.mot_format import read_mot

    gt = read_mot(out_root / "MOT17-99-FRCNN" / "gt" / "gt.txt")
    assert gt[:, COL.FRAME].min() == 1 and gt[:, COL.FRAME].max() == 10
    ini = (out_root / "MOT17-99-FRCNN" / "seqinfo.ini").read_text(encoding="utf-8")
    assert "seqLength=10" in ini


def test_half_split_and_filter() -> None:
    dev, val = half_split_frames(21)
    assert list(dev) == list(range(1, 11))
    assert list(val) == list(range(11, 22))
    gt = np.zeros((21, 9))
    gt[:, COL.FRAME] = np.arange(1, 22)
    gt[:, COL.ID] = 1
    out = filter_gt_frames(gt, val)
    assert out.shape[0] == 11
    assert out[0, COL.FRAME] == 1  # rebased
    assert out[-1, COL.FRAME] == 11
    out_raw = filter_gt_frames(gt, val, rebase=False)
    assert out_raw[0, COL.FRAME] == 11
