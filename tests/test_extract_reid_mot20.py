"""Synthetic tests for extract_reid_mot20.py (D37 2c: MOT20 tracklet extraction).
No MOT20 data or image IO needed — split-hint determinism/disjointness and the
row-selection logic are pure functions tested directly."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_module(name: str, rel_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


em = _load_module("extract_reid_mot20_test", "scripts/extract_reid_mot20.py")


def test_split_hints_deterministic_across_calls() -> None:
    idents = [f"MOT20-01_gt{i:04d}" for i in range(50)]
    a = em.split_hints_for_sequence(idents, "MOT20-01", 0.15, seed=0)
    b = em.split_hints_for_sequence(idents, "MOT20-01", 0.15, seed=0)
    assert a == b


def test_split_hints_identity_disjoint() -> None:
    idents = [f"MOT20-01_gt{i:04d}" for i in range(60)]
    hints = em.split_hints_for_sequence(idents, "MOT20-01", 0.15, seed=0)
    assert set(hints) == set(idents)
    assert set(hints.values()) <= {"train", "val"}


def test_split_hints_ratio_close_to_target() -> None:
    idents = [f"MOT20-02_gt{i:04d}" for i in range(200)]
    hints = em.split_hints_for_sequence(idents, "MOT20-02", 0.15, seed=0)
    n_val = sum(1 for v in hints.values() if v == "val")
    assert n_val == round(0.15 * 200)


def test_split_hints_differ_by_sequence_name() -> None:
    idents = [f"gt{i:04d}" for i in range(80)]
    hints_a = em.split_hints_for_sequence(idents, "MOT20-01", 0.15, seed=0)
    hints_b = em.split_hints_for_sequence(idents, "MOT20-02", 0.15, seed=0)
    assert hints_a != hints_b


def test_split_hints_differ_by_seed() -> None:
    idents = [f"gt{i:04d}" for i in range(80)]
    hints_seed0 = em.split_hints_for_sequence(idents, "MOT20-01", 0.15, seed=0)
    hints_seed1 = em.split_hints_for_sequence(idents, "MOT20-01", 0.15, seed=1)
    assert hints_seed0 != hints_seed1


def _row(frame: int, tid: int, vis: float = 0.9) -> list[float]:
    from omot.io.mot_format import COL, N_COLS
    r = [-1.0] * N_COLS
    r[COL.FRAME] = frame
    r[COL.ID] = tid
    r[COL.X], r[COL.Y], r[COL.W], r[COL.H] = 10.0, 10.0, 20.0, 40.0
    r[COL.CONF] = 1.0
    r[COL.CLS] = 1.0
    r[COL.VIS] = vis
    return r


def test_select_rows_drops_sparse_identities() -> None:
    rows = np.array([_row(f, 1) for f in range(1, 5)] + [_row(f, 2) for f in range(1, 20)])
    selected = em.select_rows_for_identities(rows, min_crops=10, max_crops_per_id=200)
    assert 1 not in selected  # only 4 rows, below min_crops
    assert 2 in selected
    assert len(selected[2]) == 19


def test_select_rows_caps_dense_identities_evenly() -> None:
    rows = np.array([_row(f, 1) for f in range(1, 501)])  # 500 rows for one identity
    selected = em.select_rows_for_identities(rows, min_crops=10, max_crops_per_id=50)
    assert len(selected[1]) <= 50
    from omot.io.mot_format import COL
    frames = sorted(int(r[COL.FRAME]) for r in selected[1])
    # even sampling across the frame span: first and last frame preserved
    assert frames[0] == 1
    assert frames[-1] == 500


def test_extract_sequence_end_to_end(tmp_path: Path) -> None:
    """Full extract_sequence over a fabricated 2-identity sequence (real image IO)."""
    import cv2

    from omot.data.mot import load_sequence
    from omot.io.mot_format import write_mot

    seq_dir = tmp_path / "MOT20-01"
    (seq_dir / "img1").mkdir(parents=True)
    (seq_dir / "gt").mkdir(parents=True)
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT20-01\nimDir=img1\nframeRate=25\nseqLength=20\n"
        "imWidth=200\nimHeight=150\nimExt=.jpg\n",
        encoding="utf-8",
    )
    rows = []
    for f in range(1, 21):
        img = np.full((150, 200, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(seq_dir / "img1" / f"{f:06d}.jpg"), img)
        rows.append(_row(f, 1, vis=0.9))
        if f > 3:
            rows.append(_row(f, 2, vis=0.3))
    write_mot(seq_dir / "gt" / "gt.txt", np.array(rows))

    seq = load_sequence(seq_dir)
    out_dir = tmp_path / "out"
    index: dict = {}
    n = em.extract_sequence(
        seq, out_dir, index, val_fraction=0.15, seed=0, min_vis=0.0, min_box_h=25.0,
        max_crops_per_id=200, min_crops=10,
    )
    assert n == 20 + 17
    assert set(index) == {"MOT20-01_gt0001", "MOT20-01_gt0002"}
    assert index["MOT20-01_gt0001"]["n"] == 20
    assert index["MOT20-01_gt0002"]["n"] == 17
    assert index["MOT20-01_gt0002"]["n_occluded"] == 17  # vis 0.3 < 0.5
    crops1 = list((out_dir / "MOT20-01_gt0001").glob("*.jpg"))
    assert len(crops1) == 20
    assert all(p.stem.endswith("_v090") for p in crops1)
