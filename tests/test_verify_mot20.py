"""Synthetic tests for verify_mot20.py's spec-table checking (D37 MOT20 integration).
No MOT20 data needed — fabricated seqinfo/gt trees exercise pass and fail paths."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_module(name: str, rel_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vm = _load_module("verify_mot20_test", "scripts/verify_mot20.py")

SEQINFO = """[Sequence]
name={name}
imDir=img1
frameRate=25
seqLength={length}
imWidth={width}
imHeight={height}
imExt=.jpg
"""


def _make_seq(
    root: Path, name: str, length: int, width: int, height: int,
    n_imgs: int | None = None, gt_rows: list[list[float]] | None = None,
) -> Path:
    d = root / "train" / name
    (d / "img1").mkdir(parents=True)
    (d / "gt").mkdir(parents=True)
    (d / "seqinfo.ini").write_text(
        SEQINFO.format(name=name, length=length, width=width, height=height), encoding="utf-8"
    )
    n_imgs = length if n_imgs is None else n_imgs
    for i in range(1, n_imgs + 1):
        (d / "img1" / f"{i:06d}.jpg").write_bytes(b"\xff\xd8\xff")  # fake jpg bytes
    if gt_rows is None:
        gt_rows = [[f, 1, 10.0, 10.0, 20.0, 40.0, 1, 1, 0.9] for f in range(1, length + 1)]
    lines = [",".join(str(v) for v in r) for r in gt_rows]
    (d / "gt" / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def test_verify_passes_on_spec_conformant_tree(tmp_path: Path) -> None:
    spec = {"MOT20-01": (5, 1920, 1080)}
    _make_seq(tmp_path, "MOT20-01", 5, 1920, 1080)
    failures, per_seq, hashes = vm.verify(tmp_path, spec=spec)
    assert failures == []
    assert per_seq["MOT20-01"]["passed"] is True
    assert "MOT20-01/gt.txt" in hashes
    assert "MOT20-01/seqinfo.ini" in hashes


def test_verify_fails_on_seqinfo_mismatch(tmp_path: Path) -> None:
    spec = {"MOT20-01": (5, 1920, 1080)}
    _make_seq(tmp_path, "MOT20-01", 5, 1280, 720)  # wrong resolution
    failures, per_seq, _ = vm.verify(tmp_path, spec=spec)
    assert any("seqinfo" in f for f in failures)
    assert per_seq["MOT20-01"]["passed"] is False


def test_verify_fails_on_frame_count_mismatch(tmp_path: Path) -> None:
    spec = {"MOT20-01": (5, 1920, 1080)}
    _make_seq(tmp_path, "MOT20-01", 5, 1920, 1080, n_imgs=3)
    failures, _, _ = vm.verify(tmp_path, spec=spec)
    assert any("img1" in f for f in failures)


def test_verify_fails_on_missing_gt(tmp_path: Path) -> None:
    spec = {"MOT20-01": (5, 1920, 1080)}
    d = _make_seq(tmp_path, "MOT20-01", 5, 1920, 1080)
    (d / "gt" / "gt.txt").unlink()
    failures, _, _ = vm.verify(tmp_path, spec=spec)
    assert any("gt.txt missing" in f for f in failures)


def test_verify_fails_on_frame_out_of_range(tmp_path: Path) -> None:
    spec = {"MOT20-01": (5, 1920, 1080)}
    bad_rows = [[6, 1, 10.0, 10.0, 20.0, 40.0, 1, 1, 0.9]]  # frame 6 > seqLength 5
    _make_seq(tmp_path, "MOT20-01", 5, 1920, 1080, gt_rows=bad_rows)
    failures, per_seq, _ = vm.verify(tmp_path, spec=spec)
    assert any("out of spec" in f for f in failures)
    assert per_seq["MOT20-01"]["frame_range_ok"] is False


def test_verify_fails_on_bad_visibility(tmp_path: Path) -> None:
    spec = {"MOT20-01": (5, 1920, 1080)}
    bad_rows = [[1, 1, 10.0, 10.0, 20.0, 40.0, 1, 1, 1.5]]  # vis > 1
    _make_seq(tmp_path, "MOT20-01", 5, 1920, 1080, gt_rows=bad_rows)
    failures, per_seq, _ = vm.verify(tmp_path, spec=spec)
    assert any("visibility" in f for f in failures)
    assert per_seq["MOT20-01"]["vis_ok"] is False


def test_verify_fails_on_bad_class(tmp_path: Path) -> None:
    spec = {"MOT20-01": (5, 1920, 1080)}
    bad_rows = [[1, 1, 10.0, 10.0, 20.0, 40.0, 1, 99, 0.9]]  # class out of MOTChallenge range
    _make_seq(tmp_path, "MOT20-01", 5, 1920, 1080, gt_rows=bad_rows)
    failures, per_seq, _ = vm.verify(tmp_path, spec=spec)
    assert any("class" in f for f in failures)
    assert per_seq["MOT20-01"]["class_ok"] is False


def test_verify_missing_sequence_dir(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    spec = {"MOT20-01": (5, 1920, 1080)}
    failures, per_seq, _ = vm.verify(tmp_path, spec=spec)
    assert any("sequence dir missing" in f for f in failures)
    assert per_seq["MOT20-01"]["passed"] is False


def test_verify_real_spec_table_values() -> None:
    assert vm.MOT20_SPEC["MOT20-01"] == (429, 1920, 1080)
    assert vm.MOT20_SPEC["MOT20-02"] == (2782, 1920, 1080)
    assert vm.MOT20_SPEC["MOT20-03"] == (2405, 1173, 880)
    assert vm.MOT20_SPEC["MOT20-05"] == (3315, 1654, 1080)
