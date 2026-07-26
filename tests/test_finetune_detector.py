"""Synthetic tests for scripts/finetune_detector.py's pure dataset-prep logic (D18
dev-half-only invariant + GT-to-YOLO conversion) and the D43-delta(b) CARLA-mix
dataset path, no MOT17 data or ultralytics needed."""
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
from omot.io.mot_format import COL, write_mot  # noqa: E402


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


# ---------------------------------------------------------------------------
# D43-delta(b): CARLA-mix dataset prep (synthetic mini MOT17 + CARLA-layout tree)
# ---------------------------------------------------------------------------

_SEQINFO = (
    "[Sequence]\nname={name}\nimDir=img1\nframeRate=20\nseqLength={length}\n"
    "imWidth={width}\nimHeight={height}\nimExt=.jpg\n"
)


def _make_mot_seq(root: Path, name: str, length: int = 20) -> Path:
    """MOT17-FRCNN-style sequence dir with real (dummy-content) jpg files, since
    prepare_dataset hardlinks/copies actual frame files."""
    d = root / name
    (d / "img1").mkdir(parents=True)
    (d / "seqinfo.ini").write_text(
        _SEQINFO.format(name=name, length=length, width=640, height=480), encoding="utf-8"
    )
    rows = []
    for f in range(1, length + 1):
        (d / "img1" / f"{f:06d}.jpg").write_bytes(b"fake-jpg")
        rows.append([f, 1, 10.0, 10.0, 20.0, 40.0, 1, 1, 1.0])
    write_mot(d / "gt" / "gt.txt", np.array(rows))
    return d


def _make_carla_scenario(root: Path, name: str, length: int = 6) -> Path:
    """CARLA scenario dir mirroring the crowd_merge_0017 layout on disk."""
    d = root / name
    (d / "img1").mkdir(parents=True)
    (d / "seqinfo.ini").write_text(
        _SEQINFO.format(name=name, length=length, width=1280, height=720), encoding="utf-8"
    )
    rows = []
    for f in range(1, length + 1):
        (d / "img1" / f"{f:06d}.jpg").write_bytes(b"fake-jpg")
        rows.append([f, 1, 100.0, 100.0, 30.0, 60.0, 1.0, 1, 0.9])
    write_mot(d / "gt" / "gt.txt", np.array(rows))
    return d


def test_default_out_dir_encodes_mix() -> None:
    assert ft._default_out_dir("mot17dev").name == "det_finetune_mot17dev"
    assert ft._default_out_dir("mot17dev_carla").name == "det_finetune_mot17dev_carla"
    assert ft._default_out_dir("mot17dev") != ft._default_out_dir("mot17dev_carla")


def test_carla_scenario_dirs_empty_when_root_missing(tmp_path: Path) -> None:
    assert ft._carla_scenario_dirs(tmp_path / "does_not_exist") == []


def test_carla_scenario_dirs_finds_valid_layout_only(tmp_path: Path) -> None:
    carla_root = tmp_path / "carla_render"
    _make_carla_scenario(carla_root, "crowd_merge_0000", length=3)
    (carla_root / "not_a_scenario").mkdir(parents=True)  # missing seqinfo/gt -> excluded
    dirs = ft._carla_scenario_dirs(carla_root)
    assert [d.name for d in dirs] == ["crowd_merge_0000"]


def test_prepare_dataset_default_mix_unchanged(tmp_path: Path) -> None:
    data_root = tmp_path / "MOT17"
    _make_mot_seq(data_root / "train", "MOT17-02-FRCNN", length=20)
    out_dir = tmp_path / "cache" / "det_finetune_mot17dev"

    dataset_yaml, counts = ft.prepare_dataset(data_root, out_dir, min_vis=0.1, val_frac=0.1)

    n_dev = 10  # mid = seq_length // 2 = 10
    n_val = max(1, round(n_dev * 0.1))
    assert counts == {"train": n_dev - n_val, "val": n_val}
    assert dataset_yaml.exists()
    assert not list((out_dir / "images" / "train").glob("carla_*"))
    assert not list((out_dir / "images" / "val").glob("carla_*"))


def test_prepare_dataset_rejects_unknown_mix(tmp_path: Path) -> None:
    data_root = tmp_path / "MOT17"
    _make_mot_seq(data_root / "train", "MOT17-02-FRCNN", length=20)
    with pytest.raises(AssertionError, match="unknown --mix"):
        ft.prepare_dataset(data_root, tmp_path / "out", min_vis=0.1, mix="bogus")


def test_prepare_dataset_carla_mix_adds_rows_to_train_only(tmp_path: Path) -> None:
    data_root = tmp_path / "MOT17"
    _make_mot_seq(data_root / "train", "MOT17-02-FRCNN", length=20)
    carla_root = tmp_path / "carla_render"
    _make_carla_scenario(carla_root, "crowd_merge_0000", length=6)
    out_dir = tmp_path / "cache" / "det_finetune_mot17dev_carla"

    dataset_yaml, counts = ft.prepare_dataset(
        data_root, out_dir, min_vis=0.1, val_frac=0.1,
        mix="mot17dev_carla", carla_root=carla_root,
    )

    n_dev = 10
    n_val = max(1, round(n_dev * 0.1))
    n_mot_train = n_dev - n_val
    assert counts == {"train": n_mot_train + 6, "val": n_val}  # +6 CARLA frames, TRAIN only
    assert dataset_yaml.exists()

    carla_train_imgs = sorted((out_dir / "images" / "train").glob("carla_*.jpg"))
    assert len(carla_train_imgs) == 6
    carla_train_lbls = sorted((out_dir / "labels" / "train").glob("carla_*.txt"))
    assert len(carla_train_lbls) == 6
    for lbl in carla_train_lbls:
        assert lbl.read_text(encoding="utf-8").strip().startswith("0 ")  # one ped, class 0

    # never leaks into the monitoring val split
    assert list((out_dir / "images" / "val").glob("carla_*")) == []
    assert list((out_dir / "labels" / "val").glob("carla_*")) == []


def test_prepare_dataset_carla_mix_requires_carla_root(tmp_path: Path) -> None:
    data_root = tmp_path / "MOT17"
    _make_mot_seq(data_root / "train", "MOT17-02-FRCNN", length=20)
    with pytest.raises(AssertionError, match="carla-root"):
        ft.prepare_dataset(data_root, tmp_path / "out", min_vis=0.1, mix="mot17dev_carla")


def test_prepare_dataset_carla_mix_fails_fast_when_no_scenarios_found(tmp_path: Path) -> None:
    data_root = tmp_path / "MOT17"
    _make_mot_seq(data_root / "train", "MOT17-02-FRCNN", length=20)
    empty_carla_root = tmp_path / "carla_render_empty"
    empty_carla_root.mkdir()
    with pytest.raises(AssertionError, match="no CARLA scenarios found"):
        ft.prepare_dataset(
            data_root, tmp_path / "out", min_vis=0.1,
            mix="mot17dev_carla", carla_root=empty_carla_root,
        )


def test_prepare_dataset_mix_encoded_paths_never_collide(tmp_path: Path) -> None:
    data_root = tmp_path / "MOT17"
    _make_mot_seq(data_root / "train", "MOT17-02-FRCNN", length=20)
    carla_root = tmp_path / "carla_render"
    _make_carla_scenario(carla_root, "crowd_merge_0000", length=4)

    out_a = tmp_path / "cache" / "det_finetune_mot17dev"
    out_b = tmp_path / "cache" / "det_finetune_mot17dev_carla"
    _, counts_a = ft.prepare_dataset(data_root, out_a, min_vis=0.1, val_frac=0.1)
    _, counts_b = ft.prepare_dataset(
        data_root, out_b, min_vis=0.1, val_frac=0.1,
        mix="mot17dev_carla", carla_root=carla_root,
    )

    assert out_a != out_b
    assert counts_b["train"] == counts_a["train"] + 4
    assert counts_b["val"] == counts_a["val"]
    # both datasets independently on disk, neither touched the other
    assert len(list((out_a / "images" / "train").glob("*.jpg"))) == counts_a["train"]
    assert len(list((out_b / "images" / "train").glob("*.jpg"))) == counts_b["train"]


def test_prepare_dataset_carla_mix_is_idempotent(tmp_path: Path) -> None:
    data_root = tmp_path / "MOT17"
    _make_mot_seq(data_root / "train", "MOT17-02-FRCNN", length=20)
    carla_root = tmp_path / "carla_render"
    _make_carla_scenario(carla_root, "crowd_merge_0000", length=4)
    out_dir = tmp_path / "cache" / "det_finetune_mot17dev_carla"

    _, counts_1 = ft.prepare_dataset(
        data_root, out_dir, min_vis=0.1, val_frac=0.1,
        mix="mot17dev_carla", carla_root=carla_root,
    )
    _, counts_2 = ft.prepare_dataset(
        data_root, out_dir, min_vis=0.1, val_frac=0.1,
        mix="mot17dev_carla", carla_root=carla_root,
    )
    assert counts_1 == counts_2
