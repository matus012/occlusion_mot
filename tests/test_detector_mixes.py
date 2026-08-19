"""Guards for the MOT17-disjoint detector mixes (perun_detector_v1.md s1/s5, D56).

Synthetic MOT-layout fixtures only -- no dataset dependency.

The load-bearing property under test is DISJOINTNESS: a mot20* dataset must contain no
MOT17-derived frame. If that ever breaks, every G2b number silently reverts to the
dev-optimistic regime the whole workstream exists to escape, and nothing downstream
would notice.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omot.io.mot_format import COL, N_COLS  # noqa: E402


def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fd = _load_module("finetune_detector_test", "scripts/finetune_detector.py")


def _make_seq(root: Path, name: str, n_frames: int, w: int = 64, h: int = 48) -> Path:
    d = root / name
    (d / "img1").mkdir(parents=True, exist_ok=True)
    (d / "gt").mkdir(parents=True, exist_ok=True)
    for f in range(1, n_frames + 1):
        Image.new("RGB", (w, h), (10, 20, 30)).save(d / "img1" / f"{f:06d}.jpg")
    rows = []
    for f in range(1, n_frames + 1):
        r = np.zeros(N_COLS)
        r[COL.FRAME], r[COL.ID] = f, 1
        r[COL.X], r[COL.Y], r[COL.W], r[COL.H] = 5, 5, 10, 20
        r[COL.CONF], r[COL.CLS], r[COL.VIS] = 1.0, 1.0, 1.0
        rows.append(r)
    np.savetxt(d / "gt" / "gt.txt", np.array(rows), delimiter=",", fmt="%.2f")
    (d / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength={n_frames}\n"
        f"imWidth={w}\nimHeight={h}\nimExt=.jpg\n", encoding="utf-8")
    return d


@pytest.fixture
def roots(tmp_path: Path):
    mot20 = tmp_path / "MOT20"
    _make_seq(mot20 / "train", "MOT20-01", 20)
    _make_seq(mot20 / "train", "MOT20-02", 10)
    mot17 = tmp_path / "MOT17"
    _make_seq(mot17 / "train", "MOT17-02-FRCNN", 20)
    carla = tmp_path / "carla"
    _make_seq(carla, "behind_static_0000", 6)
    return {"mot17": mot17, "mot20": mot20, "carla": carla, "out": tmp_path / "out"}


def test_mot20_mix_uses_every_frame_and_no_mot17(roots):
    """MOT20 is not an eval half, so all 30 frames are trainable (minus the tail
    monitoring split) -- and nothing MOT17 may appear."""
    _yaml, counts = fd.prepare_dataset(
        roots["mot17"], roots["out"], min_vis=0.1, val_frac=0.1, mix="mot20",
        mot20_root=roots["mot20"],
    )
    assert counts["train"] + counts["val"] == 30, counts
    staged = list((roots["out"] / "images").rglob("*.jpg"))
    assert staged, "no frames staged"
    assert not [p for p in staged if "MOT17" in p.name]


def test_mot20_carla_folds_carla_into_train_only(roots):
    _yaml, counts = fd.prepare_dataset(
        roots["mot17"], roots["out"], min_vis=0.1, val_frac=0.1, mix="mot20_carla",
        carla_root=roots["carla"], mot20_root=roots["mot20"],
    )
    assert counts["train"] + counts["val"] == 36, counts  # 30 MOT20 + 6 CARLA
    val_imgs = list((roots["out"] / "images" / "val").glob("*.jpg"))
    assert val_imgs, "no monitoring split"
    assert not [p for p in val_imgs if p.name.startswith("carla_")], \
        "CARLA frames must never enter the monitoring val split"


def test_disjointness_guard_fires_on_a_mot17_source(roots):
    """The guard resolves real paths, so a MOT17 sequence smuggled in under a
    MOT20-looking name is still caught."""
    smuggled = _make_seq(roots["mot17"] / "train", "MOT20-99", 5)
    with pytest.raises(AssertionError, match="MOT17-disjoint"):
        fd._assert_no_mot17([smuggled], "mot20")


def test_disjointness_guard_is_inert_for_mot17_mixes(roots):
    fd._assert_no_mot17([roots["mot17"] / "train" / "MOT17-02-FRCNN"], "mot17dev")


def test_mot17_mixes_keep_the_d18_half_split(roots):
    """Regression: adding the mot20 family must not relax the D18 guard on MOT17."""
    _yaml, counts = fd.prepare_dataset(
        roots["mot17"], roots["out"], min_vis=0.1, val_frac=0.1, mix="mot17dev",
    )
    assert counts["train"] + counts["val"] == 10, "MOT17 must still use the dev half only"


def test_all_frames_splitter_is_disjoint_and_total():
    class _S:
        name, seq_length = "S", 100
    train, monitor = fd._split_all_frames(_S(), 0.1)
    assert set(train).isdisjoint(monitor)
    assert set(train) | set(monitor) == set(range(1, 101))
    assert len(monitor) == 10


def test_mix_family_mapping_is_total():
    assert {m: fd.mix_family(m) for m in fd.VALID_MIXES} == {
        "mot17dev": "mot17dev", "mot17dev_carla": "mot17dev",
        "mot20": "mot20", "mot20_carla": "mot20",
    }
