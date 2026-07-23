"""Synthetic tests for index_market1501.py (D37 2c: Market-1501 integration).
No real Market-1501 zip needed — a fabricated mini zip + directory tree exercises
filename parsing (incl. junk exclusion), idempotent extraction, structural verification
(pass/fail/lenient), and index building."""
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_module(name: str, rel_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


im = _load_module("index_market1501_test", "scripts/index_market1501.py")


# ---------------------------------------------------------------------------
# filename -> identity parsing (incl. junk exclusion)
# ---------------------------------------------------------------------------

def test_parse_market_identity_normal() -> None:
    assert im.parse_market_identity("0002_c1s1_000451_03.jpg") == "0002"
    assert im.parse_market_identity("0751_c6s3_075900_02.jpg") == "0751"


def test_parse_market_identity_excludes_junk_zero() -> None:
    assert im.parse_market_identity("0000_c1s1_000151_01.jpg") is None


def test_parse_market_identity_excludes_junk_negative() -> None:
    assert im.parse_market_identity("-1_c1s1_000401_04.jpg") is None


def test_parse_market_identity_unmatched_pattern() -> None:
    assert im.parse_market_identity("readme.txt") is None
    assert im.parse_market_identity("not_a_market_file.jpg") is None


# ---------------------------------------------------------------------------
# idempotent extraction
# ---------------------------------------------------------------------------

def _make_market_zip(zip_path: Path, nested_dirname: str = "Market-1501-v15.09.15") -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in (
            (f"{nested_dirname}/bounding_box_train/0002_c1s1_000451_03.jpg", b"a"),
            (f"{nested_dirname}/bounding_box_train/0003_c1s1_000551_03.jpg", b"b"),
            (f"{nested_dirname}/bounding_box_test/0002_c2s1_000651_03.jpg", b"c"),
            (f"{nested_dirname}/bounding_box_test/0000_c1s1_000151_01.jpg", b"d"),
            (f"{nested_dirname}/bounding_box_test/-1_c1s1_000401_04.jpg", b"e"),
            (f"{nested_dirname}/query/0002_c3s1_000751_01.jpg", b"f"),
        ):
            zf.writestr(name, content)


def test_extract_market_zip_flattens_nested_dir(tmp_path: Path) -> None:
    zip_path = tmp_path / "downloads" / "Market-1501-v15.09.15.zip"
    _make_market_zip(zip_path)
    extract_root = tmp_path / "Market-1501"
    root = im.extract_market_zip(zip_path, extract_root)
    assert root == extract_root
    assert (extract_root / "bounding_box_train").is_dir()
    assert (extract_root / im.MARKER_NAME).exists()
    assert len(list((extract_root / "bounding_box_train").glob("*.jpg"))) == 2


def test_extract_market_zip_idempotent_skips_on_marker(tmp_path: Path) -> None:
    zip_path = tmp_path / "downloads" / "Market-1501-v15.09.15.zip"
    _make_market_zip(zip_path)
    extract_root = tmp_path / "Market-1501"
    im.extract_market_zip(zip_path, extract_root)
    # remove the zip -- if extraction were re-attempted, this would raise
    zip_path.unlink()
    root = im.extract_market_zip(zip_path, extract_root)
    assert root == extract_root
    assert (extract_root / "bounding_box_train").is_dir()


# ---------------------------------------------------------------------------
# structural verification
# ---------------------------------------------------------------------------

def _fake_extracted_root(tmp_path: Path) -> Path:
    root = tmp_path / "Market-1501"
    (root / "bounding_box_train").mkdir(parents=True)
    (root / "bounding_box_test").mkdir(parents=True)
    (root / "query").mkdir(parents=True)
    (root / "bounding_box_train" / "0002_c1s1_000451_03.jpg").write_bytes(b"a")
    (root / "bounding_box_train" / "0003_c1s1_000551_03.jpg").write_bytes(b"b")
    (root / "bounding_box_test" / "0002_c2s1_000651_03.jpg").write_bytes(b"c")
    (root / "bounding_box_test" / "0000_c1s1_000151_01.jpg").write_bytes(b"d")
    (root / "query" / "0002_c3s1_000751_01.jpg").write_bytes(b"e")
    return root


def test_verify_market_structure_passes_with_matching_counts(tmp_path: Path) -> None:
    root = _fake_extracted_root(tmp_path)
    counts = {
        "bounding_box_train": (2, 2),
        "bounding_box_test": (2, None),
        "query": (1, None),
    }
    failures = im.verify_market_structure(root, counts=counts)
    assert failures == []


def test_verify_market_structure_raises_on_mismatch(tmp_path: Path) -> None:
    root = _fake_extracted_root(tmp_path)
    counts = {"bounding_box_train": (999, 999)}
    with pytest.raises(SystemExit):
        im.verify_market_structure(root, counts=counts)


def test_verify_market_structure_lenient_warns_instead_of_raising(tmp_path: Path) -> None:
    root = _fake_extracted_root(tmp_path)
    counts = {"bounding_box_train": (999, 999)}
    failures = im.verify_market_structure(root, lenient=True, counts=counts)
    assert failures  # non-empty, but did not raise


def test_verify_market_structure_identity_count_mismatch(tmp_path: Path) -> None:
    root = _fake_extracted_root(tmp_path)
    counts = {"bounding_box_train": (2, 999)}
    with pytest.raises(SystemExit):
        im.verify_market_structure(root, counts=counts)


# ---------------------------------------------------------------------------
# index building
# ---------------------------------------------------------------------------

def test_build_index_pools_train_and_test_excludes_junk(tmp_path: Path) -> None:
    root = _fake_extracted_root(tmp_path)
    out_dir = tmp_path / "reid" / "market1501"
    index, n_crops = im.build_index(root, out_dir)
    assert n_crops == 3  # 2 train + 1 non-junk test (0000 excluded)
    assert set(index) == {"0002", "0003"}
    assert index["0002"]["n"] == 2  # one from train, one from test
    assert index["0003"]["n"] == 1
    assert all(e["split"] == "train" for e in index.values())


def test_build_index_writes_v100_tagged_crops(tmp_path: Path) -> None:
    root = _fake_extracted_root(tmp_path)
    out_dir = tmp_path / "reid" / "market1501"
    im.build_index(root, out_dir)
    crops = list((out_dir / "0002").glob("*.jpg"))
    assert len(crops) == 2
    assert all(p.stem.endswith("_v100") for p in crops)


def test_build_index_is_idempotent_no_duplicate_files(tmp_path: Path) -> None:
    root = _fake_extracted_root(tmp_path)
    out_dir = tmp_path / "reid" / "market1501"
    im.build_index(root, out_dir)
    im.build_index(root, out_dir)  # re-run
    crops = list((out_dir / "0002").glob("*.jpg"))
    assert len(crops) == 2  # not doubled
