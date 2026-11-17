"""Phase 6 / D37 (2c): unzip + index Market-1501 into the re-ID training contract.

Market-1501 has NO occlusion visibility ground truth, so unlike mot17_dev/mot20 every
identity here is assigned split="train" — Market must never enter the occluded-query val
set (D26 protocol integrity). All crops get the fixed vis tag _v100.jpg.

Two stages:
  1. extract_market_zip(): idempotent unzip of data/downloads/Market-1501-v15.09.15.zip
     to data/Market-1501/ (marker file skips re-extraction; handles the zip's nested
     top-level folder by locating bounding_box_train and flattening).
  2. build_index(): identities pooled from bounding_box_train AND bounding_box_test
     (train + gallery), EXCLUDING junk ids "0000" (distractors) and "-1" (detector
     misses) per the Market-1501 devkit convention. query/ is left unindexed (it's a
     query-only probe set overlapping bounding_box_test identities, not extra identities).

Canonical structure checked before indexing (official v15.09.15 release):
  bounding_box_train: 12,936 jpgs / 751 identities
  bounding_box_test:  19,732 jpgs (750 real identities + junk)
  query:               3,368 jpgs
Mismatches raise SystemExit (fail fast) unless --lenient (mirror-drift tolerance: warn).

Filename convention: <id>_c<cam>s<seq>_<frame>_<n>.jpg, e.g. "0002_c1s1_000451_03.jpg" ->
identity "0002". Crops are copied (hardlinked if possible) and renamed to
"<train|test>_<original-stem>_v100.jpg" under data/reid/market1501/<identity>/.

Usage:
  .venv/Scripts/python.exe scripts/index_market1501.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("index_market1501")

ROOT = Path(__file__).resolve().parents[1]
MARKER_NAME = ".extracted_ok"

_MARKET_FILENAME_RE = re.compile(r"^(-?\d+)_c\d+s\d+_\d+_\d+\.jpg$", re.IGNORECASE)
JUNK_IDS = {"0000", "-1"}

CANONICAL_COUNTS: dict[str, tuple[int, int | None]] = {
    # dirname -> (expected n_jpgs, expected n_non_junk_identities or None if not checked)
    "bounding_box_train": (12_936, 751),
    "bounding_box_test": (19_732, None),
    "query": (3_368, None),
}


def parse_market_identity(filename: str) -> str | None:
    """<id>_c<cam>s<seq>_<frame>_<n>.jpg -> identity string, or None for junk/unmatched."""
    m = _MARKET_FILENAME_RE.match(filename)
    if m is None:
        return None
    ident = m.group(1)
    if ident in JUNK_IDS:
        return None
    return ident


def _locate_canonical_root(search_root: Path) -> Path:
    if (search_root / "bounding_box_train").is_dir():
        return search_root
    for p in search_root.rglob("bounding_box_train"):
        if p.is_dir():
            return p.parent
    raise SystemExit(f"bounding_box_train not found anywhere under {search_root}")


def extract_market_zip(zip_path: Path, extract_root: Path, force: bool = False) -> Path:
    """Idempotent unzip: skips work if the marker file is already present."""
    marker = extract_root / MARKER_NAME
    if marker.exists() and not force:
        logger.info("Market-1501 already extracted at %s (marker present) — skipping unzip",
                    extract_root)
        return extract_root

    assert zip_path.exists(), f"Market-1501 zip not found: {zip_path}"
    tmp_dir = extract_root.parent / f"_market_extract_tmp_{extract_root.name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    logger.info("extracting %s -> %s", zip_path, tmp_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)

    canonical = _locate_canonical_root(tmp_dir)
    extract_root.mkdir(parents=True, exist_ok=True)
    for item in canonical.iterdir():
        dest = extract_root / item.name
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(item), str(dest))
    shutil.rmtree(tmp_dir)
    marker.write_text("ok\n", encoding="utf-8")
    logger.info("Market-1501 extracted -> %s", extract_root)
    return extract_root


def verify_market_structure(
    root: Path, lenient: bool = False,
    counts: dict[str, tuple[int, int | None]] | None = None,
) -> list[str]:
    """Fail-fast structural check against `counts` (defaults to CANONICAL_COUNTS);
    --lenient downgrades to a warning (mirror drift tolerance)."""
    counts = CANONICAL_COUNTS if counts is None else counts
    failures: list[str] = []
    for dirname, (expected_n, expected_ids) in counts.items():
        d = root / dirname
        if not d.is_dir():
            failures.append(f"{dirname}: directory missing")
            continue
        jpgs = sorted(d.glob("*.jpg"))
        if len(jpgs) != expected_n:
            failures.append(f"{dirname}: {len(jpgs)} jpgs, expected {expected_n}")
        if expected_ids is not None:
            ids = {i for i in (parse_market_identity(p.name) for p in jpgs) if i is not None}
            if len(ids) != expected_ids:
                failures.append(
                    f"{dirname}: {len(ids)} non-junk identities, expected {expected_ids}"
                )
    if failures:
        msg = "; ".join(failures)
        if lenient:
            logger.warning("structural verification mismatches (lenient mode): %s", msg)
            return failures
        raise SystemExit(f"Market-1501 structural verification failed: {msg}")
    logger.info("Market-1501 structural verification PASSED (%s)", root)
    return []


def _copy_or_link(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def build_index(root: Path, out_dir: Path) -> tuple[dict[str, dict], int]:
    """Pool identities from bounding_box_train + bounding_box_test (junk excluded);
    all split="train" (no occlusion GT -> never enters the occluded-query val set)."""
    index: dict[str, dict] = {}
    n_crops = 0
    for dirname, tag in (("bounding_box_train", "train"), ("bounding_box_test", "test")):
        d = root / dirname
        for jpg in sorted(d.glob("*.jpg")):
            ident = parse_market_identity(jpg.name)
            if ident is None:
                continue
            dest_dir = out_dir / ident
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{tag}_{jpg.stem}_v100.jpg"
            _copy_or_link(jpg, dest)
            entry = index.setdefault(
                ident, {"n": 0, "n_occluded": 0, "split": "train", "sources": []}
            )
            entry["n"] += 1
            if dirname not in entry["sources"]:
                entry["sources"].append(dirname)
            n_crops += 1
    return index, n_crops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-path", type=Path,
                    default=ROOT / "data" / "downloads" / "Market-1501-v15.09.15.zip")
    ap.add_argument("--extract-root", type=Path, default=ROOT / "data" / "Market-1501")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "reid" / "market1501")
    ap.add_argument("--lenient", action="store_true",
                    help="warn instead of failing on canonical count mismatch (mirror drift)")
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()

    root = extract_market_zip(args.zip_path, args.extract_root, force=args.force_extract)
    verify_market_structure(root, lenient=args.lenient)
    index, n_crops = build_index(root, args.out)

    splits = {s: sum(1 for e in index.values() if e["split"] == s) for s in ("train", "val")}
    meta = {
        "source": "market1501",
        "n_identities": len(index),
        "n_crops": n_crops,
        "identity_split_counts": splits,
        "min_vis": 1.0,
        "identities": index,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("source=market1501 identities=%d crops=%d split=%s -> %s",
                len(index), n_crops, splits, args.out / "index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
