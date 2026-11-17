"""D38 disjointness guard: zero identity overlap between ANY training pool and the
occluded-query eval set, across all re-ID sources (mot17_dev, sim, mot20, market1501).

Identities are namespaced "<source>/<ident>" exactly as train_reid.load_items builds
them, so this asserts the property on the same keys training consumes. Market-1501
additionally must contribute ZERO eval identities (it has no visibility GT — D37/D38).

CI semantics: sources whose data/reid/<src>/index.json is absent are skipped (data is
gitignored; a fresh clone has none) — like the license guard, enforcement is strongest
where it matters: on the machine that trains. check_gates G3 runs this on every check.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REID_ROOT = ROOT / "data" / "reid"
KNOWN_SOURCES = ("mot17_dev", "sim", "mot20", "market1501")
TRAIN_ONLY_SOURCES = ("market1501",)  # no visibility GT -> banned from eval side


def load_indexes() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for src in KNOWN_SOURCES:
        p = REID_ROOT / src / "index.json"
        if p.exists():
            out[src] = json.loads(p.read_text(encoding="utf-8"))
    if not out:
        pytest.skip("no data/reid/*/index.json present (fresh clone) — guard idle")
    return out


def test_every_identity_has_exactly_one_split() -> None:
    for src, idx in load_indexes().items():
        bad = {i: e["split"] for i, e in idx["identities"].items()
               if e.get("split") not in ("train", "val")}
        assert not bad, f"{src}: identities with invalid split labels: {bad}"


def test_zero_overlap_between_training_pool_and_eval_set() -> None:
    train_pool: set[str] = set()
    eval_set: set[str] = set()
    for src, idx in load_indexes().items():
        for ident, entry in idx["identities"].items():
            key = f"{src}/{ident}"
            (train_pool if entry["split"] == "train" else eval_set).add(key)
    overlap = train_pool & eval_set
    assert not overlap, f"identity overlap between train pool and eval set: {overlap}"


def test_market1501_contributes_no_eval_identities() -> None:
    indexes = load_indexes()
    for src in TRAIN_ONLY_SOURCES:
        if src not in indexes:
            continue
        evals = [i for i, e in indexes[src]["identities"].items() if e["split"] == "val"]
        assert not evals, f"{src} has no visibility GT and must not enter eval: {evals}"


def test_mot20_split_manifest_pinned() -> None:
    indexes = load_indexes()
    if "mot20" not in indexes:
        pytest.skip("mot20 not extracted yet")
    idx = indexes["mot20"]
    assert "split_manifest_sha256" in idx, "D38: mot20 index must pin its split manifest"
    import hashlib

    lines = "\n".join(
        f"{ident}:{entry['split']}" for ident, entry in sorted(idx["identities"].items())
    )
    recomputed = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    assert recomputed == idx["split_manifest_sha256"], "mot20 split drifted from manifest"
