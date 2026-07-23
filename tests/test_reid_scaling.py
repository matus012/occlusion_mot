"""Synthetic tests for the D29 identity-scaling study plumbing:
train_reid.py's subsample_identities (nested-subset property) and
scaling_study.py's resume/skip logic -- no MOT17/reid data or torch training needed."""
from __future__ import annotations

import importlib.util
import json
import sys
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


tr = _load_module("train_reid_test", "scripts/train_reid.py")
ss = _load_module("scaling_study_test", "scripts/scaling_study.py")


def _fake_train_items(n_src_a: int = 40, n_src_b: int = 20) -> list[dict]:
    """One record per identity per source (enough for identity-set tests)."""
    items = []
    for i in range(n_src_a):
        items.append({"path": f"a{i}.jpg", "identity": f"mot17_dev/id{i:03d}", "vis": 1.0})
    for i in range(n_src_b):
        items.append({"path": f"b{i}.jpg", "identity": f"sim/id{i:03d}", "vis": 1.0})
    return items


def test_subsample_full_frac_is_identity() -> None:
    items = _fake_train_items()
    out = tr.subsample_identities(items, ["mot17_dev", "sim"], 1.0, seed=0)
    assert out is items


def test_subsample_rejects_out_of_range_frac() -> None:
    items = _fake_train_items()
    with pytest.raises(AssertionError):
        tr.subsample_identities(items, ["mot17_dev", "sim"], 0.0, seed=0)
    with pytest.raises(AssertionError):
        tr.subsample_identities(items, ["mot17_dev", "sim"], 1.5, seed=0)


def test_subsample_counts_match_rounded_fraction() -> None:
    items = _fake_train_items(n_src_a=40, n_src_b=20)
    out = tr.subsample_identities(items, ["mot17_dev", "sim"], 0.5, seed=0)
    ids_a = {r["identity"] for r in out if r["identity"].startswith("mot17_dev/")}
    ids_b = {r["identity"] for r in out if r["identity"].startswith("sim/")}
    assert len(ids_a) == round(0.5 * 40)
    assert len(ids_b) == round(0.5 * 20)


def test_subsample_nested_subset_property() -> None:
    """The D29 hard requirement: identity set at f1 < f2 is a subset of f2 (same seed)."""
    items = _fake_train_items(n_src_a=41, n_src_b=17)  # odd counts to stress rounding
    sources = ["mot17_dev", "sim"]
    seed = 7
    fracs = [0.25, 0.5, 0.75, 1.0]
    id_sets = []
    for f in fracs:
        out = tr.subsample_identities(items, sources, f, seed) if f < 1.0 else items
        id_sets.append({r["identity"] for r in out})
    for smaller, larger in zip(id_sets, id_sets[1:], strict=False):
        assert smaller.issubset(larger), "nested-subset property violated"
    # strictly growing (not degenerate/equal) given these counts
    assert len(id_sets[0]) < len(id_sets[-1])


def test_subsample_deterministic_across_calls() -> None:
    items = _fake_train_items()
    out1 = tr.subsample_identities(items, ["mot17_dev", "sim"], 0.3, seed=3)
    out2 = tr.subsample_identities(items, ["mot17_dev", "sim"], 0.3, seed=3)
    assert {r["identity"] for r in out1} == {r["identity"] for r in out2}


def test_subsample_different_seed_can_differ() -> None:
    items = _fake_train_items(n_src_a=40, n_src_b=20)
    ids_seed0 = {r["identity"] for r in
                tr.subsample_identities(items, ["mot17_dev", "sim"], 0.5, seed=0)}
    ids_seed1 = {r["identity"] for r in
                tr.subsample_identities(items, ["mot17_dev", "sim"], 0.5, seed=1)}
    assert ids_seed0 != ids_seed1


def test_n_train_ids_for_matches_subsample_output() -> None:
    """scaling_study's recompute fallback must agree with train_reid's own logic."""
    # monkeypatch load_items so this stays a pure synthetic test (no data/reid on disk)
    items = _fake_train_items(n_src_a=40, n_src_b=20)
    orig_load_items = tr.load_items
    tr.load_items = lambda sources: (items, [])
    try:
        n = ss.n_train_ids_for(tr, ["mot17_dev", "sim"], 0.5, seed=0)
    finally:
        tr.load_items = orig_load_items
    expected = len({r["identity"] for r in
                    tr.subsample_identities(items, ["mot17_dev", "sim"], 0.5, seed=0)})
    assert n == expected


def test_pct_str() -> None:
    assert ss.pct_str(0.25) == "25"
    assert ss.pct_str(0.5) == "50"
    assert ss.pct_str(0.75) == "75"
    assert ss.pct_str(1.0) == "100"


def test_load_study_missing_file_returns_empty_scaffold(tmp_path: Path) -> None:
    study = ss.load_study(tmp_path / "nope.json")
    assert study == {"config": {}, "results": []}


def test_resume_skip_logic_via_done_fracs(tmp_path: Path) -> None:
    """Fractions already present in the study json must be recognized as done."""
    path = tmp_path / "scaling_study.json"
    seeded = {
        "config": {}, "results": [
            {"fraction": 0.25, "n_train_ids": 10, "occ_rank1": 0.5,
             "id_retention_assoc": 0.5, "n_assoc_scope": 10, "id_retention": 0.4,
             "center_err": 0.01},
        ],
    }
    path.write_text(json.dumps(seeded), encoding="utf-8")
    study = ss.load_study(path)
    done = {r["fraction"] for r in study["results"]}
    assert 0.25 in done
    assert 0.5 not in done


def test_save_study_round_trips_and_dedupes_by_fraction(tmp_path: Path) -> None:
    path = tmp_path / "scaling_study.json"
    study = {"config": {"sources": ["mot17_dev"]}, "results": []}
    ss.save_study(path, study)

    study = ss.load_study(path)
    entry_a = {"fraction": 0.25, "n_train_ids": 10, "occ_rank1": 0.5,
              "id_retention_assoc": 0.5, "n_assoc_scope": 10, "id_retention": 0.4,
              "center_err": 0.01}
    study["results"] = [r for r in study["results"] if r["fraction"] != 0.25] + [entry_a]
    ss.save_study(path, study)

    # simulate a re-run of the same fraction (--force path): dedupe, not append
    entry_a_v2 = {**entry_a, "occ_rank1": 0.6}
    study = ss.load_study(path)
    study["results"] = [r for r in study["results"] if r["fraction"] != 0.25] + [entry_a_v2]
    ss.save_study(path, study)

    final = ss.load_study(path)
    assert len(final["results"]) == 1
    assert final["results"][0]["occ_rank1"] == pytest.approx(0.6)
