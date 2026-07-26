"""Synthetic tests for scripts/sweep_common.py: config schema validation, pool->frac
arithmetic (D43 dry-run rule, incl. arm-B ignore + clamp), unit naming/result-path
round-trips, and gate-selection-by-mean (D43 amendment 2). No model/GPU/data needed."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load_module("sweep_common_test", "scripts/sweep_common.py")


def _base_cfg() -> dict[str, Any]:
    return {
        "name": "dryrun_test",
        "half": "dev",
        "gate_probe": [0.45],
        "epochs": 3,
        "batches_per_epoch": 20,
        "seeds": [0],
        "pools": [300],
        "arms": {
            "A": {"sources": None},
            "B": {"sources": ["sim"]},
            "C": {"sources": ["mot17_dev", "mot20", "market1501"]},
            "D": {"sources": ["mot17_dev", "sim", "mot20", "market1501"]},
        },
        "mcnemar_pairs": [["D", "A"], ["D", "C"], ["C", "A"]],
        "detector": {
            "enabled": True, "models": ["yolo11s"], "mixes": ["mot17dev"],
            "base_weights": None, "epochs": 1, "imgsz": 640, "batch": 4,
        },
    }


def _write_config(tmp_path: Path, cfg: dict[str, Any], name: str = "cfg.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _make_fake_reid(tmp_path: Path, sources: dict[str, dict[str, int]]) -> Path:
    """sources: {source_name: {"train": n_train, "val": n_val}}."""
    reid_root = tmp_path / "reid"
    for src, counts in sources.items():
        identities = {}
        for i in range(counts["train"]):
            identities[f"{src}_train{i:03d}"] = {"split": "train"}
        for i in range(counts["val"]):
            identities[f"{src}_val{i:03d}"] = {"split": "val"}
        d = reid_root / src
        d.mkdir(parents=True)
        (d / "index.json").write_text(json.dumps({"identities": identities}), encoding="utf-8")
    return reid_root


# ---------------------------------------------------------------------------
# config load/validation
# ---------------------------------------------------------------------------


def test_load_config_accepts_valid_dryrun(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _base_cfg())
    cfg = sc.load_config(path)
    assert cfg["name"] == "dryrun_test"
    assert cfg["arms"]["A"]["sources"] is None


def test_load_config_rejects_unknown_arm(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["arms"]["Z"] = {"sources": ["sim"]}
    path = _write_config(tmp_path, cfg)
    with pytest.raises(AssertionError, match="unknown arm"):
        sc.load_config(path)


def test_load_config_rejects_arm_a_with_sources(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["arms"]["A"]["sources"] = ["sim"]
    path = _write_config(tmp_path, cfg)
    with pytest.raises(AssertionError, match="sources: null"):
        sc.load_config(path)


def test_load_config_rejects_bad_source_name(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["arms"]["B"]["sources"] = ["not_a_real_source"]
    path = _write_config(tmp_path, cfg)
    with pytest.raises(AssertionError, match="unknown source"):
        sc.load_config(path)


def test_load_config_rejects_missing_key(tmp_path: Path) -> None:
    cfg = _base_cfg()
    del cfg["seeds"]
    path = _write_config(tmp_path, cfg)
    with pytest.raises(AssertionError, match="missing required key"):
        sc.load_config(path)


def test_load_config_rejects_mcnemar_pair_with_unknown_arm(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["mcnemar_pairs"].append(["D", "Z"])
    path = _write_config(tmp_path, cfg)
    with pytest.raises(AssertionError, match="mcnemar_pairs"):
        sc.load_config(path)


def test_load_config_rejects_empty_pools(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["pools"] = []
    path = _write_config(tmp_path, cfg)
    with pytest.raises(AssertionError, match="pools"):
        sc.load_config(path)


def test_load_config_accepts_per_arm_pool_override(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["pools"] = [None]
    cfg["arms"]["D"]["pools"] = [300, 1000, 2000, 3520]
    path = _write_config(tmp_path, cfg)
    loaded = sc.load_config(path)
    assert loaded["arms"]["D"]["pools"] == [300, 1000, 2000, 3520]


def test_shipped_dryrun_config_is_valid() -> None:
    cfg = sc.load_config(ROOT / "configs" / "sweep" / "dryrun_local.yaml")
    assert cfg["name"] == "dryrun_local"
    assert set(cfg["arms"]) == {"A", "B", "C", "D"}


def test_shipped_perun_full_config_is_valid() -> None:
    cfg = sc.load_config(ROOT / "configs" / "sweep" / "perun_full.yaml")
    assert cfg["name"] == "perun_full"
    assert cfg["arms"]["D"]["pools"] == [300, 1000, 2000, 3520]
    assert cfg["pools"] == [None]


# ---------------------------------------------------------------------------
# pool -> identity-frac arithmetic
# ---------------------------------------------------------------------------


def test_pool_to_frac_arm_b_ignores_pool(tmp_path: Path) -> None:
    reid_root = _make_fake_reid(tmp_path, {"sim": {"train": 36, "val": 9}})
    assert sc.pool_to_frac("B", 300, ["sim"], reid_root=reid_root) == 1.0
    assert sc.pool_to_frac("B", 5, ["sim"], reid_root=reid_root) == 1.0  # ignored either way


def test_pool_to_frac_none_pool_is_full(tmp_path: Path) -> None:
    reid_root = _make_fake_reid(tmp_path, {"mot17_dev": {"train": 200, "val": 50}})
    assert sc.pool_to_frac("D", None, ["mot17_dev"], reid_root=reid_root) == 1.0


def test_pool_to_frac_clamps_when_pool_exceeds_total(tmp_path: Path) -> None:
    reid_root = _make_fake_reid(tmp_path, {"mot17_dev": {"train": 269, "val": 66}})
    assert sc.pool_to_frac("C", 10_000, ["mot17_dev"], reid_root=reid_root) == 1.0


def test_pool_to_frac_clamps_when_pool_equals_total(tmp_path: Path) -> None:
    reid_root = _make_fake_reid(tmp_path, {"mot17_dev": {"train": 100, "val": 20}})
    assert sc.pool_to_frac("C", 100, ["mot17_dev"], reid_root=reid_root) == 1.0


def test_pool_to_frac_computes_ratio_over_combined_sources(tmp_path: Path) -> None:
    reid_root = _make_fake_reid(
        tmp_path, {"mot17_dev": {"train": 150, "val": 30}, "mot20": {"train": 50, "val": 10}}
    )
    frac = sc.pool_to_frac("C", 100, ["mot17_dev", "mot20"], reid_root=reid_root)
    assert frac == pytest.approx(100 / 200)


def test_total_train_ids_counts_only_train_split(tmp_path: Path) -> None:
    reid_root = _make_fake_reid(tmp_path, {"mot17_dev": {"train": 269, "val": 66}})
    assert sc.total_train_ids("mot17_dev", reid_root=reid_root) == 269


# ---------------------------------------------------------------------------
# unit naming / result-path scheme
# ---------------------------------------------------------------------------


def test_embedder_unit_id_and_parse_roundtrip() -> None:
    uid = sc.embedder_unit_id("D", 300, 2)
    assert uid == "D:300:2"
    kind, arm, pool_tok, seed_tok = sc.parse_unit(uid)
    assert (kind, arm, pool_tok, seed_tok) == ("embedder", "D", "300", "2")
    assert sc.token_to_pool(pool_tok) == 300


def test_embedder_unit_id_full_pool_roundtrip() -> None:
    uid = sc.embedder_unit_id("A", None, 0)
    assert uid == "A:full:0"
    kind, arm, pool_tok, seed_tok = sc.parse_unit(uid)
    assert sc.token_to_pool(pool_tok) is None


def test_detector_unit_id_roundtrip() -> None:
    uid = sc.detector_unit_id("yolo11s", "mot17dev")
    assert sc.parse_unit(uid) == ("detector", "yolo11s", "mot17dev")


def test_parse_unit_rejects_malformed() -> None:
    with pytest.raises(AssertionError):
        sc.parse_unit("bad")
    with pytest.raises(AssertionError):
        sc.parse_unit("Z:300:0")


def test_embedder_result_path_scheme(tmp_path: Path) -> None:
    p = sc.embedder_result_path(tmp_path, "dryrun", "D", 300, 0)
    assert p == tmp_path / "dryrun" / "D_p300_s0.json"
    p_full = sc.embedder_result_path(tmp_path, "dryrun", "A", None, 0)
    assert p_full.name == "A_pfull_s0.json"


def test_detector_result_path_scheme(tmp_path: Path) -> None:
    p = sc.detector_result_path(tmp_path, "dryrun", "yolo11s", "mot17dev")
    assert p.name == "detector_yolo11s_mot17dev.json"


def test_result_path_for_unit_dispatches_by_kind(tmp_path: Path) -> None:
    cfg = _base_cfg()
    assert sc.result_path_for_unit("D:300:0", cfg, tmp_path) == sc.embedder_result_path(
        tmp_path, cfg["name"], "D", 300, 0
    )
    assert sc.result_path_for_unit(
        "detector:yolo11s:mot17dev", cfg, tmp_path
    ) == sc.detector_result_path(tmp_path, cfg["name"], "yolo11s", "mot17dev")


def test_run_tag_includes_gate_only_when_given() -> None:
    assert sc.run_tag("dryrun", "D", 300, 0) == "sweep_dryrun_D_p300_s0"
    assert sc.run_tag("dryrun", "D", 300, 0, 0.45) == "sweep_dryrun_D_p300_s0_g0.45"


def test_reference_pool_prefers_full_over_numeric() -> None:
    assert sc.reference_pool([300, 1000, None, 3520]) is None
    assert sc.reference_pool([300, 1000, 2000, 3520]) == 3520


def test_enumerate_units_respects_per_arm_pool_override() -> None:
    cfg = _base_cfg()
    cfg["pools"] = [None]
    cfg["seeds"] = [0, 1, 2]
    cfg["arms"]["D"]["pools"] = [300, 1000, 2000, 3520]
    cfg["detector"]["enabled"] = False
    units = sc.enumerate_units(cfg)
    d_units = [u for u in units if u.startswith("D:")]
    a_units = [u for u in units if u.startswith("A:")]
    assert len(d_units) == 4 * 3  # 4 pools x 3 seeds
    assert len(a_units) == 1 * 3  # 1 (full) pool x 3 seeds


def test_enumerate_units_includes_detector_grid() -> None:
    cfg = _base_cfg()
    cfg["detector"] = {
        "enabled": True, "models": ["yolo11s", "yolo11m"], "mixes": ["mix1", "mix2"],
        "base_weights": None, "epochs": 1, "imgsz": 640, "batch": 4,
    }
    units = sc.enumerate_units(cfg)
    det_units = [u for u in units if u.startswith("detector:")]
    assert len(det_units) == 4  # 2 models x 2 mixes


# ---------------------------------------------------------------------------
# gate selection by mean (D43 amendment 2)
# ---------------------------------------------------------------------------


def test_select_gate_by_mean_picks_best_average_not_per_seed_winner() -> None:
    # gate 0.40 wins two of three seeds individually, but has a catastrophic third
    # seed that drags its MEAN below gate 0.45's steadier mean -- must pick 0.45.
    assoc = {0.40: [0.60, 0.61, 0.10], 0.45: [0.58, 0.59, 0.55]}
    gate, gate_mean, spread = sc.select_gate_by_mean(assoc)
    assert gate == 0.45
    assert gate_mean == pytest.approx((0.58 + 0.59 + 0.55) / 3)
    assert spread == pytest.approx(0.59 - 0.55)


def test_select_gate_by_mean_requires_nonempty_values() -> None:
    with pytest.raises(AssertionError):
        sc.select_gate_by_mean({0.4: []})


def test_select_gate_by_mean_single_seed_spread_is_zero() -> None:
    gate, gate_mean, spread = sc.select_gate_by_mean({0.4: [0.5]})
    assert gate == 0.4
    assert spread == 0.0


# ---------------------------------------------------------------------------
# entrypoint command shape (parity building block; full sbatch parity in
# test_sweep_launcher.py)
# ---------------------------------------------------------------------------


def test_entrypoint_cmd_shape_stable_across_device_presence() -> None:
    cmd_no_device = sc.entrypoint_cmd("cfg.yaml", "A:full:0")
    cmd_device = sc.entrypoint_cmd("cfg.yaml", "A:full:0", device="cuda")
    assert cmd_no_device == ["scripts/sweep_unit.py", "--config", "cfg.yaml",
                             "--unit", "A:full:0"]
    assert cmd_device == [*cmd_no_device, "--device", "cuda"]


def test_config_digest_changes_with_content(tmp_path: Path) -> None:
    p1 = _write_config(tmp_path, _base_cfg(), "a.yaml")
    cfg2 = _base_cfg()
    cfg2["epochs"] = 999
    p2 = _write_config(tmp_path, cfg2, "b.yaml")
    assert sc.config_digest(p1) != sc.config_digest(p2)
    assert sc.config_digest(p1) == sc.config_digest(p1)


def test_arm_sources_arm_a_returns_all_sources() -> None:
    cfg = _base_cfg()
    assert sc.arm_sources(cfg, "A") == sc.ALL_SOURCES
    assert sc.arm_sources(cfg, "B") == ["sim"]


def test_arm_pools_uses_override_when_present() -> None:
    cfg = _base_cfg()
    cfg["arms"]["D"]["pools"] = [300, 1000]
    assert sc.arm_pools(cfg, "D") == [300, 1000]
    assert sc.arm_pools(cfg, "A") == cfg["pools"]


def test_deep_copy_of_base_cfg_is_used_by_helpers() -> None:
    # sanity: mutating a returned _base_cfg() dict never leaks between tests
    cfg1 = _base_cfg()
    cfg2 = copy.deepcopy(cfg1)
    cfg1["arms"]["D"]["pools"] = [1]
    assert "pools" not in cfg2["arms"]["D"]
