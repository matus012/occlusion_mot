"""Synthetic tests for scripts/sweep_launcher.py: resumable-skip logic and the
local<->SLURM entrypoint-command PARITY check (the hard design constraint: identical
entrypoint, no code fork). No model/GPU/subprocess execution needed."""
from __future__ import annotations

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


sl = _load_module("sweep_launcher_test", "scripts/sweep_launcher.py")
sc = _load_module("sweep_common_test2", "scripts/sweep_common.py")


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
        "env": {"partition": "gpu", "account": "acct1", "time": "00:30:00",
                "python": "python", "device": "cuda"},
    }


def _write_config(tmp_path: Path, cfg: dict[str, Any]) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# resumable-skip logic
# ---------------------------------------------------------------------------


def test_units_to_run_skips_units_with_existing_results(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["detector"]["enabled"] = False
    units = sc.enumerate_units(cfg)
    results_dir = tmp_path / "results"

    done_unit = units[0]
    done_path = sc.result_path_for_unit(done_unit, cfg, results_dir)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text("{}", encoding="utf-8")

    pending = sl.units_to_run(units, cfg, results_dir, force=False)
    assert done_unit not in pending
    assert len(pending) == len(units) - 1


def test_units_to_run_force_ignores_existing_results(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["detector"]["enabled"] = False
    units = sc.enumerate_units(cfg)
    results_dir = tmp_path / "results"
    done_path = sc.result_path_for_unit(units[0], cfg, results_dir)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text("{}", encoding="utf-8")

    pending = sl.units_to_run(units, cfg, results_dir, force=True)
    assert pending == units


def test_units_to_run_all_pending_on_empty_results_dir(tmp_path: Path) -> None:
    cfg = _base_cfg()
    units = sc.enumerate_units(cfg)
    pending = sl.units_to_run(units, cfg, tmp_path / "results", force=False)
    assert pending == units


# ---------------------------------------------------------------------------
# local <-> SLURM entrypoint parity (the hard design constraint, as a TEST)
# ---------------------------------------------------------------------------


def test_sbatch_uses_identical_entrypoint_shape_as_local_mode(tmp_path: Path) -> None:
    cfg = _base_cfg()
    config_path = _write_config(tmp_path, cfg)
    results_dir = tmp_path / "results"

    # the exact command local mode would subprocess for one concrete unit
    concrete_unit = "D:300:0"
    local_cmd = sc.entrypoint_cmd(config_path, concrete_unit, device="cuda")

    sbatch_text = sl.render_sbatch(cfg, config_path, results_dir)

    # structural parity: same script + same flag shape (only --unit/--device VALUES
    # differ between the local call and the templated sbatch command)
    template_cmd = sc.entrypoint_cmd(config_path, "${UNIT}", device="${DEVICE}")
    fixed_local = [tok for tok in local_cmd if tok not in (concrete_unit, "cuda")]
    fixed_template = [tok for tok in template_cmd if tok not in ("${UNIT}", "${DEVICE}")]
    assert fixed_local == fixed_template

    # the exact templated command line is present verbatim in the emitted script
    cmd_line = " ".join(["python", *template_cmd])
    assert cmd_line in sbatch_text
    assert "scripts/sweep_unit.py" in sbatch_text
    assert f"--config {config_path}" in sbatch_text
    assert concrete_unit in _units_array(sbatch_text)


def _units_array(sbatch_text: str) -> list[str]:
    lines = sbatch_text.splitlines()
    start = lines.index("UNITS=(") + 1
    out = []
    for line in lines[start:]:
        if line.strip() == ")":
            break
        out.append(line.strip().strip('"'))
    return out


def test_sbatch_array_range_matches_unit_count(tmp_path: Path) -> None:
    cfg = _base_cfg()
    config_path = _write_config(tmp_path, cfg)
    units = sc.enumerate_units(cfg)
    sbatch_text = sl.render_sbatch(cfg, config_path, tmp_path / "results")
    assert f"#SBATCH --array=0-{len(units) - 1}" in sbatch_text
    assert len(_units_array(sbatch_text)) == len(units)


def test_sbatch_never_submits(tmp_path: Path) -> None:
    """run_slurm only writes the script to disk -- it must never invoke sbatch."""
    cfg = _base_cfg()
    config_path = _write_config(tmp_path, cfg)
    results_dir = tmp_path / "results"
    out = sl.run_slurm(cfg, config_path, results_dir)
    assert out.exists()
    assert out.name == "submit.sbatch"
    assert out.read_text(encoding="utf-8").startswith("#!/bin/bash")


def test_sbatch_env_placeholders_used_when_missing(tmp_path: Path) -> None:
    cfg = _base_cfg()
    del cfg["env"]
    config_path = _write_config(tmp_path, cfg)
    sbatch_text = sl.render_sbatch(cfg, config_path, tmp_path / "results")
    assert "<PARTITION>" in sbatch_text
    assert "<ACCOUNT>" in sbatch_text


# ---------------------------------------------------------------------------
# aggregation building block: reading unit results into a gate summary
# ---------------------------------------------------------------------------


def test_aggregate_gate_selection_reads_per_gate_blocks_from_unit_results(
    tmp_path: Path,
) -> None:
    cfg = _base_cfg()
    cfg["pools"] = [None]  # reference_pool of [None] is None ("full") -- matches the
                           # embedder_result_path(..., pool=None, ...) writes below
    cfg["gate_probe"] = [0.40, 0.45]
    cfg["seeds"] = [0, 1]
    cfg["detector"]["enabled"] = False
    cfg["mcnemar_pairs"] = []  # skip the paired_test.py subprocess step entirely
    config_path = _write_config(tmp_path, cfg)
    results_dir = tmp_path / "results"

    # arm A only, at its reference pool (full == None), for both seeds
    for seed, assoc_40, assoc_45 in ((0, 0.50, 0.55), (1, 0.52, 0.57)):
        path = sc.embedder_result_path(results_dir, cfg["name"], "A", None, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "unit": f"A:full:{seed}", "occ_metrics": {"occ_rank1": 0.9},
            "per_gate": {
                "0.4": {"id_retention_assoc": assoc_40, "pre_match_rate": 0.9,
                        "oracle_ceiling": 0.8},
                "0.45": {"id_retention_assoc": assoc_45, "pre_match_rate": 0.9,
                         "oracle_ceiling": 0.8},
            },
        }), encoding="utf-8")

    # provide fake results for the other three arms so aggregate() doesn't warn-skip
    for arm in ("B", "C", "D"):
        for seed in cfg["seeds"]:
            path = sc.embedder_result_path(results_dir, cfg["name"], arm, None, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "unit": f"{arm}:full:{seed}", "occ_metrics": {"occ_rank1": 0.5},
                "per_gate": {"0.4": {"id_retention_assoc": 0.3},
                             "0.45": {"id_retention_assoc": 0.3}},
            }), encoding="utf-8")

    summary = sl.aggregate(cfg, config_path, results_dir)
    assert summary["arms"]["A"]["selected_gate"] == 0.45  # higher mean (0.56 vs 0.51)
    assert summary["arms"]["A"]["assoc_mean"] == pytest.approx((0.55 + 0.57) / 2)
    assert summary["mcnemar"] == {}
    assert (results_dir / cfg["name"] / "summary.json").exists()
