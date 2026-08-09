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
        "env": {"device": "cuda"},
        "slurm": {"partition": "gpu", "account": "acct1", "time": "00:30:00",
                  "gres": "gpu:1", "cpus_per_task": 8, "mem": "32G"},
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
#
# D43-delta(c): the interpreter now DIFFERS by design between the two modes --
# sys.executable locally vs a $PYTHON env var (defaulting to python3) in the
# emitted sbatch script, because a Windows dev-box .venv interpreter path must
# never leak into a script meant to run on the PERUN Linux cluster. Parity is
# therefore asserted on the ARGUMENT LIST that follows the interpreter (same
# script, same flag order), masking out unit/device VALUES and the config-path
# REPRESENTATION (native path locally vs POSIX-relative-to-repo-root for the
# Linux target -- these necessarily differ in separator/absoluteness when
# rendered on a Windows machine for a path outside the repo, e.g. a pytest
# tmp_path fixture).
# ---------------------------------------------------------------------------


def test_sbatch_argument_list_matches_local_mode_after_interpreter(tmp_path: Path) -> None:
    cfg = _base_cfg()
    config_path = _write_config(tmp_path, cfg)
    results_dir = tmp_path / "results"
    concrete_unit = "D:300:0"

    # the exact argument list local mode would subprocess (interpreter prepended
    # separately by run_subprocess -- sys.executable -- not part of entrypoint_cmd)
    local_args = sc.entrypoint_cmd(config_path, concrete_unit, device="cuda")

    sbatch_text = sl.render_sbatch(cfg, config_path, results_dir)

    config_posix = sc.posix_relpath(config_path)
    # D47: the emitted line quotes its expansions ("${UNIT}") because the script runs
    # under `set -u`; parity is over the ARGUMENT LIST, so compare against the quoted form.
    template_args = sc.entrypoint_cmd(config_posix, '"${UNIT}"', device='"${DEVICE}"')

    def _mask(args: list[str], cfg_tok: str, unit_tok: str, dev_tok: str) -> list[str]:
        return [
            "<CFG>" if t == cfg_tok else "<UNIT>" if t == unit_tok
            else "<DEV>" if t == dev_tok else t
            for t in args
        ]

    assert _mask(local_args, str(config_path), concrete_unit, "cuda") == _mask(
        template_args, config_posix, '"${UNIT}"', '"${DEVICE}"'
    )

    # the exact templated command line ($PYTHON interpreter + argument list) is
    # present verbatim in the emitted script
    cmd_line = " ".join(["$PYTHON", *template_args])
    assert cmd_line in sbatch_text
    assert "scripts/sweep_unit.py" in sbatch_text
    assert config_posix in sbatch_text
    assert concrete_unit in _units_array(sbatch_text)


def test_sbatch_python_interpreter_comes_from_the_shared_offline_env(tmp_path: Path) -> None:
    """D47: $PYTHON (and TORCH_HOME / YOLO_CONFIG_DIR / MPLCONFIGDIR) are set in one
    place, scripts/hpc_env.sh, instead of being re-templated into every script."""
    cfg = _base_cfg()
    config_path = _write_config(tmp_path, cfg)
    sbatch_text = sl.render_sbatch(cfg, config_path, tmp_path / "results")
    assert "source scripts/hpc_env.sh" in sbatch_text
    assert "$PYTHON scripts/sweep_unit.py" in sbatch_text
    assert ".venv/Scripts" not in sbatch_text  # no Windows dev interpreter


def test_sbatch_never_leaks_windows_paths(tmp_path: Path) -> None:
    cfg = _base_cfg()
    config_path = _write_config(tmp_path, cfg)
    sbatch_text = sl.render_sbatch(cfg, config_path, tmp_path / "results")
    assert "\\" not in sbatch_text
    assert str(sc.ROOT) not in sbatch_text
    assert "cd " not in sbatch_text  # no cd/chdir to an absolute dev-box path


def test_sbatch_config_path_is_posix_relative_to_repo_root(tmp_path: Path) -> None:
    """Using a config that actually lives inside the repo (the real submission
    case) demonstrates the POSIX-relative behavior end to end, even when this
    test runs on Windows."""
    cfg = _base_cfg()
    real_config = sc.ROOT / "configs" / "sweep" / "dryrun_local.yaml"
    sbatch_text = sl.render_sbatch(cfg, real_config, tmp_path / "results")
    assert "configs/sweep/dryrun_local.yaml" in sbatch_text
    assert "\\" not in sbatch_text
    assert str(sc.ROOT) not in sbatch_text


def test_sbatch_emits_slurm_resource_lines(tmp_path: Path) -> None:
    cfg = _base_cfg()
    config_path = _write_config(tmp_path, cfg)
    sbatch_text = sl.render_sbatch(cfg, config_path, tmp_path / "results")
    assert "#SBATCH --partition=gpu" in sbatch_text
    assert "#SBATCH --account=acct1" in sbatch_text
    assert "#SBATCH --gres=gpu:1" in sbatch_text
    assert "#SBATCH --cpus-per-task=8" in sbatch_text
    assert "#SBATCH --mem=32G" in sbatch_text
    # D47: --time is DERIVED (max per-class wall limit), never copied from slurm.time --
    # a hand-set uniform limit cannot bound a grid whose units differ 6x in cost.
    assert f"#SBATCH --time={sc.budget_table(cfg)['max_time_limit']}" in sbatch_text
    assert "#SBATCH --array=0-" in sbatch_text


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


def test_sbatch_placeholders_used_when_slurm_block_missing(tmp_path: Path) -> None:
    cfg = _base_cfg()
    del cfg["slurm"]
    config_path = _write_config(tmp_path, cfg)
    sbatch_text = sl.render_sbatch(cfg, config_path, tmp_path / "results")
    assert "<PARTITION>" in sbatch_text
    assert "<ACCOUNT>" in sbatch_text


def test_sbatch_device_defaults_to_cuda_when_env_block_missing(tmp_path: Path) -> None:
    cfg = _base_cfg()
    del cfg["env"]
    config_path = _write_config(tmp_path, cfg)
    sbatch_text = sl.render_sbatch(cfg, config_path, tmp_path / "results")
    assert 'DEVICE="cuda"' in sbatch_text


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
