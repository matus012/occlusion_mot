"""D43: PERUN sweep launcher -- local sequential driver + SLURM array-script emitter.

Both modes enumerate the IDENTICAL unit list (sweep_common.enumerate_units) and call
the IDENTICAL entrypoint (sweep_common.entrypoint_cmd): scripts/sweep_unit.py --config
<cfg> --unit <unit> [--device ...]. SLURM specifics (partition/account/time, array
mechanics) live ONLY in this launcher and the config's `env` block -- the entrypoint
script itself never changes with execution mode (hard design constraint: no code fork).

Usage:
  .venv/Scripts/python.exe scripts/sweep_launcher.py --config configs/sweep/dryrun_local.yaml \
      --mode local [--device cuda] [--force]
  .venv/Scripts/python.exe scripts/sweep_launcher.py --config configs/sweep/perun_full.yaml \
      --mode slurm
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from sweep_common import (  # noqa: E402
    DEV_OPTIMISTIC_LABEL,
    ROOT,
    arm_pools,
    config_digest,
    detector_result_path,
    embedder_result_path,
    entrypoint_cmd,
    enumerate_units,
    load_config,
    parse_unit,
    reference_pool,
    result_path_for_unit,
    run_subprocess,
    run_tag,
    select_gate_by_mean,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    stream=sys.stdout)
logger = logging.getLogger("sweep_launcher")

RESULTS_ROOT = ROOT / "results" / "sweep"


def units_to_run(
    units: list[str], cfg: dict, results_dir: Path, force: bool = False
) -> list[str]:
    """Resumable-skip: a unit whose result JSON already exists is skipped unless --force."""
    if force:
        return list(units)
    pending: list[str] = []
    for u in units:
        p = result_path_for_unit(u, cfg, results_dir)
        if p.exists():
            logger.info("skip %s (result exists: %s)", u, p)
        else:
            pending.append(u)
    return pending


def run_paired_test(half: str, tag_a: str, tag_b: str) -> dict:
    cmd = [str(ROOT / "scripts" / "paired_test.py"), "--tag-a", tag_a, "--tag-b", tag_b,
           "--half", half]
    run_subprocess(sys.executable, cmd, RESULTS_ROOT / "paired_test.log")
    out_path = ROOT / "results" / f"paired_g2a_{half}_{tag_a}_vs_{tag_b}.json"
    assert out_path.exists(), f"paired_test.py did not produce {out_path}"
    return json.loads(out_path.read_text(encoding="utf-8"))


def aggregate(cfg: dict, config_path: Path, results_dir: Path = RESULTS_ROOT) -> dict:
    """Post-sweep aggregation: per-arm gate selection (D43 amendment 2, mean-across-
    seeds, ONE gate per arm), McNemar per pre-registered pair per seed, and a
    summary JSON with all p-values (no cherry-picking)."""
    name = cfg["name"]
    out_dir = results_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    arms_summary: dict[str, dict] = {}
    for arm in sorted(cfg["arms"]):
        ref_pool = reference_pool(arm_pools(cfg, arm))
        assoc_by_gate: dict[float, list[float]] = {g: [] for g in cfg["gate_probe"]}
        occ_rank1s: list[float] = []
        for seed in cfg["seeds"]:
            unit_path = embedder_result_path(results_dir, name, arm, ref_pool, seed)
            if not unit_path.exists():
                logger.warning("missing unit result for gate selection: %s", unit_path)
                continue
            data = json.loads(unit_path.read_text(encoding="utf-8"))
            for g_str, g2 in data.get("per_gate", {}).items():
                g = float(g_str)
                if g in assoc_by_gate:
                    assoc_by_gate[g].append(float(g2["id_retention_assoc"]))
            occ = data.get("occ_metrics", {}).get("occ_rank1")
            if occ is not None:
                occ_rank1s.append(float(occ))
        if not any(assoc_by_gate.values()):
            logger.warning("arm %s: no gate data collected, skipping gate selection", arm)
            continue
        gate, gate_mean, gate_spread = select_gate_by_mean(assoc_by_gate)
        arms_summary[arm] = {
            "reference_pool": ref_pool,
            "selected_gate": gate,
            "assoc_mean": gate_mean,
            "assoc_spread": gate_spread,
            "occ_rank1_mean": (sum(occ_rank1s) / len(occ_rank1s)) if occ_rank1s else None,
            "n_seeds": len(cfg["seeds"]),
        }
        logger.info("arm %s: selected gate=%.2f mean_assoc=%.4f spread=%.4f (D43 amendment 2)",
                    arm, gate, gate_mean, gate_spread)

    mcnemar_summary: dict[str, dict] = {}
    for arm_a, arm_b in cfg["mcnemar_pairs"]:
        pair_key = f"{arm_a}v{arm_b}"
        if arm_a not in arms_summary or arm_b not in arms_summary:
            logger.warning("mcnemar %s: missing arm summary, skipping", pair_key)
            continue
        gate_a = arms_summary[arm_a]["selected_gate"]
        gate_b = arms_summary[arm_b]["selected_gate"]
        pool_a = arms_summary[arm_a]["reference_pool"]
        pool_b = arms_summary[arm_b]["reference_pool"]
        mcnemar_summary[pair_key] = {}
        for seed in cfg["seeds"]:
            # run_hidden.py prefixes its output dirs with "hidden_" (trk_name);
            # paired_test.py consumes the on-disk name, so prefix here.
            tag_a = f"hidden_{run_tag(name, arm_a, pool_a, seed, gate_a)}"
            tag_b = f"hidden_{run_tag(name, arm_b, pool_b, seed, gate_b)}"
            paired = run_paired_test(cfg["half"], tag_a, tag_b)
            dest = out_dir / f"mcnemar_{pair_key}_s{seed}.json"
            dest.write_text(json.dumps(paired, indent=2), encoding="utf-8")
            mcnemar_summary[pair_key][f"s{seed}"] = {
                "p_value": paired["p_value"],
                "n_intersection": paired["n_intersection"],
                "retention_a": paired["retention_a"],
                "retention_b": paired["retention_b"],
            }
            logger.info("mcnemar %s seed=%d p=%.4f -> %s", pair_key, seed,
                        paired["p_value"], dest)

    summary: dict = {
        "name": name,
        "config_digest": config_digest(config_path),
        "arms": arms_summary,
        "mcnemar": mcnemar_summary,
    }

    detector = cfg.get("detector")
    if detector and detector.get("enabled"):
        det_results = []
        for model in detector["models"]:
            for mix in detector["mixes"]:
                p = detector_result_path(results_dir, name, model, mix)
                if p.exists():
                    det_results.append(json.loads(p.read_text(encoding="utf-8")))
        summary["detector"] = det_results
        if det_results:
            logger.info("dev-optimistic: %d detector unit(s) in summary (%s)",
                        len(det_results), DEV_OPTIMISTIC_LABEL)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("-> %s", summary_path)
    return summary


def run_local(
    cfg: dict, config_path: Path, device: str | None, force: bool,
    results_dir: Path = RESULTS_ROOT
) -> dict:
    units = enumerate_units(cfg)
    pending = units_to_run(units, cfg, results_dir, force)
    logger.info("local sweep '%s': %d/%d units pending", cfg["name"], len(pending), len(units))
    python_exe = sys.executable
    log_path = results_dir / cfg["name"] / "sweep.log"
    for unit in pending:
        cmd = entrypoint_cmd(config_path, unit, device)
        run_subprocess(python_exe, cmd, log_path)
    return aggregate(cfg, config_path, results_dir)


def render_sbatch(cfg: dict, config_path: Path, results_dir: Path = RESULTS_ROOT) -> str:
    """Emit (never submit) a SLURM job-array script that runs the SAME entrypoint
    command as `run_local` -- parity is structural: both build the command via
    sweep_common.entrypoint_cmd, only the --unit/--device VALUES differ per index."""
    units = enumerate_units(cfg)
    env = cfg.get("env", {})
    partition = env.get("partition", "<PARTITION>")
    account = env.get("account", "<ACCOUNT>")
    time_limit = env.get("time", "<TIME>")
    python_bin = env.get("python", "python")
    device = env.get("device", "cuda")
    array_range = f"0-{len(units) - 1}"
    unit_lines = "\n".join(f'  "{u}"' for u in units)
    cmd_line = " ".join([python_bin, *entrypoint_cmd(config_path, "${UNIT}", "${DEVICE}")])

    return f"""#!/bin/bash
#SBATCH --job-name=sweep_{cfg['name']}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --array={array_range}
#SBATCH --gpus-per-task=1
#SBATCH --output=results/sweep/{cfg['name']}/slurm_%A_%a.out

# D43 dry-run parity: this array calls the IDENTICAL entrypoint as --mode local
# (scripts/sweep_unit.py --config <cfg> --unit <unit> --device <device>). SLURM
# specifics (partition/account/time/array shape) live ONLY in this file / the
# config's env block -- never in the entrypoint script (no code fork).
set -euo pipefail

UNITS=(
{unit_lines}
)
UNIT="${{UNITS[$SLURM_ARRAY_TASK_ID]}}"
DEVICE="{device}"

cd "{ROOT}"
{cmd_line}
"""


def run_slurm(cfg: dict, config_path: Path, results_dir: Path = RESULTS_ROOT) -> Path:
    script = render_sbatch(cfg, config_path, results_dir)
    out = results_dir / cfg["name"] / "submit.sbatch"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="utf-8")
    logger.info("sbatch array script emitted (NOT submitted) -> %s", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mode", choices=["local", "slurm"], default="local")
    ap.add_argument("--device", default=None, help="cuda|cpu passthrough (local mode only)")
    ap.add_argument("--force", action="store_true", help="re-run units even if results exist")
    ap.add_argument("--seed", type=int, default=0, help="launcher-level seed (no direct "
                    "randomness here beyond determinism hygiene; per-unit seeds come "
                    "from the config's `seeds` list)")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    # fail fast on a config whose mcnemar_pairs/arms would produce unparsable units
    for u in enumerate_units(cfg):
        parse_unit(u)

    if args.mode == "local":
        run_local(cfg, args.config, args.device, args.force, RESULTS_ROOT)
    else:
        run_slurm(cfg, args.config, RESULTS_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
