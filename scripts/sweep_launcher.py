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
    assert_budget,
    budget_table,
    config_digest,
    detector_result_path,
    embedder_result_path,
    entrypoint_cmd,
    enumerate_units,
    load_config,
    parse_unit,
    posix_relpath,
    reference_pool,
    result_path_for_unit,
    run_subprocess,
    run_tag,
    select_gate_by_mean,
    unit_class,
    write_text_lf,
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


def _slurm_header(cfg: dict, job_name: str, time_limit: str, array: str | None) -> str:
    """The #SBATCH block. partition/account are the ONLY two values a human fills in
    on HPC day, and they are sourced from the config -- never typed into a script."""
    slurm = cfg.get("slurm", {})
    name = cfg["name"]
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={slurm.get('partition', '<PARTITION>')}",
        f"#SBATCH --account={slurm.get('account', '<ACCOUNT>')}",
        f"#SBATCH --time={time_limit}",
    ]
    if array:
        lines.append(f"#SBATCH --array={array}")
    # %A_%a is array-only: on a plain job %A is unset and %a expands to a sentinel
    # like 4294967294, so the smoke job's log lands under a garbage name.
    stem = "%x_%A_%a" if array else "%x_%j"
    lines += [
        f"#SBATCH --gres={slurm.get('gres', 'gpu:1')}",
        f"#SBATCH --cpus-per-task={slurm.get('cpus_per_task', 8)}",
        f"#SBATCH --mem={slurm.get('mem', '64G')}",
        f"#SBATCH --output=results/sweep/{name}/logs/{stem}.out",
        f"#SBATCH --error=results/sweep/{name}/logs/{stem}.err",
    ]
    return "\n".join(lines)


_PREAMBLE = """
set -euo pipefail

# submit FROM the repo root: SLURM runs the task in the submit directory, and every
# path in this script (and inside the entrypoint) is repo-root-relative by design.
if [ ! -f scripts/sweep_unit.py ]; then
  echo "ERROR: submit this script from the repo root (scripts/sweep_unit.py not found)" >&2
  exit 2
fi

# offline env: interpreter, TORCH_HOME, YOLO_CONFIG_DIR, MPLCONFIGDIR, thread caps
source scripts/hpc_env.sh
"""


def render_sbatch(cfg: dict, config_path: Path, results_dir: Path = RESULTS_ROOT) -> str:
    """Emit (never submit) the SLURM job-array script.

    Parity contract (D43-delta(c), unchanged): this array calls the SAME entrypoint
    ARGUMENT LIST as `--mode local` -- scripts/sweep_unit.py --config <cfg> --unit
    <unit> --device <device>. Only the interpreter differs, and every emitted path is
    POSIX and repo-root-relative so a Windows dev box cannot leak a backslash path or
    a .venv\\Scripts interpreter into a Linux script.

    D47 additions:
      * per-unit wall limits -- SLURM's --time is uniform across an array, so the
        per-CLASS limit is enforced inside the task with `timeout`. A unit that
        overruns its class budget is killed and leaves no result JSON, so the next
        submission simply retries it.
      * resume-safety -- each task exits 0 immediately if its result JSON already
        exists, mirroring units_to_run()'s skip rule for local mode. Re-submitting
        the same array after a partial run costs one scheduler slot per done unit.
    """
    units = enumerate_units(cfg)
    budget = assert_budget(cfg)
    slurm = cfg.get("slurm", {})
    device = cfg.get("env", {}).get("device", "cuda")

    max_concurrent = slurm.get("max_concurrent", 8)  # 8x H200 per node -> one node
    array_range = f"0-{len(units) - 1}%{max_concurrent}"

    unit_lines = "\n".join(f'  "{u}"' for u in units)
    result_lines = "\n".join(
        f'  "{posix_relpath(result_path_for_unit(u, cfg, results_dir))}"' for u in units
    )
    # coreutils `timeout` form (NUMBER[smhd]) -- NOT the SLURM HH:MM:SS used by
    # --time above. Passing HH:MM:SS makes every task die instantly on
    # "timeout: invalid time interval", leaving no result JSON (D49).
    limit_lines = "\n".join(
        f'  "{budget["classes"][unit_class(u)]["time_limit_timeout"]}"' for u in units
    )

    config_posix = posix_relpath(config_path)
    # quoted expansions: `set -u` is on and the values come from bash arrays
    cmd_line = " ".join(["$PYTHON", *entrypoint_cmd(config_posix, '"${UNIT}"', '"${DEVICE}"')])

    cls = budget["classes"]
    budget_note = "\n".join(
        f"#   {name:<14} {c['n_units']:>2} units x {c['est_low_h']}-{c['est_high_h']} h "
        f"-> wall limit {c['time_limit']}"
        + (f"  (HARD CAP {c['cap_h']} h: a unit over it is killed, no result JSON, "
           f"billed worst case {c['billed_high_h']} h/unit)" if c["cap_h"] else "")
        for name, c in cls.items() if c["n_units"]
    )

    return f"""{_slurm_header(cfg, f"sweep_{cfg['name']}", budget["max_time_limit"], array_range)}

# Budget guard (perun_sweep_v2.md): {budget['n_units']} units, ceiling
# {budget['ceiling_h']:.0f} H200-h. Grid total {budget['total_low_h']}-{budget['total_high_h']} h.
{budget_note}
# --time above is the array-wide maximum; each task additionally enforces its OWN
# class limit with `timeout`, so a hung unit cannot spend another class's budget.
{_PREAMBLE}
UNITS=(
{unit_lines}
)
RESULTS=(
{result_lines}
)
LIMITS=(
{limit_lines}
)

UNIT="${{UNITS[$SLURM_ARRAY_TASK_ID]}}"
RESULT="${{RESULTS[$SLURM_ARRAY_TASK_ID]}}"
LIMIT="${{LIMITS[$SLURM_ARRAY_TASK_ID]}}"
DEVICE="{device}"

# resume-safety: a unit whose result JSON exists is already done (same rule as
# sweep_launcher.units_to_run for --mode local). Delete the JSON to force a re-run.
if [ -f "$RESULT" ]; then
  echo "skip $UNIT (result exists: $RESULT)"
  exit 0
fi

echo "unit=$UNIT limit=$LIMIT device=$DEVICE node=$(hostname)"
exec timeout --signal=TERM --kill-after=120 "$LIMIT" {cmd_line}
"""


def render_smoke_sbatch(cfg: dict, config_path: Path) -> str:
    """A single short job that must run BEFORE the array (see HPC_RUNBOOK.md).

    It exists for three reasons, in order of importance:

      1. GPU probe -- CLAUDE.md forbids any run >5 min that has silently fallen back
         to CPU. This asserts torch.cuda plus a real device tensor before 40 hours of
         allocation are committed.
      2. Detector dataset prep, serialised -- finetune_detector.py builds
         data/cache/det_finetune_<mix>/ on first use. In the array, the two units
         sharing a mix start concurrently and would race on the same directory.
         Prepping both mixes here removes the race entirely.
      3. Detector-arm smoke -- a 2-epoch yolo11s finetune over the real prepped
         dataset, proving the ultralytics path (offline weights, AMP probe, dataset
         yaml, results.csv parsing) before the 100-epoch runs commit their hours.
    """
    det = cfg.get("detector") or {}
    mixes = det.get("mixes", [])
    smoke_model = (det.get("models") or ["yolo11s"])[0]
    smoke_time = cfg.get("slurm", {}).get("time_smoke", "00:40:00")
    prep = "\n".join(
        f'$PYTHON scripts/finetune_detector.py --prep-only --mix {m}' for m in mixes
    )
    smoke_mix = mixes[-1] if mixes else "mot17dev"

    return f"""{_slurm_header(cfg, f"smoke_{cfg['name']}", smoke_time, None)}
{_PREAMBLE}
echo "=== 1/3 GPU probe (CLAUDE.md: CPU fallback on GPU work = STOP) ==="
$PYTHON - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable -- do NOT submit the sweep"
dev = torch.device("cuda")
x = torch.ones(1024, 1024, device=dev) @ torch.ones(1024, 1024, device=dev)
assert x.device.type == "cuda", x.device
print(f"OK  {{torch.cuda.get_device_name(0)}}  torch={{torch.__version__}}  "
      f"cuda={{torch.version.cuda}}  capability={{torch.cuda.get_device_capability(0)}}")
PY

echo "=== 2/3 detector dataset prep (serialised here so array tasks never race) ==="
{prep}

echo "=== 3/3 detector-arm smoke: 2 epochs, {smoke_model} on mix {smoke_mix} ==="
$PYTHON scripts/finetune_detector.py --epochs 2 --imgsz {det.get('imgsz', 960)} \\
  --batch {det.get('batch', 8)} --mix {smoke_mix} --base-weights {smoke_model}.pt \\
  --tag smoke_{cfg['name']}_{smoke_model} --device cuda

echo "SMOKE OK -- safe to submit results/sweep/{cfg['name']}/submit.sbatch"
"""


def run_slurm(cfg: dict, config_path: Path, results_dir: Path = RESULTS_ROOT) -> Path:
    out_dir = results_dir / cfg["name"]
    # SLURM does not create --output directories; a missing logs/ silently drops
    # every job's stdout and the array looks like it never ran.
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    out = out_dir / "submit.sbatch"
    write_text_lf(out, render_sbatch(cfg, config_path, results_dir))
    smoke = out_dir / "submit_smoke.sbatch"
    write_text_lf(smoke, render_smoke_sbatch(cfg, config_path))

    table = budget_table(cfg)
    logger.info("budget: %d units, %.1f-%.1f h vs %.0f h ceiling (low fits=%s, high fits=%s)",
                table["n_units"], table["total_low_h"], table["total_high_h"],
                table["ceiling_h"], table["fits_low"], table["fits_high"])
    for key, val in cfg.get("slurm", {}).items():
        if isinstance(val, str) and val.startswith("<FILL-"):
            logger.warning("slurm.%s is STILL a placeholder (%s) -- sbatch will be rejected",
                           key, val)
    logger.info("emitted (NOT submitted): %s", smoke)
    logger.info("emitted (NOT submitted): %s", out)
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
