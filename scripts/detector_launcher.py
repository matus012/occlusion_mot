"""Emit (never submit) the Stage-1 detector array + aggregate its results (D56).

  python scripts/detector_launcher.py --config configs/sweep/perun_detector.yaml --mode slurm
  python scripts/detector_launcher.py --config configs/sweep/perun_detector.yaml --mode local

`--mode slurm` writes results/sweep/perun_detector/submit.sbatch and asserts the grid
against the remaining budget. `--mode local` runs nothing new once every result JSON
exists and writes summary.json: the pre-registered OLS dose-response with a 95% CI.

D49 lesson, encoded here: SLURM `--time` takes HH:MM:SS and coreutils `timeout` takes
NUMBER[smhd]. They are NOT interchangeable; passing the wrong one kills every task in
under a second. Both forms are derived from the same seconds value below and a test
asserts the grammar.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector_unit import (  # noqa: E402
    enumerate_units,
    load_config,
    parse_unit,
    result_path,
    run_unit,
    unit_tag,
)
from eval_detector_map import evaluate as eval_map  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("detector_launcher")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "sweep"


# --------------------------------------------------------------------------- budget

def unit_cost_h(cfg: dict[str, Any], model: str, mix: str) -> float:
    cm = cfg["cost_model"]
    epochs = cfg["epochs"][mix] if isinstance(cfg["epochs"], dict) else cfg["epochs"]
    train = (cm["frames"][mix] * epochs * cm["gflops"][model]
             * float(cm["k_h_per_frame_epoch_gflop"]))
    return train * float(cm["safety_factor"]) + float(cm["eval_overhead_h"])


def _hms_to_seconds(hms_str: str) -> int:
    h, m, sec = (int(v) for v in hms_str.split(":"))
    return h * 3600 + m * 60 + sec


def wall_seconds(cfg: dict[str, Any], model: str, mix: str, est_h: float) -> int:
    """The cap from perun_detector_v1.md section 4, transcribed into the config.

    Deliberately NOT derived from the cost model: the frozen document fixes these, and a
    derived cap came out tighter than the pre-registration allows (e.g. 01:00 vs the
    document's 01:15 for yolo11s/mot20), which would kill units the doc permits to run.
    """
    secs = _hms_to_seconds(cfg["wall_caps"][model][mix])
    assert secs >= est_h * 3600, (
        f"{model}/{mix}: frozen cap {cfg['wall_caps'][model][mix]} is below its own "
        f"estimate {est_h:.2f} h -- the grid cannot fit its own caps"
    )
    return secs


def hms(seconds: int) -> str:
    """SLURM --time form. NOT valid input to coreutils `timeout`."""
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def timeout_arg(seconds: int) -> str:
    """coreutils `timeout` DURATION form (NUMBER[smhd]). NOT SLURM's HH:MM:SS (D49)."""
    return f"{seconds}s"


def budget_table(cfg: dict[str, Any]) -> dict[str, Any]:
    units = enumerate_units(cfg)
    rows = []
    for u in units:
        model, mix, seed = parse_unit(u)
        cost = unit_cost_h(cfg, model, mix)
        secs = wall_seconds(cfg, model, mix, cost)
        rows.append({"unit": u, "model": model, "mix": mix, "seed": seed,
                     "est_h": round(cost, 3), "wall_s": secs,
                     "wall_hms": hms(secs), "wall_timeout": timeout_arg(secs)})
    ceiling = float(cfg["slurm"]["budget_ceiling_h"])
    nominal = sum(r["est_h"] for r in rows)
    worst = sum(r["wall_s"] for r in rows) / 3600.0
    return {
        "n_units": len(rows), "rows": rows, "ceiling_h": ceiling,
        "nominal_h": round(nominal, 2), "worst_case_h": round(worst, 2),
        "max_wall_hms": hms(max(r["wall_s"] for r in rows)),
        "fits": worst <= ceiling,
    }


def assert_budget(cfg: dict[str, Any]) -> dict[str, Any]:
    b = budget_table(cfg)
    logger.info("budget: %d units, nominal %.2f h, worst case at caps %.2f h vs %.2f h ceiling",
                b["n_units"], b["nominal_h"], b["worst_case_h"], b["ceiling_h"])
    assert b["fits"], (
        f"worst case {b['worst_case_h']:.2f} h exceeds the remaining ceiling "
        f"{b['ceiling_h']:.2f} h -- the ceiling does not move (CLAUDE.md); shrink the grid"
    )
    return b


# --------------------------------------------------------------------------- sbatch

def render_sbatch(cfg: dict[str, Any], config_path: Path,
                  results_dir: Path = RESULTS_ROOT) -> str:
    b = assert_budget(cfg)
    slurm, name = cfg["slurm"], cfg["name"]
    units = [r["unit"] for r in b["rows"]]
    cap = int(slurm.get("max_concurrent", 8))

    unit_lines = "\n".join(f'  "{r["unit"]}"' for r in b["rows"])
    result_lines = "\n".join(
        f'  "{result_path(Path("results/sweep"), name, r["model"], r["mix"], r["seed"]).as_posix()}"'  # noqa: E501
        for r in b["rows"])
    limit_lines = "\n".join(f'  "{r["wall_timeout"]}"' for r in b["rows"])
    cfg_posix = config_path.relative_to(ROOT).as_posix() if config_path.is_absolute() \
        else Path(config_path).as_posix()

    return f"""#!/bin/bash
#SBATCH --job-name=det_{name}
#SBATCH --partition={slurm['partition']}
#SBATCH --account={slurm['account']}
#SBATCH --time={b['max_wall_hms']}
#SBATCH --array=0-{len(units) - 1}%{cap}
#SBATCH --gres={slurm.get('gres', 'gpu:1')}
#SBATCH --cpus-per-task={slurm.get('cpus_per_task', 8)}
#SBATCH --mem={slurm.get('mem', '64G')}
#SBATCH --output=results/sweep/{name}/logs/%x_%A_%a.out
#SBATCH --error=results/sweep/{name}/logs/%x_%A_%a.err

# perun_detector_v1.md Stage 1 (FROZEN D53): {b['n_units']} units, nominal
# {b['nominal_h']} h, worst case at caps {b['worst_case_h']} h vs {b['ceiling_h']} h remaining.
# Caps are PER UNIT (model x mix), not per model class -- amendment 9's lesson.

set -euo pipefail

if [ ! -f scripts/detector_unit.py ]; then
  echo "ERROR: submit this script from the repo root" >&2
  exit 2
fi

source scripts/hpc_env.sh

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

if [ -f "$RESULT" ]; then
  echo "skip $UNIT (result exists: $RESULT)"
  exit 0
fi

echo "unit=$UNIT limit=$LIMIT device={cfg['env']['device']} node=$(hostname)"
exec timeout --signal=TERM --kill-after=120 "$LIMIT" \\
  $PYTHON scripts/detector_unit.py --config {cfg_posix} --unit "${{UNIT}}" \\
  --device {cfg['env']['device']}
"""


# --------------------------------------------------------------------------- analysis

def ols_with_ci(xs: list[float], ys: list[float]) -> dict[str, float]:
    """Least-squares slope with a 95% CI (t approx 1.96; n>=10 here). Pure stdlib."""
    n = len(xs)
    assert n >= 3, "need at least 3 points for a slope CI"
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    assert sxx > 0, "zero variance in detector quality -- slope undefined"
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True)]
    dof = n - 2
    se = math.sqrt(sum(r * r for r in resid) / dof / sxx) if dof > 0 else float("nan")
    syy = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum(r * r for r in resid) / syy if syy > 0 else float("nan")
    return {"n": n, "slope": slope, "intercept": intercept, "se": se,
            "ci_lo": slope - 1.96 * se, "ci_hi": slope + 1.96 * se, "r2": r2}


def _mot17_map(tag: str, cache_dir: Path) -> float | None:
    """mAP50-95 on MOT17 dev-half, cached to results/detmap_<tag>.json.

    D60: the dose-response x-axis must be measured on MOT17, per perun_detector_v1.md
    section 3. ultralytics' results.csv reports validation on the TRAINING mix's own split
    (MOT20), which is a different dataset from the y-axis and showed 0.005 of spread
    across units -- no usable x-variance.
    """
    dest = ROOT / "results" / f"detmap_{tag}.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))["mAP50_95_mot17dev"]
    try:
        res = eval_map(tag, ROOT / "data" / "MOT17", cache_dir)
    except FileNotFoundError:
        logger.warning("no cached detections for %s -- excluded from the regression", tag)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res["mAP50_95_mot17dev"]


def _reference_points(cache_dir: Path) -> list[dict[str, Any]]:
    """The two free dose-response levels: the yolo11x baseline and Stage-0 GT.

    Read from disk, never hardcoded, so they cannot drift from the runs that produced them.
    """
    out = []
    for tag, label in (("yolo11x", "baseline yolo11x"), ("gtvis", "Stage-0 GT (visible)")):
        hidden = ROOT / "results" / f"hidden_dev_stage0_{tag}.json"
        if not hidden.exists():
            logger.warning("reference point %s missing (%s) -- skipped", tag, hidden)
            continue
        g2 = json.loads(hidden.read_text(encoding="utf-8"))["g2"]
        m = _mot17_map(tag, cache_dir)
        if m is None:
            continue
        out.append({"unit": label, "model": tag, "mix": "-", "seed": -1, "map50_95": m,
                    "oracle_ceiling": g2["oracle_ceiling"], "id_retention": g2["id_retention"],
                    "id_retention_assoc": g2["id_retention_assoc"], "reference": True})
    return out


def aggregate(cfg: dict[str, Any], results_dir: Path,
              cache_dir: Path | None = None) -> dict[str, Any]:
    cache_dir = cache_dir or (ROOT / "data" / "cache" / "detections")
    name = cfg["name"]
    units = []
    for u in enumerate_units(cfg):
        model, mix, seed = parse_unit(u)
        p = result_path(results_dir, name, model, mix, seed)
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        d["map50_95_mot20_split"] = d.get("map50_95")   # keep the old number, relabelled
        d["map50_95"] = _mot17_map(unit_tag(cfg, model, mix, seed), cache_dir)
        d["reference"] = False
        units.append(d)
    logger.info("aggregating %d/%d units", len(units), len(enumerate_units(cfg)))

    points = units + _reference_points(cache_dir)
    usable = [p for p in points if p.get("map50_95") is not None]
    xs = [p["map50_95"] for p in usable]
    ys_ceiling = [p["oracle_ceiling"] for p in usable]
    ys_e2e = [p["id_retention"] for p in usable]

    out: dict[str, Any] = {
        "name": name,
        "n_units": len(units),
        "n_regression_points": len(usable),
        "x_axis": "mAP50-95 measured on MOT17 dev-half (D60)",
        "units": [
            {k: p.get(k) for k in ("unit", "model", "mix", "seed", "map50_95",
                                   "map50_95_mot20_split", "oracle_ceiling",
                                   "id_retention", "id_retention_assoc", "reference")}
            for p in points
        ],
    }
    if len(usable) >= 3:
        out["dose_response_ceiling"] = ols_with_ci(xs, ys_ceiling)
        out["dose_response_e2e"] = ols_with_ci(xs, ys_e2e)
        ci = out["dose_response_ceiling"]
        out["mechanism_supported"] = bool(ci["ci_lo"] > 0)
        logger.info("PRIMARY oracle_ceiling ~ mAP50-95(MOT17): slope=%.4f CI=[%.4f, %.4f] R2=%.3f",
                    ci["slope"], ci["ci_lo"], ci["ci_hi"], ci["r2"])
        logger.info("mechanism supported (CI excludes zero): %s", out["mechanism_supported"])
    else:
        logger.warning("only %d usable points -- no regression written", len(usable))

    trained = [p for p in points if not p.get("reference")]
    if trained:
        best = max(trained, key=lambda p: p["id_retention"])
        out["best_unit"] = {"unit": best["unit"], "id_retention": best["id_retention"]}
        out["g2b_met"] = bool(best["id_retention"] >= 0.55)
        logger.info("best TRAINED e2e id_retention %.4f (%s) -> G2b >=0.55: %s",
                    best["id_retention"], best["unit"], out["g2b_met"])

    dest = results_dir / name / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("-> %s", dest)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mode", choices=["local", "slurm"], default="local")
    ap.add_argument("--device", default=None)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode == "slurm":
        b = assert_budget(cfg)
        out_dir = args.results_dir / cfg["name"]
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        out = out_dir / "submit.sbatch"
        out.write_text(render_sbatch(cfg, args.config, args.results_dir), encoding="utf-8",
                       newline="\n")
        for r in b["rows"][:4]:
            logger.info("  %-28s est %.2f h  wall %s / %s",
                        r["unit"], r["est_h"], r["wall_hms"], r["wall_timeout"])
        logger.info("emitted (NOT submitted): %s", out)
        return 0

    device = args.device or cfg["env"]["device"]
    for u in enumerate_units(cfg):
        model, mix, seed = parse_unit(u)
        if result_path(args.results_dir, cfg["name"], model, mix, seed).exists():
            logger.info("skip %s (result exists)", u)
            continue
        run_unit(cfg, u, args.config, device, args.results_dir)
    aggregate(cfg, args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
