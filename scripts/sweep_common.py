"""Shared config/schema/plumbing for the PERUN sweep launcher + entrypoint (D43).

Both scripts/sweep_unit.py (the entrypoint) and scripts/sweep_launcher.py (local +
SLURM driver) import this module so unit naming, pool arithmetic, and gate-selection
rules exist in exactly ONE place -- the hard design constraint is that local and
SLURM execution paths can never drift apart (identical entrypoint, no code fork).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import subprocess
from pathlib import Path
from statistics import mean
from types import ModuleType
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
REID_ROOT = ROOT / "data" / "reid"
RESULTS_ROOT = ROOT / "results" / "sweep"

VALID_ARMS = ("A", "B", "C", "D")
ALL_SOURCES = ["mot17_dev", "sim", "mot20", "market1501"]

# Fixed dev-half geometric tracker config the sweep probes the appearance gate over
# (D26 best-known dev geometric config -- same defaults scaling_study.py locked in).
GATE_CONFIG: dict[str, str] = {
    "occl-buffer": "90", "damping": "1.0", "recover-gate": "1.5",
    "overlap-thresh": "0.25", "noise-scale": "1.0", "lowconf-mode": "kf",
}

DEV_OPTIMISTIC_LABEL = (
    "dev-optimistic: detector trained on MOT17 dev-half GT and scored on dev-half "
    "(D43 amendment 3) -- only the R6-at-val number is honest"
)


def _validate_pools(pools: Any, path: Path, ctx: str) -> None:
    assert isinstance(pools, list) and pools, f"{path}: {ctx} pools must be a non-empty list"
    for p in pools:
        assert p is None or (isinstance(p, int) and p > 0), (
            f"{path}: {ctx} pool entries must be null or a positive int, got {p!r}"
        )


def load_config(path: Path) -> dict[str, Any]:
    """Load + validate one sweep YAML config. Fails fast on schema violations."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict), f"{path}: config must be a YAML mapping"
    for key in ("name", "half", "gate_probe", "epochs", "batches_per_epoch",
                "seeds", "pools", "arms", "mcnemar_pairs"):
        assert key in cfg, f"{path}: missing required key '{key}'"
    assert cfg["half"] in ("dev", "val"), f"{path}: half must be dev or val"

    arms = cfg["arms"]
    assert isinstance(arms, dict) and arms, f"{path}: arms must be a non-empty mapping"
    bad_arms = set(arms) - set(VALID_ARMS)
    assert not bad_arms, f"{path}: unknown arm(s) {sorted(bad_arms)} (valid: {VALID_ARMS})"
    for arm, arm_cfg in arms.items():
        assert "sources" in arm_cfg, f"{path}: arm {arm} missing 'sources'"
        sources = arm_cfg["sources"]
        if arm == "A":
            assert sources is None, f"{path}: arm A (ImageNet null) must have sources: null"
        else:
            assert isinstance(sources, list) and sources, (
                f"{path}: arm {arm} sources must be a non-empty list"
            )
            bad_src = set(sources) - set(ALL_SOURCES)
            assert not bad_src, f"{path}: arm {arm} has unknown source(s) {bad_src}"
        if "pools" in arm_cfg:
            _validate_pools(arm_cfg["pools"], path, f"arm {arm}")

    _validate_pools(cfg["pools"], path, "top-level")
    assert cfg["gate_probe"], f"{path}: gate_probe must be non-empty"
    assert cfg["seeds"], f"{path}: seeds must be non-empty"

    for pair in cfg["mcnemar_pairs"]:
        assert len(pair) == 2 and set(pair) <= set(arms), (
            f"{path}: mcnemar_pairs entry {pair} references an arm not in 'arms'"
        )

    detector = cfg.get("detector")
    if detector and detector.get("enabled"):
        for key in ("models", "mixes", "epochs", "imgsz", "batch"):
            assert key in detector, f"{path}: detector.{key} required when enabled"
        assert detector["models"], f"{path}: detector.models must be non-empty"
        assert detector["mixes"], f"{path}: detector.mixes must be non-empty"

    eval_every = cfg.get("eval_every", 1)
    assert isinstance(eval_every, int) and eval_every > 0, (
        f"{path}: eval_every must be a positive int, got {eval_every!r}"
    )

    _warn_on_slurm_placeholders(cfg, path)
    return cfg


def _warn_on_slurm_placeholders(cfg: dict[str, Any], path: Path) -> None:
    """D43-delta(d): a `slurm:` block with unfilled `<FILL-...>` placeholders never
    blocks local use (--mode local doesn't read it); warn so a real submission
    doesn't silently go out with a placeholder partition/account."""
    slurm = cfg.get("slurm")
    if not slurm:
        return
    for key, val in slurm.items():
        if isinstance(val, str) and val.startswith("<FILL-"):
            logger.warning(
                "%s: slurm.%s is an unfilled placeholder (%s) -- fine for local runs, "
                "but SLURM submission is INCOMPLETE until this is filled in",
                path, key, val,
            )


def config_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    """Write with LF endings on every platform.

    Path.write_text() opens in text mode with newline=None, so on Windows every '\\n'
    becomes '\\r\\n'. A CRLF sbatch script reaches the cluster with a '#!/bin/bash\\r'
    shebang and dies with "bad interpreter"; CRLF also corrupts every quoted value in
    the emitted bash arrays. Anything destined for Linux goes through this function.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def pool_token(pool: int | None) -> str:
    return "full" if pool is None else str(int(pool))


def token_to_pool(token: str) -> int | None:
    return None if token == "full" else int(token)


def arm_sources(cfg: dict[str, Any], arm: str) -> list[str]:
    """Sources an arm trains on; arm A (no training) EVALUATES over ALL sources
    (the null baseline is measured on the full val pool, not a per-arm subset)."""
    sources = cfg["arms"][arm]["sources"]
    return list(sources) if sources is not None else list(ALL_SOURCES)


def arm_pools(cfg: dict[str, Any], arm: str) -> list[int | None]:
    """Per-arm pool grid: the arm's own 'pools' override if present, else the
    config's top-level default -- e.g. arm D's identity-scaling curve override."""
    return cfg["arms"][arm].get("pools", cfg["pools"])


def reference_pool(pools: list[int | None]) -> int | None:
    """The pool used for cross-arm comparison (McNemar, gate selection): full pool
    (None) if present, else the largest identity count -- the tail of a scaling
    curve is the closest stand-in for 'the arm's real result'."""
    if None in pools:
        return None
    return max(pools)


def total_train_ids(source: str, reid_root: Path = REID_ROOT) -> int:
    index = json.loads((reid_root / source / "index.json").read_text(encoding="utf-8"))
    return sum(1 for e in index["identities"].values() if e["split"] == "train")


def pool_to_frac(
    arm: str, pool: int | None, sources: list[str], reid_root: Path = REID_ROOT
) -> float:
    """D43 dry-run pool -> identity-frac rule.

    frac = pool / total_train_ids(sources); frac = 1.0 when pool is null or >= total.
    Arm B (sim-only, 36 fixed identities -- D34 blueprint ceiling exhausted) IGNORES
    pool entirely: it always trains on its full fixed pool.
    """
    if arm == "B":
        logger.info("arm B ignores pool=%s (fixed sim identity pool, D34 ceiling exhausted)",
                    pool)
        return 1.0
    if pool is None:
        return 1.0
    total = sum(total_train_ids(src, reid_root) for src in sources)
    if pool >= total:
        logger.info("pool=%d >= total_train_ids=%d for %s -> frac clamped to 1.0",
                    pool, total, sources)
        return 1.0
    frac = pool / total
    logger.info("pool=%d / total_train_ids=%d (%s) -> identity-frac=%.4f",
                pool, total, sources, frac)
    return frac


def embedder_unit_id(arm: str, pool: int | None, seed: int) -> str:
    return f"{arm}:{pool_token(pool)}:{seed}"


def detector_unit_id(model: str, mix: str) -> str:
    return f"detector:{model}:{mix}"


def parse_unit(unit: str) -> tuple[str, ...]:
    """'A:300:0' -> ('embedder', 'A', '300', '0'); 'detector:yolo11s:mix' ->
    ('detector', 'yolo11s', 'mix'). Fails fast on malformed / unknown-arm units."""
    parts = unit.split(":")
    assert len(parts) == 3, f"malformed --unit '{unit}' (expected 'a:b:c')"
    kind, a, b = parts
    if kind == "detector":
        return ("detector", a, b)
    assert kind in VALID_ARMS, f"malformed --unit '{unit}': unknown arm '{kind}'"
    return ("embedder", kind, a, b)


def run_tag(name: str, arm: str, pool: int | None, seed: int, gate: float | None = None) -> str:
    tag = f"sweep_{name}_{arm}_p{pool_token(pool)}_s{seed}"
    if gate is not None:
        tag += f"_g{gate}"
    return tag


def embedder_result_path(
    results_dir: Path, name: str, arm: str, pool: int | None, seed: int
) -> Path:
    return results_dir / name / f"{arm}_p{pool_token(pool)}_s{seed}.json"


def detector_result_path(results_dir: Path, name: str, model: str, mix: str) -> Path:
    return results_dir / name / f"detector_{model}_{mix}.json"


def result_path_for_unit(unit: str, cfg: dict[str, Any], results_dir: Path) -> Path:
    parsed = parse_unit(unit)
    name = cfg["name"]
    if parsed[0] == "detector":
        _, model, mix = parsed
        return detector_result_path(results_dir, name, model, mix)
    _, arm, pool_tok, seed_tok = parsed
    return embedder_result_path(results_dir, name, arm, token_to_pool(pool_tok), int(seed_tok))


def enumerate_units(cfg: dict[str, Any]) -> list[str]:
    """Deterministic full unit list for a config -- the single source of truth both
    local and SLURM modes enumerate from (keeps the two execution paths in lockstep)."""
    units: list[str] = []
    for arm in sorted(cfg["arms"]):
        for pool in arm_pools(cfg, arm):
            for seed in cfg["seeds"]:
                units.append(embedder_unit_id(arm, pool, seed))
    detector = cfg.get("detector")
    if detector and detector.get("enabled"):
        for model in detector["models"]:
            for mix in detector["mixes"]:
                units.append(detector_unit_id(model, mix))
    return units


# ---------------------------------------------------------------- budget / SLURM
#
# D47: the 40 H200-hour ceiling (perun_sweep_v2.md "Budget") is enforced at sbatch
# GENERATION time, not discovered at the end of a run. Two independent mechanisms:
#
#   1. per-job wall limits -- each array task is capped at its own unit class's
#      estimate x SLURM_TIME_MARGIN, so one hung unit can never eat the allocation.
#   2. a grid-level assertion -- the LOW-end total must fit under the ceiling (hard
#      fail if not), and a HIGH-end total over the ceiling is reported loudly.
#
# Estimates come from perun_sweep_v2.md's own measured-basis table; they are inputs
# to a guard, never a threshold that moves to make the guard pass.

SLURM_TIME_MARGIN = 1.35  # wall limit = estimate x margin, rounded up to 5 min

# hours per unit, by class. (low, high) brackets the doc's measured range.
UNIT_EST_H: dict[str, tuple[float, float]] = {
    # ImageNet null: retrieval eval only, no training, then the shared embed+track tail
    "embedder_null": (0.45, 0.45),
    # 60 ep x 400 batches with eval-every-5 (~1.2 h) + embedding cache (~0.2 h)
    # + a 3-point gate probe on the dev half (~0.15 h)
    "embedder": (1.55, 1.55),
    # yolo11s/yolo11m, ~100 ep @ 960 px -- the doc's 1.5-3 h/run spread
    "detector": (1.5, 3.0),
}

# D48: hard wall caps, per class, that OVERRIDE estimate x margin when lower.
#
# A cap is a BUDGET instrument, not an estimate -- and it is not free. estimate x margin
# sizes a limit so a healthy unit always finishes; a cap deliberately sits INSIDE the
# estimate's own spread, so a unit at the top of that spread is killed by `timeout` and
# leaves no result JSON (re-submitting the array retries it -- the resume rule already
# covers this). It buys a bounded worst case for the grid.
#
# detector 2.5 h: the class's estimate spread is 1.5-3.0 h/run. Uncapped, the grid's
# worst case is 41.25 h against a 40 h ceiling -- over. Capped, a detector unit cannot
# bill more than 2.5 h, so the worst case is 39.25 h and the ceiling holds by
# construction rather than by hope. The 40 h ceiling itself does not move (CLAUDE.md).
UNIT_TIME_CAP_H: dict[str, float] = {
    "detector": 2.5,
}


def unit_wall_h(cls: str) -> float:
    """Wall-clock limit for one unit of `cls`, in hours: estimate x margin, clamped
    down by the class's hard cap when one is declared."""
    _low, high = UNIT_EST_H[cls]
    wall = high * SLURM_TIME_MARGIN
    cap = UNIT_TIME_CAP_H.get(cls)
    return min(wall, cap) if cap is not None else wall


def unit_class(unit: str) -> str:
    """Which budget/time class an enumerated unit belongs to."""
    parsed = parse_unit(unit)
    if parsed[0] == "detector":
        return "detector"
    return "embedder_null" if parsed[1] == "A" else "embedder"


def _hms(hours: float) -> str:
    """Round UP to the next 5 minutes -- a wall limit must never round down."""
    total_min = int(-(-hours * 60 // 5) * 5)
    return f"{total_min // 60:02d}:{total_min % 60:02d}:00"


def budget_table(cfg: dict[str, Any]) -> dict[str, Any]:
    """Per-class counts, wall limits, and low/high grid totals against the ceiling.

    `billed_high_h` -- not `est_high_h` -- drives the grid's worst case: a unit cannot
    consume more than its own wall limit, because `timeout` kills it there. For an
    uncapped class the two are identical; for a capped one the cap IS the worst case.
    """
    units = enumerate_units(cfg)
    ceiling_h = float(cfg.get("slurm", {}).get("budget_ceiling_h", 40))
    classes: dict[str, dict[str, Any]] = {}
    for cls, (low, high) in UNIT_EST_H.items():
        n = sum(1 for u in units if unit_class(u) == cls)
        wall_h = unit_wall_h(cls)
        billed_high = min(high, wall_h)
        classes[cls] = {
            "n_units": n,
            "est_low_h": low,
            "est_high_h": high,
            "cap_h": UNIT_TIME_CAP_H.get(cls),
            "billed_high_h": billed_high,
            "time_limit": _hms(wall_h),
            "subtotal_low_h": round(n * low, 2),
            "subtotal_high_h": round(n * billed_high, 2),
        }
    total_low = round(sum(c["subtotal_low_h"] for c in classes.values()), 2)
    total_high = round(sum(c["subtotal_high_h"] for c in classes.values()), 2)
    return {
        "ceiling_h": ceiling_h,
        "n_units": len(units),
        "classes": classes,
        "total_low_h": total_low,
        "total_high_h": total_high,
        "max_time_limit": max(c["time_limit"] for c in classes.values()),
        "fits_low": total_low <= ceiling_h,
        "fits_high": total_high <= ceiling_h,
    }


def assert_budget(cfg: dict[str, Any]) -> dict[str, Any]:
    """Hard-fail a grid whose best case already busts the ceiling; report a bad worst
    case rather than hiding it. Never rescales the estimates to make the check pass."""
    table = budget_table(cfg)
    if not table["fits_low"]:
        raise AssertionError(
            f"grid busts the {table['ceiling_h']:.0f} H200-h ceiling even at the LOW "
            f"estimate: {table['total_low_h']} h over {table['n_units']} units. "
            f"Cut seeds, pools, or detector runs -- the ceiling does not move "
            f"(CLAUDE.md: thresholds never move to make a gate pass)."
        )
    if not table["fits_high"]:
        logger.warning(
            "BUDGET RISK: low estimate %.1f h fits the %.0f h ceiling, but the HIGH "
            "estimate is %.1f h (over by %.1f h). Driver: detector runs at the top of "
            "their 1.5-3 h/run spread. Per-job wall limits still cap each unit; the "
            "grid total is the operator's call.",
            table["total_low_h"], table["ceiling_h"], table["total_high_h"],
            table["total_high_h"] - table["ceiling_h"],
        )
    return table


def posix_relpath(path: Path | str, root: Path = ROOT) -> str:
    """D43-delta(c): POSIX-style path, relative to `root` when possible. The SLURM
    (Linux) target must never see a Windows-separator path or a Windows absolute
    path emitted from a Windows dev machine -- render_sbatch uses this for every
    path it writes into the emitted script. Falls back to an as-posix() absolute
    path when `path` isn't inside `root` (e.g. a pytest tmp_path fixture)."""
    p = Path(path)
    try:
        rel = p.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except ValueError:
        return p.as_posix()


def entrypoint_cmd(config_path: Path | str, unit: str, device: str | None = None) -> list[str]:
    """The IDENTICAL command shape run locally and on SLURM (hard design constraint):
    same script, same flag order; only the --unit (and --device) VALUES vary."""
    cmd = ["scripts/sweep_unit.py", "--config", str(config_path), "--unit", unit]
    if device is not None:
        cmd += ["--device", device]
    return cmd


def select_gate_by_mean(
    assoc_by_gate: dict[float, list[float]],
) -> tuple[float, float, float]:
    """D43 amendment 2: ONE gate per arm, chosen by MEAN assoc across seeds -- never
    per-seed. Returns (selected_gate, mean_assoc, spread=max-min across seeds)."""
    assert assoc_by_gate, "no gates to select from"
    means = {g: mean(vals) for g, vals in assoc_by_gate.items() if vals}
    assert means, "no gate has any recorded assoc values"
    best_gate = max(means, key=lambda g: means[g])
    vals = assoc_by_gate[best_gate]
    spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
    return best_gate, means[best_gate], spread


def load_module(name: str, path: Path) -> ModuleType:
    """Load a scripts/*.py file as a module without running its __main__ block (the
    scaling_study.py pattern) -- guarantees identity/eval logic is reused verbatim."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load spec for {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_subprocess(python_exe: str, cmd: list[str], log_path: Path, cwd: Path = ROOT) -> None:
    """Run one subprocess, streaming output to `logger` and appending to `log_path`.
    Fail-fast: a non-zero return code raises immediately (no silent swallowing)."""
    full_cmd = [python_exe, *cmd]
    logger.info("$ %s", " ".join(full_cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n=== {' '.join(full_cmd)} ===\n")
        with subprocess.Popen(
            full_cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            # never trust the platform codec: ultralytics progress bars emit bytes
            # that crash cp1250 decoding on Windows (dry-run incident 2026-07-24)
            encoding="utf-8", errors="replace",
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                logger.info("  | %s", line.rstrip("\n"))
                lf.write(line if line.endswith("\n") else line + "\n")
            returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"subprocess failed (rc={returncode}): {' '.join(full_cmd)}")
