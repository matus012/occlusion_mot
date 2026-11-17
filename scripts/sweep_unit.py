"""D43: PERUN sweep entrypoint -- THE identical unit executor for local dry-run and
PERUN SLURM array jobs alike (hard design constraint: same script, different
config/env; no code fork). One invocation = one unit end-to-end via subprocesses of
the already-frozen scripts (train_reid.py, cache_embeddings.py, run_hidden.py,
finetune_detector.py) -- no logic duplication. FIXED DETECTIONS invariant intact:
only the embedder tag changes across units; the yolo11x detection cache never does.

Usage:
  .venv/Scripts/python.exe scripts/sweep_unit.py --config configs/sweep/dryrun_local.yaml \
      --unit D:300:0 [--device cuda]
  .venv/Scripts/python.exe scripts/sweep_unit.py --config configs/sweep/dryrun_local.yaml \
      --unit detector:yolo11s:mot17dev [--device cuda]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROOT / "src"))

from sweep_common import (  # noqa: E402
    DEV_OPTIMISTIC_LABEL,
    GATE_CONFIG,
    RESULTS_ROOT,
    arm_sources,
    config_digest,
    detector_result_path,
    embedder_result_path,
    embedder_unit_id,
    load_config,
    load_module,
    parse_unit,
    pool_to_frac,
    run_subprocess,
    run_tag,
    token_to_pool,
)

from omot.detect.cache import select_device  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    stream=sys.stdout)
logger = logging.getLogger("sweep_unit")


def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_reid_cmd(
    cfg: dict[str, Any], arm: str, pool: int | None, seed: int, device: str | None
) -> list[str]:
    sources = cfg["arms"][arm]["sources"]
    assert sources is not None, f"arm {arm} has no training sources"
    frac = pool_to_frac(arm, pool, sources)
    cmd = [str(SCRIPTS_DIR / "train_reid.py"),
           "--sources", *sources,
           "--epochs", str(cfg["epochs"]),
           "--batches-per-epoch", str(cfg["batches_per_epoch"]),
           "--identity-frac", f"{frac:.6f}",
           "--identity-seed", str(seed),
           "--seed", str(seed),
           "--eval-every", str(cfg.get("eval_every", 1)),
           "--tag", run_tag(cfg["name"], arm, pool, seed)]
    if device:
        cmd += ["--device", device]
    return cmd


def cache_embeddings_cmd(tag: str, weights: Path | None, device: str | None) -> list[str]:
    cmd = [str(SCRIPTS_DIR / "cache_embeddings.py"), "--tag", tag]
    if weights is not None:
        cmd += ["--weights", str(weights)]
    if device:
        cmd += ["--device", device]
    return cmd


def run_hidden_cmd(
    cfg: dict[str, Any], arm: str, pool: int | None, seed: int, gate: float, embedder_tag: str
) -> list[str]:
    cmd = [str(SCRIPTS_DIR / "run_hidden.py"), "--half", cfg["half"]]
    for flag, val in GATE_CONFIG.items():
        cmd += [f"--{flag}", val]
    cmd += ["--app-gate-lost", str(gate), "--app-gate-recover", str(gate),
            "--embedder-tag", embedder_tag,
            "--tag", run_tag(cfg["name"], arm, pool, seed, gate),
            "--seed", str(seed), "--skip-trackeval"]
    return cmd


def finetune_detector_cmd(
    cfg: dict[str, Any], model: str, mix: str, tag: str, device: str | None, seed: int
) -> list[str]:
    det = cfg["detector"]
    base_weights = det.get("base_weights") or f"{model}.pt"
    cmd = [str(SCRIPTS_DIR / "finetune_detector.py"),
           "--epochs", str(det["epochs"]),
           "--imgsz", str(det["imgsz"]),
           "--batch", str(det["batch"]),
           "--seed", str(seed),
           "--tag", tag,
           "--mix", mix,
           "--base-weights", base_weights]
    if device:
        cmd += ["--device", device]
    return cmd


def parse_yolo_results_csv(path: Path) -> dict[str, float]:
    """Final-epoch metrics from an ultralytics results.csv (read back from disk --
    no ultralytics import needed, keeps this module GPU/model-free at import time)."""
    assert path.exists(), f"ultralytics results.csv missing: {path} (did training run?)"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, f"{path}: no rows written"
    last = rows[-1]
    metrics: dict[str, float] = {}
    for key, val in last.items():
        try:
            metrics[key.strip()] = float(val)
        except (TypeError, ValueError):
            continue
    return metrics


def imagenet_occ_metrics(sources: list[str], device: str) -> dict[str, float]:
    """Arm A occ-metrics: ImageNet trunk, NO training, eval-only path -- reuses
    train_reid.py's evaluate()/ReidDataset internals via importlib (no duplication
    of the retrieval-eval code, same pattern scaling_study.py uses)."""
    import torch

    from omot.detect.embed import _build_embedder

    train_reid_mod = load_module("train_reid_sweep", SCRIPTS_DIR / "train_reid.py")
    _, val_items = train_reid_mod.load_items(sources)
    val_ds = train_reid_mod.ReidDataset(val_items, training=False)
    trunk, _normalize = _build_embedder(device, weights=None)  # 512-d ImageNet trunk

    class _TrunkOnly(torch.nn.Module):
        def __init__(self, net: torch.nn.Module) -> None:
            super().__init__()
            self.trunk = net

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            return self.trunk(x), None

    model = _TrunkOnly(trunk).to(device)
    original_dim = train_reid_mod.EMBED_DIM
    train_reid_mod.EMBED_DIM = 512  # trunk output width; restored below regardless of outcome
    try:
        metrics = train_reid_mod.evaluate(model, val_ds, device)
    finally:
        train_reid_mod.EMBED_DIM = original_dim
    return metrics


def run_embedder_unit(
    cfg: dict[str, Any], arm: str, pool: int | None, seed: int, config_path: Path, device: str
) -> dict[str, Any]:
    name = cfg["name"]
    python_exe = sys.executable
    log_path = RESULTS_ROOT / name / "sweep.log"
    runtimes: dict[str, float] = {"train_s": 0.0, "embed_s": 0.0, "track_s": 0.0}

    if arm == "A":
        eval_sources = arm_sources(cfg, arm)
        logger.info("arm A (ImageNet null): no training; pool=%s ignored", pool)
        embedder_tag = "imagenet"
        weights: Path | None = None
        n_train_ids: int | None = None
        t0 = time.time()
        occ_metrics = imagenet_occ_metrics(eval_sources, device)
        runtimes["train_s"] = time.time() - t0  # eval-only pass, logged under the same key
    else:
        tag = run_tag(name, arm, pool, seed)
        ckpt = ROOT / "data" / "models" / f"reid_{tag}.pt"
        t0 = time.time()
        if ckpt.exists():
            logger.info("checkpoint exists, skipping training: %s", ckpt)
        else:
            run_subprocess(python_exe, train_reid_cmd(cfg, arm, pool, seed, device), log_path)
        assert ckpt.exists(), f"train_reid.py did not produce {ckpt}"
        runtimes["train_s"] = time.time() - t0
        embedder_tag = tag
        weights = ckpt

        import torch
        ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
        occ_metrics = dict(ckpt_data["metrics"])
        n_train_ids = int(ckpt_data.get("n_train_ids", 0))

    t0 = time.time()
    run_subprocess(python_exe, cache_embeddings_cmd(embedder_tag, weights, device), log_path)
    runtimes["embed_s"] = time.time() - t0

    t0 = time.time()
    per_gate: dict[str, Any] = {}
    for gate in cfg["gate_probe"]:
        run_subprocess(
            python_exe, run_hidden_cmd(cfg, arm, pool, seed, gate, embedder_tag), log_path
        )
        hidden_json = (
            ROOT / "results" / f"hidden_{cfg['half']}_{run_tag(name, arm, pool, seed, gate)}.json"
        )
        assert hidden_json.exists(), f"run_hidden.py did not produce {hidden_json}"
        per_gate[str(gate)] = json.loads(hidden_json.read_text(encoding="utf-8"))["g2"]
    runtimes["track_s"] = time.time() - t0

    result: dict[str, Any] = {
        "unit": embedder_unit_id(arm, pool, seed),
        "arm": arm,
        "pool": pool,
        "seed": seed,
        "config_digest": config_digest(config_path),
        "n_train_ids": n_train_ids,
        "occ_metrics": occ_metrics,
        "per_gate": per_gate,
        "runtimes": runtimes,
    }
    dest = embedder_result_path(RESULTS_ROOT, name, arm, pool, seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("unit %s -> %s", result["unit"], dest)
    return result


def run_detector_unit(
    cfg: dict[str, Any], model: str, mix: str, config_path: Path, device: str, seed: int = 0
) -> dict[str, Any]:
    det = cfg.get("detector")
    assert det is not None and det.get("enabled"), "detector block missing/disabled in config"
    if mix not in det["mixes"]:
        raise ValueError(f"mix '{mix}' not declared in config.detector.mixes={det['mixes']}")
    name = cfg["name"]
    tag = f"sweep_{name}_detector_{model}_{mix}"
    log_path = RESULTS_ROOT / name / "sweep.log"

    t0 = time.time()
    run_subprocess(
        sys.executable, finetune_detector_cmd(cfg, model, mix, tag, device, seed), log_path
    )
    train_s = time.time() - t0

    project = ROOT / "data" / "models" / "det_finetune"
    metrics = parse_yolo_results_csv(project / tag / "results.csv")

    result: dict[str, Any] = {
        "unit": f"detector:{model}:{mix}",
        "model": model,
        "mix": mix,
        "config_digest": config_digest(config_path),
        "metrics": metrics,
        "runtimes": {"train_s": train_s},
        "label": "dev-optimistic",
        "note": DEV_OPTIMISTIC_LABEL,
    }
    dest = detector_result_path(RESULTS_ROOT, name, model, mix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("dev-optimistic detector unit %s:%s -> %s", model, mix, dest)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--unit", required=True,
                    help="'<arm>:<pool|full>:<seed>' or 'detector:<model>:<mix>'")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto, CPU fallback)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = select_device(args.device)
    parsed = parse_unit(args.unit)

    if parsed[0] == "detector":
        _, model, mix = parsed
        set_seeds(0)
        run_detector_unit(cfg, model, mix, args.config, device)
        return 0

    _, arm, pool_tok, seed_tok = parsed
    pool = token_to_pool(pool_tok)
    seed = int(seed_tok)
    set_seeds(seed)
    run_embedder_unit(cfg, arm, pool, seed, args.config, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
