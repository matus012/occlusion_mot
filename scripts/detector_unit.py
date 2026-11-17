"""Run ONE Stage-1 detector unit end-to-end (perun_detector_v1.md section 3, D56).

A unit is (model x mix x seed). It produces one dose-response point:

  1. finetune the detector on a MOT17-DISJOINT mix (mot20 / mot20_carla)
  2. cache its detections over MOT17 (the evaluation set it has never seen)
  3. cache conv embeddings for those detections
  4. run the FROZEN canonical tracker on the dev half
  5. write {mAP50-95, oracle_ceiling, id_retention, ...} as the unit's result JSON

Steps 2-5 are what make this a dose-response point rather than just a detector: the
regression in section 3 needs (detector quality, ceiling, e2e) per unit.

Identical entrypoint local and on SLURM:
  python scripts/detector_unit.py --config configs/sweep/perun_detector.yaml \
      --unit yolo11s:mot20:0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_common import run_subprocess  # noqa: E402
from sweep_unit import parse_yolo_results_csv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("detector_unit")

ROOT = Path(__file__).resolve().parents[1]

# The tracker is FROZEN across every unit (perun_detector_v1.md section 1). Only the
# detection source varies. These are the D28-final canonical settings, byte-for-byte the
# same list used by val_manifest.md R1 -- never edit them to chase a number.
FROZEN_TRACKER = [
    "--occl-buffer", "90", "--damping", "1.0", "--recover-gate", "1.5",
    "--overlap-thresh", "0.25", "--noise-scale", "1.0", "--lowconf-mode", "kf",
    "--app-gate-lost", "0.45", "--app-gate-recover", "0.45",
    "--embedder-tag", "conv",
]
EMBEDDER_CKPT = ROOT / "data" / "models" / "reid_conv.pt"


def load_config(path: Path) -> dict[str, Any]:
    import yaml
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("name", "models", "mixes", "seeds", "epochs", "imgsz", "batch"):
        assert key in cfg, f"{path}: missing required key {key!r}"
    return cfg


def enumerate_units(cfg: dict[str, Any]) -> list[str]:
    """Deterministic order -- the array index maps to this list and must never drift."""
    return [
        f"{model}:{mix}:{seed}"
        for model in cfg["models"]
        for mix in cfg["mixes"]
        for seed in cfg["seeds"]
    ]


def parse_unit(unit: str) -> tuple[str, str, int]:
    model, mix, seed = unit.split(":")
    return model, mix, int(seed)


def unit_tag(cfg: dict[str, Any], model: str, mix: str, seed: int) -> str:
    return f"det_{cfg['name']}_{model}_{mix}_s{seed}"


def result_path(results_dir: Path, name: str, model: str, mix: str, seed: int) -> Path:
    return results_dir / name / f"{model}_{mix}_s{seed}.json"


def run_unit(cfg: dict[str, Any], unit: str, config_path: Path, device: str,
             results_dir: Path) -> dict[str, Any]:
    model, mix, seed = parse_unit(unit)
    assert model in cfg["models"], f"model {model!r} not in config"
    assert mix in cfg["mixes"], f"mix {mix!r} not in config"
    assert seed in cfg["seeds"], f"seed {seed!r} not in config"

    name = cfg["name"]
    tag = unit_tag(cfg, model, mix, seed)
    log_path = results_dir / name / "detector.log"
    py = sys.executable
    runtimes: dict[str, float] = {}

    epochs = cfg["epochs"][mix] if isinstance(cfg["epochs"], dict) else cfg["epochs"]

    # ---- 1. finetune on the MOT17-disjoint mix -----------------------------------
    project = ROOT / "data" / "models" / "det_finetune"
    best = project / tag / "weights" / "best.pt"
    t0 = time.time()
    if best.exists():
        logger.info("checkpoint exists, skipping training: %s", best)
    else:
        run_subprocess(py, [
            "scripts/finetune_detector.py", "--mix", mix, "--epochs", str(epochs),
            "--imgsz", str(cfg["imgsz"]), "--batch", str(cfg["batch"]),
            "--base-weights", f"{model}.pt", "--tag", tag, "--seed", str(seed),
            "--device", device,
        ], log_path)
    assert best.exists(), f"finetune_detector.py did not produce {best}"
    runtimes["train_s"] = time.time() - t0

    det_metrics = parse_yolo_results_csv(project / tag / "results.csv")

    # ---- 2. cache detections over MOT17 (never seen in training) ------------------
    t0 = time.time()
    run_subprocess(py, [
        "scripts/cache_detections.py", "--model", tag, "--weights", str(best),
        "--device", device, "--seed", str(seed),
    ], log_path)
    runtimes["detect_s"] = time.time() - t0

    # ---- 3. conv embeddings for those detections ---------------------------------
    t0 = time.time()
    run_subprocess(py, [
        "scripts/cache_embeddings.py", "--model", tag, "--tag", "conv",
        "--weights", str(EMBEDDER_CKPT), "--device", device, "--seed", str(seed),
    ], log_path)
    runtimes["embed_s"] = time.time() - t0

    # ---- 4. frozen canonical tracker on the dev half ------------------------------
    t0 = time.time()
    run_subprocess(py, [
        "scripts/run_hidden.py", "--half", "dev", *FROZEN_TRACKER,
        "--model", tag, "--tag", tag, "--seed", str(seed), "--skip-trackeval",
    ], log_path)
    hidden_json = ROOT / "results" / f"hidden_dev_{tag}.json"
    assert hidden_json.exists(), f"run_hidden.py did not produce {hidden_json}"
    g2 = json.loads(hidden_json.read_text(encoding="utf-8"))["g2"]
    runtimes["track_s"] = time.time() - t0

    result: dict[str, Any] = {
        "unit": unit,
        "model": model,
        "mix": mix,
        "seed": seed,
        "epochs": epochs,
        "tag": tag,
        "det_metrics": det_metrics,
        "g2": g2,
        # the two dose-response coordinates, hoisted so the analysis never re-derives them
        "map50_95": det_metrics.get("metrics/mAP50-95(B)"),
        "oracle_ceiling": g2["oracle_ceiling"],
        "id_retention": g2["id_retention"],
        "id_retention_assoc": g2["id_retention_assoc"],
        "runtimes": runtimes,
        "label": "MOT17-disjoint training (perun_detector_v1.md s1) -- NOT dev-optimistic",
    }
    dest = result_path(results_dir, name, model, mix, seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("unit %s -> %s", unit, dest)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--unit", required=True, help="<model>:<mix>:<seed>")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "sweep")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run_unit(cfg, args.unit, args.config, device, args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
