"""Phase 6 / D29: embedder identity-scaling study.

Training-identity count (currently 305, sources mot17_dev+sim) is the suspected binding
constraint on re-ID quality (D29 L3: convergence saturates ~0.60 assoc, refuting the
"more epochs" hypothesis). This runner sweeps --identity-frac through train_reid.py to
measure assoc-retention (G2a) and occ_rank1 vs. training-identity count, to size/justify
the PERUN full-scale run.

FIXED DETECTIONS: the detection cache tag stays the default (yolo11x) for every fraction
-- only the embedder changes. run_hidden.py is called with --skip-trackeval (G2-only;
this is an embedder study, not a G1 tracker comparison).

Per fraction f, orchestrates three subprocesses (same .venv interpreter as this process):
  1. train_reid.py   --identity-frac f --identity-seed <seed> --tag scale{pct}
                      (skipped if data/models/reid_scale{pct}.pt exists, unless --force)
  2. cache_embeddings.py --weights <ckpt> --tag scale{pct}
  3. run_hidden.py   fixed dev-half gate config, --embedder-tag scale{pct} --skip-trackeval
Then collects id_retention_assoc / n_assoc_scope / id_retention / center_err from
results/hidden_dev_scale{pct}.json and occ_rank1 / n_train_ids from the checkpoint,
appending to results/scaling_study.json (written after EVERY fraction -- crash-safe
resume: a fraction already present in that file is skipped unless --force). Finally
renders viz/scaling_study.png.

Usage:
  .venv/Scripts/python.exe scripts/scaling_study.py --fractions 0.25 0.5 0.75 1.0 \
      [--sources mot17_dev sim] [--epochs 40] [--seed 0] [--device cuda] [--force]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    stream=sys.stdout)
logger = logging.getLogger("scaling_study")

STUDY_JSON = ROOT / "results" / "scaling_study.json"
STUDY_LOG = ROOT / "results" / "scaling_study.log"
VIZ_PATH = ROOT / "viz" / "scaling_study.png"

# Fixed dev-half tracker gate config (D26 best-known dev config, D29 §L3/L4).
GATE_CONFIG: dict[str, str] = {
    "half": "dev", "occl-buffer": "90", "damping": "1.0", "recover-gate": "1.5",
    "overlap-thresh": "0.25", "noise-scale": "1.0", "lowconf-mode": "kf",
    "app-gate-lost": "0.45", "app-gate-recover": "0.45",
}
IMAGENET_NULL = 0.556
G2A_FLOOR = 0.58


def _load_module(name: str, path: Path) -> ModuleType:
    """Load a scripts/*.py file as a module without running its __main__ block.

    Chosen over duplicating the subsample logic: this guarantees scaling_study.py's
    n_train_ids fallback path uses the EXACT SAME code (subsample_identities +
    load_items) train_reid.py used, with no drift risk. Same pattern already used by
    tests/test_finetune_detector.py for importing a sibling script under test.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load spec for {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pct_str(frac: float) -> str:
    pct = int(round(frac * 100))
    assert 0 < pct <= 100, f"invalid fraction {frac}"
    return str(pct)


def run_subprocess(python_exe: str, cmd: list[str], log_path: Path) -> None:
    """Run a subprocess, streaming its stdout/stderr to `logger` and to `log_path`.

    Fail fast: non-zero returncode raises immediately.
    """
    full_cmd = [python_exe, *cmd]
    logger.info("$ %s", " ".join(full_cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n=== {' '.join(full_cmd)} ===\n")
        with subprocess.Popen(
            full_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                logger.info("  | %s", line)
                lf.write(line + "\n")
            returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"subprocess failed (rc={returncode}): {' '.join(full_cmd)}")


def train_reid_cmd(sources: list[str], epochs: int, frac: float, identity_seed: int,
                   tag: str, device: str | None) -> list[str]:
    cmd = [str(SCRIPTS_DIR / "train_reid.py"),
           "--sources", *sources,
           "--epochs", str(epochs),
           "--identity-frac", str(frac),
           "--identity-seed", str(identity_seed),
           "--tag", tag]
    if device:
        cmd += ["--device", device]
    return cmd


def cache_embeddings_cmd(ckpt: Path, tag: str, device: str | None) -> list[str]:
    cmd = [str(SCRIPTS_DIR / "cache_embeddings.py"), "--weights", str(ckpt), "--tag", tag]
    if device:
        cmd += ["--device", device]
    return cmd


def run_hidden_cmd(tag: str) -> list[str]:
    cmd = [str(SCRIPTS_DIR / "run_hidden.py")]
    for k, v in GATE_CONFIG.items():
        cmd += [f"--{k}", v]
    cmd += ["--embedder-tag", tag, "--tag", tag, "--skip-trackeval"]
    return cmd


def n_train_ids_for(train_reid_mod: ModuleType, sources: list[str], frac: float,
                    seed: int) -> int:
    """Recompute the exact identity count train_reid.py would have trained on.

    Used only as a fallback when the checkpoint predates the `n_train_ids` field that
    train_reid.py now stores directly (the primary, authoritative source -- it records
    the count actually used for THAT run, avoiding any recompute drift).
    """
    train_items, _ = train_reid_mod.load_items(sources)
    if frac < 1.0:
        train_items = train_reid_mod.subsample_identities(train_items, sources, frac, seed)
    return len({rec["identity"] for rec in train_items})


def load_study(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"config": {}, "results": []}


def save_study(path: Path, study: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(study, indent=2), encoding="utf-8")


def render_plot(study: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted({r.get("identity_seed", 0) for r in study["results"]})
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_xlabel("n_train_ids")
    ax1.set_ylabel("id_retention_assoc (G2a)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("occ_rank1", color="tab:green")
    for seed in seeds:
        results = sorted(
            (r for r in study["results"] if r.get("identity_seed", 0) == seed),
            key=lambda r: r["n_train_ids"],
        )
        n_ids = [r["n_train_ids"] for r in results]
        ax1.plot(n_ids, [r["id_retention_assoc"] for r in results], "o-",
                 color="tab:blue", alpha=1.0 if seed == seeds[0] else 0.55,
                 label=f"id_retention_assoc (seed {seed})")
        ax2.plot(n_ids, [r["occ_rank1"] for r in results], "s--",
                 color="tab:green", alpha=1.0 if seed == seeds[0] else 0.55,
                 label=f"occ_rank1 (seed {seed})")
    ax1.axhline(IMAGENET_NULL, color="gray", linestyle="--", linewidth=1,
                label=f"ImageNet null ({IMAGENET_NULL})")
    ax1.axhline(G2A_FLOOR, color="tab:red", linestyle=":", linewidth=1,
                label=f"G2a floor ({G2A_FLOOR})")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    ax1.set_title("D29 identity-scaling study: assoc retention & occ_rank1 vs n_train_ids")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("-> %s", out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--sources", nargs="+", default=["mot17_dev", "sim"],
                    choices=["mot17_dev", "sim"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="passthrough to train_reid.py / "
                    "cache_embeddings.py subprocesses (None = let them auto-select)")
    ap.add_argument("--force", action="store_true",
                    help="re-run fractions even if already present / checkpointed")
    ap.add_argument("--out", type=Path, default=STUDY_JSON,
                    help="study JSON path; the plot lands at viz/<stem>.png. Use a "
                         "fresh path when changing epochs so curves never mix.")
    args = ap.parse_args()
    for f in args.fractions:
        assert 0.0 < f <= 1.0, f"--fractions entries must be in (0, 1], got {f}"

    random.seed(args.seed)
    np.random.seed(args.seed)

    python_exe = sys.executable
    train_reid_mod = _load_module("train_reid_ss", SCRIPTS_DIR / "train_reid.py")

    study = load_study(args.out)
    study["config"] = {
        "sources": args.sources, "epochs": args.epochs, "gate": GATE_CONFIG,
    }
    # runs are keyed by (fraction, identity_seed) so multi-seed curves accumulate
    done_runs = {(r["fraction"], r.get("identity_seed", 0)) for r in study["results"]}

    for frac in args.fractions:
        if (frac, args.seed) in done_runs and not args.force:
            logger.info("fraction %.3f seed %d already in %s, skipping (--force to re-run)",
                        frac, args.seed, args.out)
            continue

        pct = pct_str(frac)
        tag = f"scale{pct}s{args.seed}"
        ckpt = ROOT / "data" / "models" / f"reid_{tag}.pt"
        t0 = time.time()

        if ckpt.exists() and not args.force:
            logger.info("checkpoint exists, skipping training: %s", ckpt)
        else:
            run_subprocess(
                python_exe,
                train_reid_cmd(args.sources, args.epochs, frac, args.seed, tag, args.device),
                STUDY_LOG,
            )
        assert ckpt.exists(), f"train_reid.py did not produce {ckpt}"

        run_subprocess(python_exe, cache_embeddings_cmd(ckpt, tag, args.device), STUDY_LOG)
        run_subprocess(python_exe, run_hidden_cmd(tag), STUDY_LOG)

        hidden_json = ROOT / "results" / f"hidden_dev_{tag}.json"
        assert hidden_json.exists(), f"run_hidden.py did not produce {hidden_json}"
        g2 = json.loads(hidden_json.read_text(encoding="utf-8"))["g2"]

        import torch
        ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
        occ_rank1 = float(ckpt_data["metrics"]["occ_rank1"])
        n_train_ids = ckpt_data.get("n_train_ids")
        if n_train_ids is None:
            n_train_ids = n_train_ids_for(train_reid_mod, args.sources, frac, args.seed)
        n_train_ids = int(n_train_ids)

        entry = {
            "fraction": frac,
            "identity_seed": args.seed,
            "epochs": args.epochs,
            "n_train_ids": n_train_ids,
            "occ_rank1": occ_rank1,
            "id_retention_assoc": float(g2["id_retention_assoc"]),
            "n_assoc_scope": int(g2["n_assoc_scope"]),
            "id_retention": float(g2["id_retention"]),
            "center_err": float(g2["reemergence_center_err_med"]),
        }
        study["results"] = [
            r for r in study["results"]
            if not (r["fraction"] == frac and r.get("identity_seed", 0) == args.seed)
        ] + [entry]
        save_study(args.out, study)
        logger.info("fraction %.3f seed %d done in %.0fs: %s",
                    frac, args.seed, time.time() - t0, entry)

    render_plot(study, ROOT / "viz" / f"{args.out.stem}.png")
    logger.info("-> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
