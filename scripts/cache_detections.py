"""Phase 2: cache yolo11x person detections for every MOT17 train sequence (run ONCE).

The cache is the fixed-detections source of truth (context.md D1). ~8GB VRAM is plenty
for batch-1 inference. Usage:
  .venv/Scripts/python.exe scripts/cache_detections.py [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import load_split  # noqa: E402
from omot.detect.cache import cache_detections  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("cache_detections")

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT17")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "detections")
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seqs = load_split(args.data_root, "train", detector="FRCNN")
    for seq in seqs:
        cache_detections(
            seq, args.cache_dir, model_name=args.model, device=args.device, seed=args.seed
        )
    logger.info("all %d sequences cached", len(seqs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
