"""Phase 5a: one-time GPU pass — appearance embeddings for every cached detection (D20).

Usage: .venv/Scripts/python.exe scripts/cache_embeddings.py [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import load_split  # noqa: E402
from omot.detect.cache import cache_path  # noqa: E402
from omot.detect.embed import cache_embeddings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("cache_embeddings")

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT17")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "detections")
    ap.add_argument("--model", default="yolo11x")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", type=Path, default=None,
                    help="train_reid.py checkpoint; default = ImageNet trunk")
    ap.add_argument("--tag", default="imagenet",
                    help="embedder tag baked into cache filenames")
    args = ap.parse_args()
    assert (args.weights is None) == (args.tag == "imagenet"), \
        "--weights requires a non-imagenet --tag (and vice versa)"

    seqs = load_split(args.data_root, "train", detector="FRCNN")
    for seq in seqs:
        det_cache = cache_path(args.cache_dir, seq.name, args.model)
        if not det_cache.exists():
            raise FileNotFoundError(f"detection cache missing: {det_cache}")
        cache_embeddings(seq, det_cache, args.cache_dir, device=args.device,
                         seed=args.seed, weights=args.weights, embedder_tag=args.tag)
    logger.info("all %d sequences embedded", len(seqs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
