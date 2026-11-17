"""D37: download MOT20 train (mirror, D13 protocol) + Market-1501 zip from HF.

Official motchallenge.net attempted first 2026-07-23 (curl exit 28, site down —
same outage pattern as D13). Mirror fallback per the D13 user directive: free,
reputable, pinned revision, structural verification afterward, provenance logged
in context.md. License note: MOT20 is CC BY-NC-SA 3.0; Market-1501 research-only —
nothing dataset-derived is ever committed (D37 guard).

Pinned provenance:
  MOT20     hf:Lekim89/MOT20 @ 5fcaa0ef66906b63195a44672ec260d033545ef4
            (same mirror author as the D13 MOT17 cross-check mirror Lekim89/MOT17)
  Market    hf:aveocr/Market-1501-v15.09.15.zip @ b5e654a472251f193f78b6ad0af73f74e577a474
            file Market-1501-v15.09.15.zip (canonical v15.09.15 release zip)

Usage: .venv/Scripts/python.exe scripts/download_reid_sources.py [--skip-mot20] [--skip-market]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    stream=sys.stdout)
logger = logging.getLogger("download_reid_sources")

ROOT = Path(__file__).resolve().parents[1]

MOT20_REPO = "Lekim89/MOT20"
MOT20_REV = "5fcaa0ef66906b63195a44672ec260d033545ef4"
MARKET_REPO = "aveocr/Market-1501-v15.09.15.zip"
MARKET_REV = "b5e654a472251f193f78b6ad0af73f74e577a474"
MARKET_FILE = "Market-1501-v15.09.15.zip"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mot20", action="store_true")
    ap.add_argument("--skip-market", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    if not args.skip_mot20:
        dest = ROOT / "data" / "MOT20"
        logger.info("downloading MOT20 train split: %s @ %s -> %s",
                    MOT20_REPO, MOT20_REV[:8], dest)
        snapshot_download(
            repo_id=MOT20_REPO, repo_type="dataset", revision=MOT20_REV,
            allow_patterns=["train/**", "README.md"],
            local_dir=dest,
        )
        logger.info("MOT20 train done")

    if not args.skip_market:
        dest_dir = ROOT / "data" / "downloads"
        logger.info("downloading Market-1501 zip: %s @ %s -> %s",
                    MARKET_REPO, MARKET_REV[:8], dest_dir)
        path = hf_hub_download(
            repo_id=MARKET_REPO, repo_type="dataset", revision=MARKET_REV,
            filename=MARKET_FILE, local_dir=dest_dir,
        )
        logger.info("Market zip done: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
