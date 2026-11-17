"""Mirror download of MOT17 train split from Hugging Face (user directive 2026-07-22:
free mirror allowed after official-site poller window; verify + log source & hashes).

Downloads train/* only (~2.7GB; test split not needed for the train half-split protocol).
Prints the pinned revision for provenance; run scripts/verify_mot17.py afterwards.
Usage: .venv/Scripts/python.exe scripts/download_mot17_hf.py [--repo ling1016/MOT17]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("download_mot17_hf")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "ling1016/MOT17"
FALLBACK_REPO = "Morrison1025/MOT17"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--dest", type=Path, default=ROOT / "data" / "MOT17")
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    info = HfApi().dataset_info(args.repo)
    revision = info.sha
    log.info("downloading %s @ %s -> %s (train/* only, %d workers)",
             args.repo, revision, args.dest, args.max_workers)
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["train/*"],
        local_dir=args.dest,
        max_workers=args.max_workers,
    )
    print(f"MIRROR_OK repo={args.repo} revision={revision}")
    print(f"NEXT: verify with scripts/verify_mot17.py --source hf:{args.repo}@{revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
