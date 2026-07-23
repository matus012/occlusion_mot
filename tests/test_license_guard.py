"""D37 license-hygiene guard: no dataset-derived content may be TRACKED by git.

Binding rule (D37): crops, images, frame dumps, and serialized embeddings from any
dataset (MOT17/MOT20/Market-1501/CARLA renders) must never enter the repo or any
remote — only loaders, manifests, SHA256 records, and download docs are committable.

Convention for visuals: pure plots (matplotlib) may be tracked under viz/;
dataset-derived visuals (crop grids, annotated frames, clips) stay untracked
(gitignored patterns: *.mp4, viz/crops_*.png).

Runs under pytest, so scripts/check_gates.py G3 enforces it on every gate check.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_IMAGE_EXTS = (".jpg", ".jpeg", ".bmp", ".webp", ".ppm", ".tif", ".tiff")
BLOB_EXTS = (".pt", ".pth", ".npz", ".npy", ".onnx", ".engine", ".tar")


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True,
        timeout=60,
    )
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def test_no_tracked_files_under_data() -> None:
    offenders = [f for f in tracked_files() if f.startswith("data/")]
    assert not offenders, f"dataset-derived content tracked under data/: {offenders}"


def test_no_tracked_raw_images_anywhere() -> None:
    offenders = [f for f in tracked_files() if f.lower().endswith(RAW_IMAGE_EXTS)]
    assert not offenders, f"raw image files tracked (license risk): {offenders}"


def test_no_tracked_binary_blobs_outside_fixtures() -> None:
    offenders = [
        f for f in tracked_files()
        if f.lower().endswith(BLOB_EXTS) and not f.startswith("tests/fixtures/")
    ]
    assert not offenders, f"serialized weights/embeddings tracked: {offenders}"


def test_tracked_pngs_are_plots_only() -> None:
    """PNGs are committable only as plots under viz/, never as crop grids."""
    pngs = [f for f in tracked_files() if f.lower().endswith(".png")]
    outside_viz = [f for f in pngs if not f.startswith("viz/")]
    assert not outside_viz, f"tracked PNGs outside viz/: {outside_viz}"
    crop_named = [f for f in pngs if Path(f).name.startswith("crops_")]
    assert not crop_named, f"dataset-derived crop grids tracked: {crop_named}"
