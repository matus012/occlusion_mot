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
# D41 residual-surface fix: any serialized container that could hold dataset
# crops/pixels/embeddings — not just torch/numpy formats.
BLOB_EXTS = (".pt", ".pth", ".npz", ".npy", ".onnx", ".engine", ".tar",
             ".pkl", ".pickle", ".h5", ".hdf5", ".lmdb", ".mdb", ".arrow",
             ".parquet", ".feather", ".msgpack")
# Default-deny size ceiling: big binaries cannot slip in under an unlisted
# extension; anything over the ceiling must be an allowlisted visual.
SIZE_CEILING_BYTES = 2 * 1024 * 1024


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


# D41 (closes the D39 loophole): path prefixes cannot distinguish content classes —
# a MOT crop grid under demo/ would have passed the old rule. Every tracked visual
# must be EXPLICITLY classified here as "plot" (pure matplotlib, no dataset pixels)
# or "carla-render" (synthetic sim content). Images containing ANY MOT17/MOT20
# pixels (crop grids, annotated frames, clips) are the same class as demo clips:
# local-only, gitignored, never allowlisted. Adding a new visual to the repo
# requires adding it here — a deliberate human classification step.
VISUAL_EXTS = (".png", ".gif", ".mp4", ".avi", ".webm", ".svg")
ALLOWED_TRACKED_VISUALS: dict[str, str] = {
    "viz/summary_retention_grid.png": "plot",
    "viz/ablation_3arm_dev.png": "plot",
    "viz/detector_arm_dev.png": "plot",
    "viz/scaling_study.png": "plot",
    "viz/scaling_curve_local.png": "plot",
    "demo/s0_teaser.png": "carla-render",
    "demo/s5_blueprint_grid.png": "carla-render",
    "demo/s5_carla.mp4": "carla-render",
}


def test_tracked_visuals_are_allowlisted() -> None:
    """Every tracked image/video must be explicitly classified (D41)."""
    visuals = [f for f in tracked_files() if f.lower().endswith(VISUAL_EXTS)]
    unclassified = [f for f in visuals if f not in ALLOWED_TRACKED_VISUALS]
    assert not unclassified, (
        f"tracked visuals not in the D41 allowlist (classify as 'plot' or "
        f"'carla-render' in tests/test_license_guard.py, or keep them local): "
        f"{unclassified}"
    )


def test_no_large_tracked_files_outside_visual_allowlist() -> None:
    """Default-deny for big binaries: a crop cache renamed to an unlisted extension
    still fails here (D41 residual surface)."""
    offenders = []
    for f in tracked_files():
        p = ROOT / f
        if p.exists() and p.stat().st_size > SIZE_CEILING_BYTES:
            if f not in ALLOWED_TRACKED_VISUALS:
                offenders.append(f"{f} ({p.stat().st_size / 1e6:.1f} MB)")
    assert not offenders, f"large tracked files outside the visual allowlist: {offenders}"


def test_allowlist_itself_is_clean() -> None:
    """The allowlist may never contain dataset-pixel content by construction."""
    for path, cls in ALLOWED_TRACKED_VISUALS.items():
        assert cls in ("plot", "carla-render"), f"{path}: unknown class {cls!r}"
        assert not Path(path).name.startswith("crops_"), (
            f"{path}: crop grids are dataset-derived, never allowlistable"
        )
