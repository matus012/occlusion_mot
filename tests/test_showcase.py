"""Synthetic-data tests for scripts/showcase.py's pure helpers (D42 wild-clip
showcase). No model weights, video files, or CUDA needed -- these tests never call
detect_frame/embed_crops/process_clip/main (those need torch + ultralytics + real
clips and are exercised manually via --help / a real batch run, per the D42 contract)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "showcase", ROOT / "scripts" / "showcase.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load_module()


# ---------------------------------------------------------------------------------
# stable_color
# ---------------------------------------------------------------------------------
def test_stable_color_deterministic_same_id() -> None:
    assert sc.stable_color(7) == sc.stable_color(7)
    assert sc.stable_color(123) == sc.stable_color(123)


def test_stable_color_distinct_across_most_ids() -> None:
    colors = {sc.stable_color(i) for i in range(1, 20)}
    assert len(colors) >= 15  # golden-angle-ish spread; a handful of collisions is fine


def test_stable_color_returns_valid_bgr_tuple() -> None:
    color = sc.stable_color(42)
    assert len(color) == 3
    assert all(0 <= c <= 255 for c in color)
    assert all(isinstance(c, int) for c in color)


# ---------------------------------------------------------------------------------
# output naming
# ---------------------------------------------------------------------------------
def test_output_name_includes_ckpt_label() -> None:
    assert sc.output_name(Path("clip1.mp4"), "myckpt") == "clip1__myckpt.mp4"


def test_output_name_strips_source_extension_only() -> None:
    name = sc.output_name(Path("some/dir/street.walk.mp4"), "conv45")
    assert name == "street.walk__conv45.mp4"


def test_contact_sheet_name_includes_ckpt_label() -> None:
    assert sc.contact_sheet_name("myckpt") == "contact_sheet__myckpt.png"


# ---------------------------------------------------------------------------------
# select_best_frame_index
# ---------------------------------------------------------------------------------
def test_select_best_frame_index_picks_max_count() -> None:
    assert sc.select_best_frame_index([0, 2, 5, 1]) == 2


def test_select_best_frame_index_first_max_on_tie() -> None:
    assert sc.select_best_frame_index([0, 3, 3, 1]) == 1


def test_select_best_frame_index_falls_back_to_mid_when_all_zero() -> None:
    counts = [0, 0, 0, 0, 0]
    assert sc.select_best_frame_index(counts) == len(counts) // 2


def test_select_best_frame_index_requires_nonempty() -> None:
    with pytest.raises(AssertionError):
        sc.select_best_frame_index([])


# ---------------------------------------------------------------------------------
# contact sheet assembly
# ---------------------------------------------------------------------------------
def _fake_frame(w: int, h: int, value: int) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_make_thumbnail_matches_target_width_and_aspect() -> None:
    frame = _fake_frame(960, 540, 100)
    thumb = sc.make_thumbnail(frame, width=480)
    assert thumb.shape[1] == 480
    assert thumb.shape[0] == 270  # aspect preserved (540/960 * 480)


def test_caption_cell_stacks_bar_below_thumbnail() -> None:
    thumb = _fake_frame(100, 50, 10)
    cell = sc.caption_cell(thumb, "clipA (3.2s)")
    assert cell.shape[1] == 100
    assert cell.shape[0] == 50 + 24  # thumb + caption bar


def test_assemble_contact_sheet_two_column_grid_shape() -> None:
    cells = [_fake_frame(100, 80, v) for v in (10, 20, 30)]  # 3 cells -> 2 rows x 2 cols
    grid = sc.assemble_contact_sheet(cells, cols=2)
    assert grid.shape == (160, 200, 3)  # 2 rows * 80h, 2 cols * 100w


def test_assemble_contact_sheet_pads_uneven_cell_sizes() -> None:
    cells = [_fake_frame(100, 80, 1), _fake_frame(90, 70, 2)]
    grid = sc.assemble_contact_sheet(cells, cols=2)
    assert grid.shape == (80, 200, 3)  # max cell dims used, smaller cell zero-padded


def test_build_contact_sheet_from_fake_frames_and_captions() -> None:
    entries = [
        ("street_walk", _fake_frame(640, 360, 50), 8.4),
        ("plaza_crowd", _fake_frame(640, 360, 90), 11.9),
        ("alley_cross", _fake_frame(640, 360, 30), 5.0),
    ]
    sheet = sc.build_contact_sheet(entries, cols=2)
    thumb_h = 360 * (480 / 640)  # 270
    cell_h = int(thumb_h) + 24
    assert sheet.shape[1] == 480 * 2
    assert sheet.shape[0] == cell_h * 2  # 3 entries -> 2 rows, second row padded


# ---------------------------------------------------------------------------------
# draw_overlay (pure image ops, no model)
# ---------------------------------------------------------------------------------
def test_draw_overlay_active_and_coasting_boxes_no_crash() -> None:
    frame = _fake_frame(200, 150, 5)
    active = np.array([[10, 10, 40, 60, 0.9, 1]], dtype=np.float64)
    coasting = np.array([[80, 20, 30, 50, 0.0, 2]], dtype=np.float64)
    out = sc.draw_overlay(frame, active, coasting, "myckpt")
    assert out.shape == (frame.shape[0] + sc.LEGEND_H, frame.shape[1], 3)
    assert out.dtype == np.uint8


def test_draw_overlay_empty_tracks_no_crash() -> None:
    frame = _fake_frame(120, 90, 5)
    out = sc.draw_overlay(frame, np.zeros((0, 6)), np.zeros((0, 6)), "myckpt")
    assert out.shape == (frame.shape[0] + sc.LEGEND_H, frame.shape[1], 3)


# ---------------------------------------------------------------------------------
# config / device resolution
# ---------------------------------------------------------------------------------
def test_build_hidden_config_matches_conv_app45_operating_point() -> None:
    cfg = sc.build_hidden_config()
    assert cfg.occl_buffer == 90
    assert cfg.vel_damping == 1.0
    assert cfg.recover_gate == 1.5
    assert cfg.occl_overlap_thresh == 0.25
    assert cfg.lowconf_mode == "kf"
    assert cfg.lowconf_noise_scale == 1.0
    assert cfg.app_gate_lost == 0.45
    assert cfg.app_gate_recover == 0.45


def test_resolve_device_explicit_cpu() -> None:
    assert sc.resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_cpu_or_cuda() -> None:
    assert sc.resolve_device("auto") in ("cpu", "cuda")


def test_resolve_device_no_hardcoded_cuda_zero() -> None:
    assert sc.resolve_device("cuda:1") == "cuda:1"  # passes through untouched


# ---------------------------------------------------------------------------------
# CLI defaults
# ---------------------------------------------------------------------------------
def test_parse_args_defaults() -> None:
    args = sc.parse_args(["--src", "showcase/sources", "--out", "showcase/renders"])
    assert args.max_seconds == 12.0
    assert args.stride == 1
    assert args.device == "auto"
    assert args.ckpt_label == "placeholder-ckpt"
    assert args.seed == 0
    assert args.weights == sc.DEFAULT_WEIGHTS


def test_parse_args_overrides() -> None:
    args = sc.parse_args([
        "--src", "a", "--out", "b", "--max-seconds", "5", "--stride", "3",
        "--device", "cpu", "--ckpt-label", "custom", "--seed", "7",
    ])
    assert args.max_seconds == 5.0
    assert args.stride == 3
    assert args.device == "cpu"
    assert args.ckpt_label == "custom"
    assert args.seed == 7
