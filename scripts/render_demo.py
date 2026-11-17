"""D39 demo package: guided-tour artifacts (clips, plots, README) for the whole project.

READ-ONLY over existing artifacts (tracker outputs, cached detections, CARLA renders,
results JSONs) — no tracking/training re-runs, no live detector inference (FIXED
DETECTIONS invariant, context.md D1). Reuses scripts/render_dev_viz.py's panel-drawing
machinery (draw_panel/load_tracker/outcome_for) via importlib, same pattern
tests/test_finetune_detector.py uses for sibling-script imports.

Sections (--only <name>, default runs all):
  segments  -- score + pick the dev-half segments used by the clips below (logs only)
  s1_vs_s2  -- demo/clips/s1_vs_s2.mp4   baseline vs geometric hidden-state
  s2_solo   -- demo/clips/s2_solo.mp4    geometric, single panel, coasting highlighted
  s3_vs_s4  -- demo/clips/s3_vs_s4.mp4   ImageNet appearance veto vs trained embedder
  s6        -- demo/clips/s6_detector.mp4  stock vs finetuned detector, dets only
  s5        -- demo/s5_carla.mp4 + demo/s5_blueprint_grid.png  CARLA feeder (committable)
  s0        -- demo/s0_teaser.png        CARLA occlusion teaser frame
  hero      -- demo/clips/hero.mp4       THE clip: baseline vs full stack, side by side
  readme    -- demo/README.md

Usage: .venv/Scripts/python.exe scripts/render_demo.py --all
       .venv/Scripts/python.exe scripts/render_demo.py --only s6
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omot.data.mot import load_sequence  # noqa: E402
from omot.detect.cache import cache_path, load_cached_detections  # noqa: E402
from omot.eval.occlusion import OcclusionSegment  # noqa: E402
from omot.io.mot_format import COL, read_mot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("render_demo")

DEMO_DIR = ROOT / "demo"
CLIPS_DIR = DEMO_DIR / "clips"
VIZ_DIR = ROOT / "viz"
FONT = cv2.FONT_HERSHEY_SIMPLEX
PAD_FRAMES = 30  # wider than render_dev_viz's 15 -- demo clips target 15-40s, not 6-10s
CARLA_SCENARIO = ROOT / "data" / "sim" / "carla_render" / "crowd_merge_0017"

# tag/label maps shared by segment scoring, clip rendering, and the README
TAGS = {
    "base": "hidden_audit_base",
    "geom": "hidden_audit_geom",
    "in45": "hidden_audit_in45",
    "conv": "hidden_conv_app45",
}
LABELS = {
    "base": "S1 ByteTrack baseline",
    "geom": "S2 + geometric hidden-state",
    "in45": "S3 + ImageNet appearance veto",
    "conv": "S4 + trained embedder",
}
SECTIONS = ["segments", "s1_vs_s2", "s2_solo", "s3_vs_s4", "s6", "s5", "s0", "hero", "readme"]


def set_seeds(seed: int) -> None:
    """No torch/GPU work happens in this script (read-only viz over cached artifacts,
    no model inference) -- seeding random/numpy covers every place a tie-break or crop
    pick could become sampling-based."""
    random.seed(seed)
    np.random.seed(seed)


def _load_render_dev_viz() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "render_dev_viz", ROOT / "scripts" / "render_dev_viz.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclasses' field-type resolution looks the module up via sys.modules by name --
    # register it before exec_module (render_dev_viz defines dataclasses at module level).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


dv = _load_render_dev_viz()


# ---------------------------------------------------------------------------------
# video/image helpers
# ---------------------------------------------------------------------------------
def _transcode(tmp: Path, dest: Path) -> None:
    """cv2's mp4v is unplayable in stock Windows players -- transcode to H.264."""
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
        check=True, timeout=600,
    )
    tmp.unlink()


def _write_video(frames: Iterable[np.ndarray], dest: Path, fps: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".raw.mp4")
    writer: cv2.VideoWriter | None = None
    n = 0
    for frame in frames:
        if writer is None:
            writer = cv2.VideoWriter(
                str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                (frame.shape[1], frame.shape[0]),
            )
        writer.write(frame)
        n += 1
    assert writer is not None and n > 0, f"no frames rendered for {dest}"
    writer.release()
    _transcode(tmp, dest)
    log.info("wrote %s (%d frames @ %dfps -> %.1fs)", dest, n, fps, n / fps)


LEGEND_H = 34
_LEGEND_ITEMS = [
    ((0, 220, 0), "active focal"),
    ((0, 165, 255), "coasting / hidden prediction"),
    ((60, 60, 255), "verdict fail"),
]


def _append_legend(canvas: np.ndarray) -> np.ndarray:
    bar = np.full((LEGEND_H, canvas.shape[1], 3), 20, np.uint8)
    x = 10
    for color, text in _LEGEND_ITEMS:
        cv2.rectangle(bar, (x, 8), (x + 18, 26), color, -1)
        cv2.putText(bar, text, (x + 26, 24), FONT, 0.5, (255, 255, 255), 1)
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        x += 26 + tw + 24
    return np.vstack([canvas, bar])


def _title_card(lines: list[str], size: tuple[int, int]) -> np.ndarray:
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    y = h // 2 - (len(lines) - 1) * 14
    for line in lines:
        (tw, _th), _ = cv2.getTextSize(line, FONT, 0.7, 2)
        x = max(10, (w - tw) // 2)
        cv2.putText(img, line, (x, y), FONT, 0.7, (255, 255, 255), 2)
        y += 28
    return img


# ---------------------------------------------------------------------------------
# 1. SEGMENT AUTO-PICK
# ---------------------------------------------------------------------------------
def _seg_key(seq_name: str, seg: OcclusionSegment) -> str:
    return f"{seq_name}:t{seg.track_id}:f{seg.last_visible_frame}-{seg.reemergence_frame}"


def _legibility_score(seg: OcclusionSegment) -> float:
    """Bigger last-visible GT box height is better (legible in a demo clip); gap length
    in [15, 60] frames is preferred (long enough to show a real gap, short enough to fit
    the target clip duration) -- linear falloff outside that band."""
    height = float(seg.last_visible_box[3])
    gap = seg.gap_length
    if 15 <= gap <= 60:
        gap_term = 1.0
    else:
        dist = (15 - gap) if gap < 15 else (gap - 60)
        gap_term = max(0.0, 1.0 - dist / 60.0)
    return height * (0.3 + 0.7 * gap_term)


@dataclass(frozen=True)
class SegPick:
    seq_name: str
    seg: OcclusionSegment
    key: str
    score: float


def _build_seq_ctx(entry: dict, seq_name: str) -> dict[str, object]:
    tds = {key: dv.load_tracker(seq_name, tag) for key, tag in TAGS.items()}
    dets = load_cached_detections(
        cache_path(ROOT / "data" / "cache" / "detections", seq_name, "yolo11x")
    )
    gt = read_mot(ROOT / "data" / "MOT17" / "train" / seq_name / "gt" / "gt.txt")
    gt = gt[(gt[:, COL.CLS] == 1.0) & (gt[:, COL.CONF] == 1.0)]
    segs = [
        OcclusionSegment(
            int(d["track_id"]), int(d["last_visible_frame"]), int(d["reemergence_frame"]),
            np.array(d["last_visible_box"]), np.array(d["reemergence_box"]),
        )
        for d in entry["dev_half"]
    ]
    return {"tds": tds, "dets": dets, "gt": gt, "segs": segs, "mid": int(entry["seq_length"]) // 2}


def load_all_ctx() -> dict[str, dict]:
    seg_index = json.loads(
        (ROOT / "results" / "occlusion_segments.json").read_text(encoding="utf-8")
    )
    ctxs: dict[str, dict] = {}
    for seq_name, entry in seg_index["sequences"].items():
        if not entry["dev_half"]:
            continue
        ctxs[seq_name] = _build_seq_ctx(entry, seq_name)
    return ctxs


def pick_segments(ctxs: dict[str, dict]) -> dict[str, SegPick]:
    """Score every dev segment for legibility and filter by the per-section outcome
    pattern (context.md D28-final config tags): S1vS2 needs baseline SWITCHED/POST_NONE
    with geometric RETAINED; S3vS4 needs in45 SWITCHED with conv RETAINED; HERO wants
    both (base fails AND in45 switches AND conv retains), falling back to the S3vS4
    pick when no segment satisfies all three at once."""
    s1s2_cands: list[SegPick] = []
    s3s4_cands: list[SegPick] = []
    hero_cands: list[SegPick] = []
    for seq_name, ctx in ctxs.items():
        for seg in ctx["segs"]:
            out = {key: dv.outcome_for(seg, td) for key, td in ctx["tds"].items()}
            score = _legibility_score(seg)
            pick = SegPick(seq_name, seg, _seg_key(seq_name, seg), score)
            base_fails = out["base"].category in ("SWITCHED", "POST_NONE")
            geom_retains = out["geom"].category == "RETAINED"
            in45_switches = out["in45"].category == "SWITCHED"
            conv_retains = out["conv"].category == "RETAINED"
            if base_fails and geom_retains:
                s1s2_cands.append(pick)
            if in45_switches and conv_retains:
                s3s4_cands.append(pick)
            if base_fails and in45_switches and conv_retains:
                hero_cands.append(pick)

    assert s1s2_cands, "no dev segment with base SWITCHED/POST_NONE + geom RETAINED"
    assert s3s4_cands, "no dev segment with in45 SWITCHED + conv RETAINED"
    s1s2_cands.sort(key=lambda p: -p.score)
    s3s4_cands.sort(key=lambda p: -p.score)
    picks: dict[str, SegPick] = {"s1_vs_s2": s1s2_cands[0], "s3_vs_s4": s3s4_cands[0]}
    if hero_cands:
        hero_cands.sort(key=lambda p: -p.score)
        # two segments, ideally from different sequences, so the hero cannot be read as
        # one lucky pick; falls back to the top two if only one sequence qualifies.
        chosen = [hero_cands[0]]
        for cand in hero_cands[1:]:
            if cand.seq_name != chosen[0].seq_name:
                chosen.append(cand)
                break
        if len(chosen) == 1 and len(hero_cands) > 1:
            chosen.append(hero_cands[1])
        picks["hero"] = chosen
        log.info("hero: %d dedicated candidate(s), using %d segment(s) from %s",
                 len(hero_cands), len(chosen), [c.seq_name for c in chosen])
    else:
        picks["hero"] = [picks["s3_vs_s4"]]
        log.info("hero: no dedicated base-fails+in45-switches+conv-retains segment -- "
                  "falling back to the s3_vs_s4 pick")
    for name, pick in picks.items():
        for one in (pick if isinstance(pick, list) else [pick]):
            log.info("picked %s: %s (score=%.1f, gap=%d)", name, one.key, one.score,
                      one.seg.gap_length)
    return picks


# ---------------------------------------------------------------------------------
# 2. CLIP RENDERING -- tracker panels (reuses dv.draw_panel/outcome_for)
# ---------------------------------------------------------------------------------
def render_pair_clip(
    seq_name: str, seg: OcclusionSegment, ctx: dict, left_key: str, right_key: str,
    left_label: str, right_label: str, dest: Path,
    canvas_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Side-by-side clip. Returns the (w, h) canvas used, so several segments can be
    concatenated into one hero reel without ffmpeg refusing on a size change."""
    seq = load_sequence(ROOT / "data" / "MOT17" / "train" / seq_name)
    left_td, right_td = ctx["tds"][left_key], ctx["tds"][right_key]
    out_l = dv.outcome_for(seg, left_td)
    out_r = dv.outcome_for(seg, right_td)
    f0 = max(1, seg.last_visible_frame - PAD_FRAMES)
    f1 = min(int(ctx["mid"]), seg.reemergence_frame + PAD_FRAMES)

    used_size: tuple[int, int] | None = None

    def frames() -> Iterable[np.ndarray]:
        nonlocal used_size
        for f in range(f0, f1 + 1):
            img = cv2.imread(str(seq.frame_path(f)))
            assert img is not None, f"missing frame {seq.frame_path(f)}"
            left = dv.draw_panel(img, f, left_td, ctx["dets"], ctx["gt"], seg, out_l, left_label)
            right = dv.draw_panel(img, f, right_td, ctx["dets"], ctx["gt"], seg, out_r, right_label)
            canvas = _append_legend(np.hstack([left, right]))
            if canvas_size:
                canvas = cv2.resize(canvas, canvas_size)
            used_size = (canvas.shape[1], canvas.shape[0])
            n = 2 if seg.last_visible_frame <= f <= seg.reemergence_frame else 1
            for _ in range(n):
                yield canvas

    _write_video(frames(), dest, dv.FPS)
    assert used_size is not None
    return used_size


def render_solo_clip(
    seq_name: str, seg: OcclusionSegment, ctx: dict, key: str, label: str, dest: Path,
    canvas_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Single-panel clip; returns the (w, h) canvas size actually used (hero reuses it)."""
    seq = load_sequence(ROOT / "data" / "MOT17" / "train" / seq_name)
    td = ctx["tds"][key]
    out = dv.outcome_for(seg, td)
    f0 = max(1, seg.last_visible_frame - PAD_FRAMES)
    f1 = min(int(ctx["mid"]), seg.reemergence_frame + PAD_FRAMES)
    used_size: tuple[int, int] | None = None

    def frames() -> Iterable[np.ndarray]:
        nonlocal used_size
        for f in range(f0, f1 + 1):
            img = cv2.imread(str(seq.frame_path(f)))
            assert img is not None, f"missing frame {seq.frame_path(f)}"
            panel = dv.draw_panel(img, f, td, ctx["dets"], ctx["gt"], seg, out, label)
            canvas = _append_legend(panel)
            if canvas_size:
                canvas = cv2.resize(canvas, canvas_size)
            used_size = (canvas.shape[1], canvas.shape[0])
            n = 2 if seg.last_visible_frame <= f <= seg.reemergence_frame else 1
            for _ in range(n):
                yield canvas

    _write_video(frames(), dest, dv.FPS)
    assert used_size is not None
    return used_size


# ---------------------------------------------------------------------------------
# 3. S6 DETECTOR OVERLAY -- cached detections only, no tracker
# ---------------------------------------------------------------------------------
def _pick_crowded_window(gt: np.ndarray, dev_end: int, window: int = 150) -> tuple[int, int]:
    counts = {f: int((gt[:, COL.FRAME] == f).sum()) for f in range(1, dev_end + 1)}
    peak = max(counts, key=counts.get)
    f0 = max(1, peak - window // 2)
    f1 = min(dev_end, f0 + window)
    f0 = max(1, f1 - window)
    return f0, f1


def _draw_det_panel(
    img: np.ndarray, frame: int, dets: dict[int, np.ndarray], label: str,
    caveat: str | None = None, score_floor: float = 0.25,
) -> np.ndarray:
    vis = img.copy()
    boxes = dets.get(frame, np.zeros((0, 5)))
    kept = boxes[boxes[:, 4] >= score_floor] if len(boxes) else boxes
    for b in kept:
        x, y, w, h = (int(v) for v in b[:4])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (14, 127, 255), 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (30, 30, 30), -1)
    cv2.putText(vis, f"{label} | f{frame} | {len(kept)} dets>={score_floor:.2f}",
                (8, 18), FONT, 0.55, (255, 255, 255), 1)
    if caveat:
        cv2.rectangle(vis, (0, vis.shape[0] - 26), (vis.shape[1], vis.shape[0]), (0, 0, 160), -1)
        cv2.putText(vis, caveat, (8, vis.shape[0] - 8), FONT, 0.5, (255, 255, 255), 1)
    scale = dv.PANEL_H / vis.shape[0]
    return cv2.resize(vis, (int(vis.shape[1] * scale), dv.PANEL_H))


def render_detector_clip(dest: Path, seq_name: str = "MOT17-02-FRCNN") -> None:
    seg_index = json.loads(
        (ROOT / "results" / "occlusion_segments.json").read_text(encoding="utf-8")
    )
    dev_end = int(seg_index["sequences"][seq_name]["val_start_frame"]) - 1
    seq = load_sequence(ROOT / "data" / "MOT17" / "train" / seq_name)
    gt = read_mot(ROOT / "data" / "MOT17" / "train" / seq_name / "gt" / "gt.txt")
    gt = gt[(gt[:, COL.CLS] == 1.0) & (gt[:, COL.CONF] == 1.0)]
    f0, f1 = _pick_crowded_window(gt, dev_end)
    dets_x = load_cached_detections(
        cache_path(ROOT / "data" / "cache" / "detections", seq_name, "yolo11x")
    )
    dets_ft = load_cached_detections(
        cache_path(ROOT / "data" / "cache" / "detections", seq_name, "yolo11s_ft")
    )
    log.info("s6 window: %s frames %d-%d (crowded dev-half window)", seq_name, f0, f1)

    def frames() -> Iterable[np.ndarray]:
        for f in range(f0, f1 + 1):
            img = cv2.imread(str(seq.frame_path(f)))
            assert img is not None, f"missing frame {seq.frame_path(f)}"
            left = _draw_det_panel(img, f, dets_x, "stock yolo11x")
            right = _draw_det_panel(
                img, f, dets_ft, "finetuned yolo11s (dev-half)",
                caveat="detector finetuned ON dev half - optimistic here",
            )
            yield np.hstack([left, right])

    _write_video(frames(), dest, dv.FPS)


# ---------------------------------------------------------------------------------
# 4/5. CARLA (S5) + teaser (S0) -- gt.txt already carries per-frame visibility (COL.VIS)
# ---------------------------------------------------------------------------------
def _vis_color(v: float) -> tuple[int, int, int]:
    if v >= 0.5:
        return (0, 200, 0)
    if v >= 0.25:
        return (0, 165, 255)
    return (0, 0, 220)


def render_carla_clip(
    dest: Path, scenario_dir: Path, frame_range: tuple[int, int] | None = None,
    canvas_size: tuple[int, int] | None = None, fps: int | None = None,
) -> None:
    seq = load_sequence(scenario_dir)
    assert seq.gt is not None and len(seq.gt), f"no GT for scenario {scenario_dir}"
    gt = seq.gt
    f0, f1 = frame_range or (1, seq.seq_length)
    use_fps = fps or int(round(seq.frame_rate))

    def frames() -> Iterable[np.ndarray]:
        for f in range(f0, f1 + 1):
            img = cv2.imread(str(seq.frame_path(f)))
            assert img is not None, f"missing carla frame {seq.frame_path(f)}"
            vis_img = img.copy()
            for r in gt[gt[:, COL.FRAME] == f]:
                wid = int(r[COL.ID])
                x, y, w, h, v = r[COL.X], r[COL.Y], r[COL.W], r[COL.H], r[COL.VIS]
                color = _vis_color(v)
                x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
                cv2.rectangle(vis_img, (x0, y0), (x1, y1), color, 2)
                cv2.putText(vis_img, f"{wid} v{v:.2f}", (x0, max(12, y0 - 4)),
                            FONT, 0.45, color, 1)
            cv2.rectangle(vis_img, (0, 0), (vis_img.shape[1], 26), (30, 30, 30), -1)
            cv2.putText(vis_img, f"{scenario_dir.name} | f{f} | CARLA GT visibility overlay",
                        (8, 18), FONT, 0.5, (255, 255, 255), 1)
            if canvas_size:
                vis_img = cv2.resize(vis_img, canvas_size)
            yield vis_img

    _write_video(frames(), dest, use_fps)


def render_teaser(dest: Path, scenario_dir: Path) -> None:
    seq = load_sequence(scenario_dir)
    assert seq.gt is not None and len(seq.gt), f"no GT for scenario {scenario_dir}"
    gt = seq.gt
    best_f, best_n = None, -1
    for f in np.unique(gt[:, COL.FRAME]):
        n_occ = int((gt[gt[:, COL.FRAME] == f][:, COL.VIS] < 0.4).sum())
        if n_occ > best_n:
            best_n, best_f = n_occ, int(f)
    assert best_f is not None
    img = cv2.imread(str(seq.frame_path(best_f)))
    assert img is not None, f"missing teaser frame {seq.frame_path(best_f)}"
    vis_img = img.copy()
    for r in gt[gt[:, COL.FRAME] == best_f]:
        wid = int(r[COL.ID])
        x, y, w, h, v = r[COL.X], r[COL.Y], r[COL.W], r[COL.H], r[COL.VIS]
        color = _vis_color(v)
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(vis_img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(vis_img, f"{wid} v{v:.2f}", (x0, max(12, y0 - 4)), FONT, 0.5, color, 1)
    cv2.rectangle(vis_img, (0, 0), (vis_img.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(vis_img, f"{scenario_dir.name} | f{best_f} | {best_n} walkers occluded (vis<0.4)",
                (8, 21), FONT, 0.55, (255, 255, 255), 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), vis_img)
    log.info("wrote %s (frame %d, %d simultaneously-occluded walkers)", dest, best_f, best_n)


def render_blueprint_grid(dest: Path, cols: int = 9) -> None:
    reid_dir = ROOT / "data" / "reid" / "sim"
    index = json.loads((reid_dir / "index.json").read_text(encoding="utf-8"))
    identities = sorted(index["identities"].keys())
    assert len(identities) == 45, f"expected 45 sim blueprint identities, got {len(identities)}"
    cells = []
    for ident in identities:
        crops = sorted((reid_dir / ident).glob("*.jpg"))
        assert crops, f"no crops for identity {ident}"
        crop = cv2.imread(str(crops[len(crops) // 2]))  # median crop -- deterministic pick
        assert crop is not None, f"unreadable crop {crops[len(crops) // 2]}"
        cell = cv2.resize(crop, (72, 144))
        cell = cv2.copyMakeBorder(cell, 2, 16, 2, 2, cv2.BORDER_CONSTANT, value=(230, 230, 230))
        label = ident.replace("walker_pedestrian_", "")
        cv2.putText(cell, label, (4, cell.shape[0] - 4), FONT, 0.35, (0, 0, 0), 1)
        cells.append(cell)
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i : i + cols]
        row += [np.full_like(cells[0], 255)] * (cols - len(row))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    header = np.full((30, grid.shape[1], 3), 25, np.uint8)
    cv2.putText(header,
                f"CARLA sim re-ID: {len(identities)} walker blueprints (D24 identity-correct)",
                (8, 20), FONT, 0.5, (255, 255, 255), 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), np.vstack([header, grid]))
    log.info("wrote %s (%d cells, %dx%d grid)", dest, len(cells), len(rows), cols)


# ---------------------------------------------------------------------------------
# 6. HERO -- 4 solo stages + title cards + CARLA tail, ffmpeg concat demuxer
# ---------------------------------------------------------------------------------
_HERO_STAGES = [
    ("base", "S1: ByteTrack baseline\nocclusion breaks the ID"),
    ("geom", "S2: + geometric hidden-state\ncoasts through the gap"),
    ("in45", "S3: + ImageNet appearance veto\nstill switches identity"),
    ("conv", "S4: + trained embedder\nretains identity"),
]


def render_hero(picks: list[SegPick], ctxs: dict, dest: Path) -> None:
    """THE hero clip: ByteTrack baseline (left) vs the full stack (right), side by side,
    through real occlusions, with persistent ID labels.

    Design constraints (D64, user): ONE clip, ~30 s, readable in the first few seconds by
    someone who will not read the README. So: no four-stage build-up, no CARLA tail, no
    per-stage title cards. Just the comparison that carries the claim, over several
    segments so it cannot be dismissed as one lucky pick.
    """
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        parts: list[Path] = []
        canvas_size: tuple[int, int] | None = None

        for i, pick in enumerate(picks, 1):
            seg_path = td / f"seg{i}.mp4"
            canvas_size = render_pair_clip(
                pick.seq_name, pick.seg, ctxs[pick.seq_name], "base", "conv",
                "ByteTrack baseline", "+ hidden-state + trained re-ID",
                seg_path, canvas_size,
            )
            if i == 1:  # one short opener, then straight into the evidence
                title = td / "title.mp4"
                card = _title_card(
                    ["Same video. Same detections. Same occlusion.",
                     "Left: ByteTrack.   Right: ours.",
                     "Watch the ID number survive the gap."],
                    canvas_size,
                )
                _write_video((card for _ in range(2 * dv.FPS)), title, dv.FPS)
                parts.append(title)
            parts.append(seg_path)

        assert canvas_size is not None
        outro = td / "outro.mp4"
        card = _title_card(
            ["MOT17 dev-half, 168 occlusion segments:",
             "identity retention 0.292 -> 0.345",
             "standard tracking quality unchanged (HOTA/IDF1 parity)"],
            canvas_size,
        )
        _write_video((card for _ in range(3 * dv.FPS)), outro, dv.FPS)
        parts.append(outro)

        list_file = td / "concat.txt"
        list_file.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(list_file),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
            check=True, timeout=900,
        )
    log.info("wrote %s (baseline vs full stack, %d segment(s) side by side)", dest, len(picks))


# ---------------------------------------------------------------------------------
# 7. README
# ---------------------------------------------------------------------------------
def _load_g2(tag: str) -> dict:
    path = ROOT / "results" / f"hidden_dev_{tag}.json"
    return json.loads(path.read_text(encoding="utf-8"))["g2"]


def write_readme(picks: dict[str, SegPick]) -> None:
    base_g2 = _load_g2("audit_base")
    geom_g2 = _load_g2("audit_geom")
    in45_g2 = _load_g2("audit_in45")
    conv_g2 = _load_g2("conv_app45")
    ftgeom_g2 = _load_g2("ftdet_geom")
    ftconv_g2 = _load_g2("ftdet_conv_app45")
    paired = json.loads(
        (ROOT / "results" / "paired_g2a_dev_hidden_conv_app45_vs_hidden_audit_in45.json")
        .read_text(encoding="utf-8")
    )
    scaling_path = ROOT / "results" / "scaling_curve_local.json"
    scaling = json.loads(scaling_path.read_text(encoding="utf-8"))
    rank1s = [r["occ_rank1"] for r in scaling["results"]]
    n_ids = sorted({r["n_train_ids"] for r in scaling["results"]})

    text = f"""\
# occlusion-mot demo package (D39)

Guided tour S0-S8: the failure mode -> geometric hidden-state -> appearance re-ID
(ImageNet null vs trained embedder) -> detector upgrade -> CARLA sim2real feeder ->
identity-scaling study -> gate scoreboard. Regenerate everything:
`.venv/Scripts/python.exe scripts/render_demo.py --all`

## Watch this one first -- `clips/hero.mp4` (~30 s)

**If you only have half a minute, this is the clip.** ByteTrack baseline on the left, the
full stack on the right: same video, same cached detections, same occlusion. Watch the ID
number above the tracked person survive the gap on the right and reset on the left.
Two segments from two different sequences, so it is not one lucky pick.

Open: `start demo/clips/hero.mp4`

Stack shown: yolo11x cached detections + geometric hidden-state + trained conv embedder
(app-gate 0.45) -- the exact configuration behind the dev and val tables in the top-level
README. The Stage-1 PERUN detectors are deliberately NOT used here: every one of them
scored below this baseline detector on MOT17 (context.md D61), so showing one would
misrepresent the result.

## S0 -- teaser
![teaser](s0_teaser.png)
One CARLA frame (`{CARLA_SCENARIO.name}`) with heavy inter-walker occlusion, GT
visibility burned in (green >=0.5, orange 0.25-0.5, red <0.25).
Open: `start demo/s0_teaser.png`
Caveat: synthetic scene, sets up the problem -- not a tracker result.

## S1 vs S2 -- geometric hidden-state fixes coasting failures
`clips/s1_vs_s2.mp4` -- side by side on segment `{picks["s1_vs_s2"].key}`: S1 baseline
(left) loses the identity through occlusion, S2 geometric hidden-state (right) coasts
through it and retains.
Dev-half (n=168): retention {base_g2["id_retention"]:.3f} -> {geom_g2["id_retention"]:.3f};
assoc-scope retention {base_g2["id_retention_assoc"]:.3f} -> {geom_g2["id_retention_assoc"]:.3f}.
Open: `start demo/clips/s1_vs_s2.mp4`
Caveat: geometric mechanisms ceiling out at +1.8pt retention (context.md D19) -- motivates S3/S4.

## S2 solo -- coasting box while hidden
`clips/s2_solo.mp4` -- single panel, same segment as S1vS2, geometric tracker only;
orange box = the coasting hidden-state position estimate while occluded.
Open: `start demo/clips/s2_solo.mp4`
Caveat: constant-velocity-damped estimate, not a learned motion predictor (v1 design, D19).

## S3 vs S4 -- trained embedder beats the ImageNet null
`clips/s3_vs_s4.mp4` -- segment `{picks["s3_vs_s4"].key}`: S3 ImageNet appearance veto
(left) still switches identity, S4 trained embedder (right) retains it.
Dev-half: ImageNet retention {in45_g2["id_retention"]:.3f}
(assoc {in45_g2["id_retention_assoc"]:.3f}) -> trained-embedder retention
{conv_g2["id_retention"]:.3f} (assoc {conv_g2["id_retention_assoc"]:.3f}).
Paired on the intersection set (n={paired["n_intersection"]}): {paired["retention_a"]:.3f} vs
{paired["retention_b"]:.3f}, discordant {paired["discordant_a_only"]}/{paired["discordant_b_only"]},
McNemar p={paired["p_value"]:.2f}.
Open: `start demo/clips/s3_vs_s4.mp4`
Caveat: p={paired["p_value"]:.2f} is NOT significant at dev scale (n~{paired["n_intersection"]}) --
qualitative win only; the frozen G2a-paired claim needs PERUN-scale identities (D29).

## S4 -- re-ID crop grid (pre/post gap appearance)
![crops](../viz/crops_assoc_scope.png)
Pre-gap | post-gap GT crops for every assoc-scope dev segment under S4, bordered green
(retained) / red (switched) -- where trained appearance still fails (blur, crowd handoff).
Untracked (D37 license hygiene: dataset-derived crop grid). Regenerate:
`.venv/Scripts/python.exe scripts/render_dev_viz.py --pngs-only`
Caveat: MOT17 GT crops -- never committed to the repo per D37.

## S5 -- CARLA sim2real feeder
`s5_carla.mp4` -- one CARLA scenario (`{CARLA_SCENARIO.name}`, ~20s) with per-walker GT
visibility overlaid; committed (synthetic render, not dataset-derived -- D37).
`s5_blueprint_grid.png` -- one crop per walker blueprint identity (45 total, 5x9 grid)
from the sim re-ID pool, appearance-correct labels after the D24 audit fix.
Open: `start demo/s5_carla.mp4` and `start demo/s5_blueprint_grid.png`
Caveat: 45 blueprints is CARLA's exhausted appearance ceiling (D34) -- no further sim
identity diversity without UE4-editor asset work.

## S6 -- detector finetune (honest optimistic read)
`clips/s6_detector.mp4` -- cached detections only (no tracker), stock yolo11x (left) vs
yolo11s finetuned on MOT17 dev-half GT (right), crowded MOT17-02 dev-half window.
Dev-half effect (geometric config): pre-match {ftgeom_g2["pre_match_rate"]:.3f}, oracle
ceiling {ftgeom_g2["oracle_ceiling"]:.3f}, e2e retention {ftgeom_g2["id_retention"]:.3f}
(full stack + trained embedder: {ftconv_g2["id_retention"]:.3f}).
Open: `start demo/clips/s6_detector.mp4`
Caveat (burned into the right panel too): the detector TRAINED on these dev-half frames
-- train-on-train, optimistic; the clean read is the val-half run (D31, D35).

## S7 -- identity-scaling study
![scaling](../viz/scaling_curve_local.png)
![scaling-smoke](../viz/scaling_study.png)
Identity count scales re-ID retrieval quality monotonically (occ-rank1
{min(rank1s):.3f} -> {max(rank1s):.3f} over {min(n_ids)} -> {max(n_ids)} identities, both
seeds); tracker-level assoc retention does NOT resolve at dev scale (seed spread up to
5.8pt swamps the identity effect at n~97 segments) -- D36.
Caveat: this local curve motivated the PERUN sweep -- and the sweep then REFUTED the
tracker-level hypothesis. At full scale (23 units, 3 seeds) retrieval kept climbing
(occ-rank1 0.408 -> 0.610 over 300 -> 3520 identities) while assoc retention moved only
0.523 -> 0.578 against a seed spread of 0.078. More identities buy a better embedder,
not a better tracker (D50). See ../viz/perun_sweep_identity_curve.png.

## S8 -- gate scoreboard + next steps
| gate | status |
|---|---|
| G0 repro | frozen PASS |
| G1 parity | **PASS on val** -- HOTA 51.07 / IDF1 60.09 / IDsw 298, all beat baseline (D45) |
| G2 center-err | {conv_g2["reemergence_center_err_med"]:.4f} PASS (<=0.015); val 0.0090 PASS |
| G2 cov_prematched | 0.915 dev PASS / 0.907 val PASS (>=0.90) |
| G2a-floor (assoc) | dev 0.598-0.604 PASS; **val 0.440 MISS**; PERUN 0.578 vs >=0.58 **FAIL** |
| G2a-paired (McNemar) | **FAIL at scale** -- best p=0.143 over 3 seeds vs <0.05 (D50) |
| G2b (end-to-end) | 0.345 -- detector confirmed as the binding constraint (D55) |
| G3 quality | PASS |
| G4 CARLA feeder | PASS |

The G2a row is the honest headline: the appearance-scaling route was tested properly at
PERUN scale and **failed both limbs**. It is reported as a null, not tuned away (D50/D51).

Stage 0 then located the real bottleneck: feeding the frozen tracker perfect (visible) GT
boxes and changing nothing else lifts the recoverable-occlusion ceiling 0.571 -> 0.970 and
end-to-end retention 0.345 -> 0.786 (D55, ../viz/stage0_kill_gate.png). Two caveats travel
with it: 0.786 is an upper bound, not a promise, and ~19% of recoverable segments are
still lost INSIDE the tracker -- a residual no detector can close.

Next: Stage-1 detector dose-response, 12 units trained only on MOT17-disjoint sources so
the dev read is honest by construction (perun_detector_v1.md, frozen D53). Val is CLOSED
for that workstream (D53 decision 1).

Regenerate all clips/images: `.venv/Scripts/python.exe scripts/render_demo.py --all`
"""
    dest = DEMO_DIR / "README.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    log.info("wrote %s", dest)


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=SECTIONS, default=None,
                     help="run a single section (default: all)")
    ap.add_argument("--all", action="store_true", help="run every section (default behavior)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert not (args.only and args.all), "--only and --all are mutually exclusive"
    set_seeds(args.seed)

    sections = [args.only] if args.only else SECTIONS
    DEMO_DIR.mkdir(exist_ok=True)
    CLIPS_DIR.mkdir(exist_ok=True)

    picks_sections = {"segments", "s1_vs_s2", "s2_solo", "s3_vs_s4", "hero", "readme"}
    needs_picks = bool(picks_sections & set(sections))
    ctxs = load_all_ctx() if needs_picks else {}
    picks = pick_segments(ctxs) if needs_picks else {}

    if "s1_vs_s2" in sections:
        p = picks["s1_vs_s2"]
        render_pair_clip(p.seq_name, p.seg, ctxs[p.seq_name], "base", "geom",
                          LABELS["base"], LABELS["geom"], CLIPS_DIR / "s1_vs_s2.mp4")
    if "s2_solo" in sections:
        p = picks["s1_vs_s2"]
        render_solo_clip(p.seq_name, p.seg, ctxs[p.seq_name], "geom", LABELS["geom"],
                          CLIPS_DIR / "s2_solo.mp4")
    if "s3_vs_s4" in sections:
        p = picks["s3_vs_s4"]
        render_pair_clip(p.seq_name, p.seg, ctxs[p.seq_name], "in45", "conv",
                          LABELS["in45"], LABELS["conv"], CLIPS_DIR / "s3_vs_s4.mp4")
    if "s6" in sections:
        render_detector_clip(CLIPS_DIR / "s6_detector.mp4")
    if "s5" in sections:
        render_carla_clip(DEMO_DIR / "s5_carla.mp4", CARLA_SCENARIO)
        render_blueprint_grid(DEMO_DIR / "s5_blueprint_grid.png")
    if "s0" in sections:
        render_teaser(DEMO_DIR / "s0_teaser.png", CARLA_SCENARIO)
    if "hero" in sections:
        hero_picks = picks["hero"] if isinstance(picks["hero"], list) else [picks["hero"]]
        render_hero(hero_picks, ctxs, CLIPS_DIR / "hero.mp4")
    if "readme" in sections:
        write_readme(picks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
