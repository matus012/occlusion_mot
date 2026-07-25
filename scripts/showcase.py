"""D42 wild-clip showcase: batch qualitative demo over arbitrary stock mp4s.

    .venv/Scripts/python.exe scripts/showcase.py --src showcase/sources \\
        --out showcase/renders [--max-seconds 12] [--stride 1] [--device auto] \\
        [--weights data/models/reid_conv.pt] [--ckpt-label placeholder-ckpt] [--seed 0]

Per clip: decode with cv2.VideoCapture, run yolo11x person detection + the reid
embedder LIVE on every kept frame, feed both into OcclusionAwareTracker (conv_app45
operating point), overlay track boxes (colored by a stable per-id palette) with
occluded/coasting tracks drawn as dashed orange hidden-state predictions, and
transcode to H.264. After every clip is rendered, assemble one 2-column contact
sheet PNG from each clip's busiest frame.

FIXED DETECTIONS invariant (context.md D1) does NOT apply here: that invariant
governs tracker-vs-tracker comparisons that must share one cached detection set.
This script runs a single tracker configuration once per clip on wild, unlabeled
video for a qualitative showcase -- there is no comparison and no cache to share.

No metrics, counts, scores, or GT are ever drawn on the frames (D36): this is a
qualitative demo by decree, not a benchmark result.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import imageio_ffmpeg
import numpy as np

if TYPE_CHECKING:
    import torch
    from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omot.detect.cache import PERSON_CLASS, select_device, set_seeds  # noqa: E402
from omot.detect.embed import CROP_HW, _build_embedder  # noqa: E402
from omot.hidden.occlusion_tracker import HiddenConfig, OcclusionAwareTracker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("showcase")

FONT = cv2.FONT_HERSHEY_SIMPLEX
CONF_FLOOR = 0.05  # same conventions as detect/cache.py's default
IMGSZ = 1280
MODEL_NAME = "yolo11x.pt"
DEFAULT_WEIGHTS = ROOT / "data" / "models" / "reid_conv.pt"

LEGEND_H = 30
HIDDEN_COLOR = (0, 165, 255)  # BGR orange: occluded / coasting hidden-state prediction
ACTIVE_LEGEND_COLOR = (0, 220, 0)
WATERMARK_ALPHA = 0.55
THUMB_WIDTH = 480
CONTACT_COLS = 2


# ---------------------------------------------------------------------------------
# config / device / seeding
# ---------------------------------------------------------------------------------
def build_hidden_config() -> HiddenConfig:
    """The conv_app45 operating point (context.md D28-final; val_manifest.md R1),
    frozen for this qualitative showcase."""
    return HiddenConfig(
        occl_buffer=90,
        vel_damping=1.0,
        recover_gate=1.5,
        occl_overlap_thresh=0.25,
        lowconf_mode="kf",
        lowconf_noise_scale=1.0,
        app_gate_lost=0.45,
        app_gate_recover=0.45,
    )


def resolve_device(requested: str) -> str:
    """--device auto maps to select_device's cuda/cpu auto-detect (no hardcoded
    cuda:0); an explicit string (e.g. "cuda", "cpu", "cuda:1" for future SLURM
    multi-GPU) passes straight through with CPU always available as a fallback."""
    return select_device(None if requested == "auto" else requested)


# ---------------------------------------------------------------------------------
# stable per-id color palette + drawing helpers
# ---------------------------------------------------------------------------------
def stable_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic per-id BGR color: golden-angle-ish hue rotation keyed on the
    id, so colors stay identical for the same id across every frame/run (no RNG
    state, no growing dict) while spreading nearby ids apart in hue."""
    hue = (int(track_id) * 137) % 180  # OpenCV hue range is [0, 179]
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _dashed_rect(
    img: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int],
    color: tuple[int, int, int], thickness: int = 1, dash: int = 8,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def _append_legend(canvas: np.ndarray) -> np.ndarray:
    bar = np.full((LEGEND_H, canvas.shape[1], 3), 20, np.uint8)
    items = [
        (ACTIVE_LEGEND_COLOR, "active track (id color)"),
        (HIDDEN_COLOR, "occluded / hidden-state prediction"),
    ]
    x = 10
    for color, text in items:
        cv2.rectangle(bar, (x, 6), (x + 16, 22), color, -1)
        cv2.putText(bar, text, (x + 22, 20), FONT, 0.45, (255, 255, 255), 1)
        (tw, _th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        x += 22 + tw + 20
    return np.vstack([canvas, bar])


def _draw_watermark(vis: np.ndarray, ckpt_label: str) -> None:
    text = f"{ckpt_label} | qualitative only"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    x0, y0 = vis.shape[1] - tw - 16, vis.shape[0] - th - 16
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 6), (vis.shape[1] - 4, vis.shape[0] - 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, WATERMARK_ALPHA, vis, 1 - WATERMARK_ALPHA, 0, dst=vis)
    cv2.putText(vis, text, (x0, y0 + th), FONT, 0.45, (255, 255, 255), 1)


def draw_overlay(
    frame: np.ndarray, active: np.ndarray, coasting: np.ndarray, ckpt_label: str,
) -> np.ndarray:
    """Active tracks: solid box + id-stable color. Coasting/occluded tracks: dashed
    thin orange box + "hidden" tag (the hidden-agent predicted pose, not a real
    detection). No metrics/scores/counts drawn anywhere (D36)."""
    vis = frame.copy()
    for x, y, w, h, _score, tid in active:
        color = stable_color(int(tid))
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        cv2.putText(vis, f"id {int(tid)}", (x0, max(12, y0 - 4)), FONT, 0.5, color, 1)
    for x, y, w, h, _score, tid in coasting:
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        _dashed_rect(vis, (x0, y0), (x1, y1), HIDDEN_COLOR, thickness=1)
        cv2.putText(vis, f"id {int(tid)} hidden", (x0, max(12, y0 - 4)), FONT, 0.45,
                    HIDDEN_COLOR, 1)
    vis = _append_legend(vis)
    _draw_watermark(vis, ckpt_label)
    return vis


# ---------------------------------------------------------------------------------
# naming + best-frame selection (pure, testable)
# ---------------------------------------------------------------------------------
def output_name(clip_path: Path, ckpt_label: str) -> str:
    return f"{clip_path.stem}__{ckpt_label}.mp4"


def contact_sheet_name(ckpt_label: str) -> str:
    return f"contact_sheet__{ckpt_label}.png"


def select_best_frame_index(track_counts: list[int]) -> int:
    """Pick the index with the most simultaneous tracks (active + coasting/hidden);
    falls back to the mid-clip frame when every frame has zero tracks (e.g. nothing
    detected) so the contact sheet still gets a representative thumbnail."""
    assert track_counts, "no frames to select a contact-sheet thumbnail from"
    best = max(range(len(track_counts)), key=lambda i: track_counts[i])
    if track_counts[best] <= 0:
        return len(track_counts) // 2
    return best


# ---------------------------------------------------------------------------------
# contact sheet assembly (pure, testable)
# ---------------------------------------------------------------------------------
def make_thumbnail(frame: np.ndarray, width: int = THUMB_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, max(1, int(round(h * scale)))))


def caption_cell(thumb: np.ndarray, caption: str) -> np.ndarray:
    bar_h = 24
    bar = np.full((bar_h, thumb.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, caption, (8, 17), FONT, 0.5, (255, 255, 255), 1)
    return np.vstack([thumb, bar])


def assemble_contact_sheet(cells: list[np.ndarray], cols: int = CONTACT_COLS) -> np.ndarray:
    """Grid a list of same-format cells into `cols` columns, padding the trailing
    row with blank cells and padding any undersized cell to the max cell size."""
    assert cells, "no cells to assemble into a contact sheet"
    cell_h = max(c.shape[0] for c in cells)
    cell_w = max(c.shape[1] for c in cells)
    padded = []
    for c in cells:
        pad = np.zeros((cell_h, cell_w, 3), np.uint8)
        pad[: c.shape[0], : c.shape[1]] = c
        padded.append(pad)
    blank = np.zeros((cell_h, cell_w, 3), np.uint8)
    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i : i + cols]
        row = row + [blank] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def build_contact_sheet(
    entries: list[tuple[str, np.ndarray, float]], cols: int = CONTACT_COLS,
) -> np.ndarray:
    """entries: (clip stem, busiest-frame thumbnail source, duration seconds)."""
    cells = [
        caption_cell(make_thumbnail(frame), f"{stem} ({duration:.1f}s)")
        for stem, frame, duration in entries
    ]
    return assemble_contact_sheet(cells, cols)


# ---------------------------------------------------------------------------------
# video I/O
# ---------------------------------------------------------------------------------
def _transcode(tmp: Path, dest: Path) -> None:
    """cv2's mp4v is unplayable in stock Windows players -- transcode to H.264
    (same pattern as render_demo.py)."""
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
        check=True, timeout=600,
    )
    tmp.unlink()


def write_video(frames: list[np.ndarray], dest: Path, fps: float) -> None:
    assert frames, f"no frames to write for {dest}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for f in frames:
        writer.write(f)
    writer.release()
    _transcode(tmp, dest)
    logger.info("wrote %s (%d frames @ %.2ffps)", dest, len(frames), fps)


# ---------------------------------------------------------------------------------
# live detection + embedding (per frame)
# ---------------------------------------------------------------------------------
def detect_frame(yolo: YOLO, frame: np.ndarray, device: str) -> np.ndarray:
    """Person detections for one frame: (N, 5) [x, y, w, h, score] tlwh -- same
    conf floor / imgsz / person-class conventions as detect/cache.py."""
    result = yolo.predict(
        source=frame, device=device, classes=[PERSON_CLASS], conf=CONF_FLOOR,
        verbose=False, imgsz=IMGSZ,
    )[0]
    xywh = result.boxes.xywh.cpu().numpy()
    if not len(xywh):
        return np.zeros((0, 5))
    conf = result.boxes.conf.cpu().numpy()
    tlwh = xywh.copy()
    tlwh[:, 0] -= tlwh[:, 2] / 2
    tlwh[:, 1] -= tlwh[:, 3] / 2
    return np.hstack([tlwh, conf[:, None]]).astype(np.float64)


def embed_crops(
    net: torch.nn.Module,
    normalize: Callable[[torch.Tensor], torch.Tensor],
    frame: np.ndarray,
    dets: np.ndarray,
    device: str,
) -> np.ndarray:
    """Crop + embed every detection box (reuses detect/embed's crop/normalize
    convention); L2-normalized, row-aligned with `dets`."""
    import torch

    if not len(dets):
        return np.zeros((0, 1), dtype=np.float32)
    ih, iw = frame.shape[:2]
    crops = []
    for box in dets:
        x, y, w, h = box[:4]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
        if x2 - x1 < 4 or y2 - y1 < 8:
            crop = np.zeros((*CROP_HW, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(frame[y1:y2, x1:x2], (CROP_HW[1], CROP_HW[0]))
        crops.append(crop[:, :, ::-1].copy())  # BGR -> RGB
    arr = np.stack(crops)
    with torch.no_grad():
        t = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).float()
        emb = net(normalize(t)).cpu().numpy().astype(np.float32)
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    return emb


# ---------------------------------------------------------------------------------
# per-clip pipeline
# ---------------------------------------------------------------------------------
def process_clip(
    clip_path: Path,
    out_dir: Path,
    yolo: YOLO,
    embed_net: torch.nn.Module,
    normalize: Callable[[torch.Tensor], torch.Tensor],
    cfg: HiddenConfig,
    device: str,
    max_seconds: float,
    stride: int,
    ckpt_label: str,
) -> tuple[np.ndarray, int, float]:
    """Decode/detect/embed/track/overlay one clip; writes
    <out_dir>/<stem>__<ckpt_label>.mp4 and returns (contact-sheet thumbnail source
    frame, frame count written, clip duration seconds).

    Live inference end-to-end (detector + embedder + tracker) on unlabeled wild
    video -- this is a QUALITATIVE showcase, not a tracker-vs-tracker comparison,
    so the FIXED DETECTIONS invariant (context.md D1) does not apply: there is no
    cached detection set and no second tracker run this needs to stay identical to.
    """
    cap = cv2.VideoCapture(str(clip_path))
    assert cap.isOpened(), f"cannot open clip: {clip_path}"
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(round(max_seconds * native_fps))

    tracker = OcclusionAwareTracker(cfg)
    overlays: list[np.ndarray] = []
    track_counts: list[int] = []
    frame_id = 0
    read_idx = 0
    try:
        while read_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if read_idx % stride == 0:
                frame_id += 1
                dets = detect_frame(yolo, frame, device)
                embs = embed_crops(embed_net, normalize, frame, dets, device)
                active = tracker.update(dets, frame_id=frame_id, embeddings=embs)
                coasting = tracker.coasting
                overlays.append(draw_overlay(frame, active, coasting, ckpt_label))
                track_counts.append(len(active) + len(coasting))
            read_idx += 1
    finally:
        cap.release()

    assert overlays, f"no frames decoded from {clip_path}"
    dest = out_dir / output_name(clip_path, ckpt_label)
    write_video(overlays, dest, native_fps)
    best_idx = select_best_frame_index(track_counts)
    duration = len(overlays) * stride / native_fps
    return overlays[best_idx], len(overlays), duration


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="folder of source .mp4 clips")
    ap.add_argument(
        "--out", type=Path, required=True, help="folder for rendered .mp4 + contact sheet"
    )
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth decoded frame")
    ap.add_argument("--device", default="auto", help='"auto", "cpu", "cuda", or "cuda:N"')
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="reid checkpoint")
    ap.add_argument("--ckpt-label", default="placeholder-ckpt")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args()
    assert args.src.is_dir(), f"--src is not a directory: {args.src}"
    assert args.max_seconds > 0, "--max-seconds must be positive"
    assert args.stride >= 1, "--stride must be >= 1"
    assert args.weights.exists(), f"reid weights not found: {args.weights}"
    set_seeds(args.seed)
    device = resolve_device(args.device)
    logger.info("device=%s ckpt_label=%s weights=%s", device, args.ckpt_label, args.weights)

    clips = sorted(args.src.glob("*.mp4"))
    assert clips, f"no .mp4 clips found in {args.src}"
    args.out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    yolo = YOLO(MODEL_NAME)
    embed_net, normalize = _build_embedder(device, args.weights)
    cfg = build_hidden_config()

    entries: list[tuple[str, np.ndarray, float]] = []
    for clip in clips:
        logger.info("processing %s", clip.name)
        t0 = time.time()
        thumb, n_frames, duration = process_clip(
            clip, args.out, yolo, embed_net, normalize, cfg, device,
            args.max_seconds, args.stride, args.ckpt_label,
        )
        elapsed = time.time() - t0
        fps_proc = n_frames / elapsed if elapsed > 0 else 0.0
        logger.info(
            "%s: %d frames, %.1fs wall time, %.1f fps processed",
            clip.name, n_frames, elapsed, fps_proc,
        )
        entries.append((clip.stem, thumb, duration))

    sheet = build_contact_sheet(entries)
    sheet_path = args.out / contact_sheet_name(args.ckpt_label)
    cv2.imwrite(str(sheet_path), sheet)
    logger.info("wrote %s", sheet_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
