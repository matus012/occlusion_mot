"""Qualitative dev-half read-out: any two tracker tags, side by side (D27 viz pipeline).

Renders side-by-side MP4 clips for selected occlusion segments, a per-segment
retention-outcome PNG grid, and a pre/post re-ID crop grid. READ-ONLY over existing
artifacts — no tracking re-runs.

Outputs to viz/ (mp4 files are gitignored; PNGs are committable).
Usage: .venv/Scripts/python.exe scripts/render_dev_viz.py \
    --base-tag hidden_audit_base --pick-tag hidden_d26best --pick-label "d26best"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import load_sequence  # noqa: E402
from omot.detect.cache import cache_path, load_cached_detections  # noqa: E402
from omot.eval.hidden_eval import _match_id, _rows_at  # noqa: E402
from omot.eval.occlusion import OcclusionSegment  # noqa: E402
from omot.io.mot_format import COL, read_mot  # noqa: E402
from omot.track.bytetrack import iou_matrix  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("render_dev_viz")

ROOT = Path(__file__).resolve().parents[1]
PANEL_H = 540
FPS = 10
PAD_FRAMES = 15
CROP_H = 128
CROP_PAD = 0.15

PALETTE = [(80, 175, 76), (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214),
           (189, 103, 148), (75, 86, 140), (194, 119, 227), (127, 127, 127), (34, 189, 188)]


@dataclass(frozen=True)
class TrackerData:
    active: np.ndarray  # [frame, id, x, y, w, h]
    coasting: np.ndarray


@dataclass(frozen=True)
class Outcome:
    pre_id: int | None
    post_id: int | None
    retained: bool
    category: str  # RETAINED | SWITCHED | POST_NONE | PRE_NONE


def load_tracker(seq: str, tag: str) -> TrackerData:
    trk = read_mot(ROOT / "results" / "raw" / "dev_half" / "trackers" / tag / f"{seq}.txt")
    cpath = ROOT / "results" / "raw" / "dev_half" / "coasting" / tag / f"{seq}.txt"
    coast = read_mot(cpath) if cpath.exists() else np.zeros((0, 9))
    return TrackerData(active=trk[:, :6], coasting=coast[:, :6])


def outcome_for(seg: OcclusionSegment, td: TrackerData) -> Outcome:
    pre = _match_id(td.active, seg.last_visible_frame, seg.last_visible_box, 0.5)
    post = _match_id(td.active, seg.reemergence_frame, seg.reemergence_box, 0.5)
    if pre is None:
        return Outcome(pre, post, False, "PRE_NONE")
    if post is None:
        return Outcome(pre, post, False, "POST_NONE")
    return Outcome(pre, post, pre == post, "RETAINED" if pre == post else "SWITCHED")


def id_color(tid: int) -> tuple[int, int, int]:
    return PALETTE[int(tid) % len(PALETTE)]


def draw_panel(
    img: np.ndarray, frame: int, td: TrackerData, dets: dict[int, np.ndarray],
    gt_rows: np.ndarray, seg: OcclusionSegment, out: Outcome, label: str,
) -> np.ndarray:
    vis = img.copy()
    for d in dets.get(frame, np.zeros((0, 5))):
        x, y, w, h = (int(v) for v in d[:4])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (160, 160, 160), 1)
    gt_here = gt_rows[gt_rows[:, COL.FRAME] == frame]
    for r in gt_here:
        x, y, w, h = (int(v) for v in r[COL.X : COL.H + 1])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 1)

    status = "ABSENT"
    for row in _rows_at(td.active, frame):
        _, tid, x, y, w, h = row[:6]
        x, y, w, h = int(x), int(y), int(w), int(h)
        focal = out.pre_id is not None and int(tid) == out.pre_id
        color = (0, 220, 0) if focal else id_color(int(tid))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3 if focal else 1)
        cv2.putText(vis, str(int(tid)), (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if focal:
            status = f"ACTIVE id {int(tid)}"
    if out.pre_id is not None and status == "ABSENT":
        coast = _rows_at(td.coasting, frame)
        hit = coast[coast[:, 1] == out.pre_id]
        if len(hit):
            x, y, w, h = (int(v) for v in hit[0, 2:6])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(vis, f"coasting {out.pre_id}", (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            status = f"COASTING id {out.pre_id}"

    in_gap = seg.last_visible_frame < frame < seg.reemergence_frame
    banner = f"{label} | f{frame}"
    banner += " | GT OCCLUDED" if in_gap else ""
    banner += f" | focal: {status}"
    if frame >= seg.reemergence_frame:
        verdict = {
            "RETAINED": "RETAINED",
            "SWITCHED": f"SWITCHED -> id {out.post_id}",
            "POST_NONE": "NO DETECTION AT RE-EMERGENCE",
            "PRE_NONE": "NEVER TRACKED PRE-GAP",
        }[out.category]
        banner += f" | {verdict}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (30, 30, 30), -1)
    cv2.putText(vis, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255) if out.category == "RETAINED" or frame < seg.reemergence_frame
                else (60, 60, 255), 1)
    scale = PANEL_H / vis.shape[0]
    return cv2.resize(vis, (int(vis.shape[1] * scale), PANEL_H))


def render_clip(
    seq_name: str, seg: OcclusionSegment, base: TrackerData, pick: TrackerData,
    dets: dict[int, np.ndarray], gt_rows: np.ndarray, dest: Path, mid: int,
    base_label: str, pick_label: str,
) -> None:
    seq = load_sequence(ROOT / "data" / "MOT17" / "train" / seq_name)
    out_b = outcome_for(seg, base)
    out_p = outcome_for(seg, pick)
    f0 = max(1, seg.last_visible_frame - PAD_FRAMES)
    f1 = min(mid, seg.reemergence_frame + PAD_FRAMES)
    tmp = dest.with_suffix(".raw.mp4")
    writer: cv2.VideoWriter | None = None
    for f in range(f0, f1 + 1):
        img = cv2.imread(str(seq.frame_path(f)))
        assert img is not None, f"missing frame {seq.frame_path(f)}"
        left = draw_panel(img, f, base, dets, gt_rows, seg, out_b, base_label)
        right = draw_panel(img, f, pick, dets, gt_rows, seg, out_p, pick_label)
        canvas = np.hstack([left, right])
        if writer is None:
            writer = cv2.VideoWriter(
                str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                (canvas.shape[1], canvas.shape[0]),
            )
        writer.write(canvas)
    assert writer is not None
    writer.release()

    # cv2's mp4v is unplayable in stock Windows players — transcode to H.264.
    import subprocess

    import imageio_ffmpeg

    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
        check=True, timeout=600,
    )
    tmp.unlink()
    log.info("wrote %s (frames %d-%d, gap %d)", dest, f0, f1, seg.gap_length)


def _crop_cell(
    seq_name: str, seg: OcclusionSegment, out: Outcome,
) -> np.ndarray:
    """One grid cell: GT crop at last-visible | GT crop at re-emergence, outcome border."""
    seq = load_sequence(ROOT / "data" / "MOT17" / "train" / seq_name)
    halves = []
    for frame, box in (
        (seg.last_visible_frame, seg.last_visible_box),
        (seg.reemergence_frame, seg.reemergence_box),
    ):
        img = cv2.imread(str(seq.frame_path(frame)))
        assert img is not None, f"missing frame {seq.frame_path(frame)}"
        x, y, w, h = box
        px, py = w * CROP_PAD, h * CROP_PAD
        x0, y0 = max(0, int(x - px)), max(0, int(y - py))
        x1 = min(img.shape[1], int(x + w + px))
        y1 = min(img.shape[0], int(y + h + py))
        halves.append(cv2.resize(img[y0:y1, x0:x1], (64, CROP_H)))
    cell = np.hstack([halves[0], np.full((CROP_H, 2, 3), 255, np.uint8), halves[1]])
    color = (0, 160, 0) if out.category == "RETAINED" else (0, 0, 200)
    cell = cv2.copyMakeBorder(cell, 3, 14, 3, 3, cv2.BORDER_CONSTANT, value=color)
    cv2.putText(cell, f"gap {seg.gap_length}", (6, cell.shape[0] - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return cell


def render_crop_grid(
    per_seq: dict[str, dict], dest: Path, pick_label: str, max_cells: int = 36,
) -> None:
    """Pre/post GT crops for assoc-scope segments, bordered by the pick's outcome.

    Green = retained, red = switched — the re-ID qualitative read-out: does the
    re-emerged appearance visibly match the pre-gap appearance where we fail?
    """
    scoped: list[tuple[str, OcclusionSegment, Outcome]] = []
    for seq_name, ctx in per_seq.items():
        for s in ctx["segs"]:
            op = outcome_for(s, ctx["pick"])
            if op.category in ("RETAINED", "SWITCHED"):
                scoped.append((seq_name, s, op))
    n_ret = sum(o.category == "RETAINED" for _, _, o in scoped)
    assert scoped, "no assoc-scope segments — wrong tag?"
    scoped.sort(key=lambda t: (t[2].category != "SWITCHED", -t[1].gap_length))
    cells = [_crop_cell(sq, s, o) for sq, s, o in scoped[:max_cells]]
    cols = 6
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i : i + cols]
        row += [np.zeros_like(cells[0])] * (cols - len(row))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    header = np.full((34, grid.shape[1], 3), 25, np.uint8)
    cv2.putText(
        header,
        f"{pick_label}: pre|post crops - retained {n_ret}/{len(scoped)}"
        f" (top {min(max_cells, len(scoped))}, switched first)",
        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )
    cv2.imwrite(str(dest), np.vstack([header, grid]))
    log.info("wrote %s (%d cells)", dest, min(max_cells, len(scoped)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-tag", default="hidden_audit_base")
    ap.add_argument("--pick-tag", default="hidden_d26best")
    ap.add_argument("--base-label", default="BASELINE")
    ap.add_argument("--pick-label", default="PICK")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "viz")
    ap.add_argument("--pngs-only", action="store_true",
                    help="skip MP4 clip rendering (cheap PNG iteration)")
    args = ap.parse_args()
    viz_dir: Path = args.out_dir
    viz_dir.mkdir(exist_ok=True)
    seg_index = json.loads(
        (ROOT / "results" / "occlusion_segments.json").read_text(encoding="utf-8")
    )

    per_seq: dict[str, dict] = {}
    rows_for_grid: list[tuple[str, Outcome, Outcome, int]] = []
    for seq_name, entry in seg_index["sequences"].items():
        if not entry["dev_half"]:
            continue
        base = load_tracker(seq_name, args.base_tag)
        pick = load_tracker(seq_name, args.pick_tag)
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
        per_seq[seq_name] = {
            "base": base, "pick": pick, "dets": dets, "gt": gt, "segs": segs,
            "mid": int(entry["seq_length"]) // 2,
        }
        for s in segs:
            rows_for_grid.append(
                (seq_name, outcome_for(s, base), outcome_for(s, pick), s.gap_length)
            )

    # ---- clip selection -------------------------------------------------------------
    wins: list[tuple[str, OcclusionSegment]] = []
    pick_fail: list[tuple[str, OcclusionSegment]] = []
    det_ceiling: list[tuple[str, OcclusionSegment]] = []
    for seq_name, ctx in per_seq.items():
        for s in ctx["segs"]:
            ob, op = outcome_for(s, ctx["base"]), outcome_for(s, ctx["pick"])
            if op.category == "RETAINED" and not ob.retained and s.gap_length >= 12:
                wins.append((seq_name, s))
            elif op.category == "SWITCHED" and s.gap_length >= 10:
                pick_fail.append((seq_name, s))
            elif op.category == "POST_NONE" and ob.category == "POST_NONE":
                gt_box = s.reemergence_box.reshape(1, 4)
                frame_dets = ctx["dets"].get(s.reemergence_frame, np.zeros((0, 5)))
                no_det = (
                    not len(frame_dets)
                    or float(iou_matrix(gt_box, frame_dets[:, :4]).max()) < 0.3
                )
                if no_det and s.gap_length >= 8:
                    det_ceiling.append((seq_name, s))

    wins.sort(key=lambda t: -t[1].gap_length)
    chosen_wins = []
    used_seqs: set[str] = set()
    for seq_name, s in wins:  # prefer distinct sequences
        if seq_name not in used_seqs or len(chosen_wins) >= len(per_seq):
            chosen_wins.append((seq_name, s))
            used_seqs.add(seq_name)
        if len(chosen_wins) == 2:
            break
    pick_fail.sort(key=lambda t: -t[1].gap_length)
    det_ceiling.sort(key=lambda t: -t[1].gap_length)
    log.info("candidates: wins=%d pick_fail=%d det_ceiling=%d",
             len(wins), len(pick_fail), len(det_ceiling))
    assert chosen_wins and pick_fail and det_ceiling, "selection came up empty somewhere"

    mpath = viz_dir / "manifest.json"
    manifest: dict[str, str] = (
        json.loads(mpath.read_text(encoding="utf-8")) if args.pngs_only and mpath.exists()
        else {}
    )
    manifest.update({"base_tag": args.base_tag, "pick_tag": args.pick_tag})
    chosen_wins = [] if args.pngs_only else chosen_wins
    for i, (seq_name, s) in enumerate(chosen_wins, 1):
        dest = viz_dir / f"a{i}_pick_retains_{seq_name}_t{s.track_id}_gap{s.gap_length}.mp4"
        ctx = per_seq[seq_name]
        render_clip(seq_name, s, ctx["base"], ctx["pick"], ctx["dets"], ctx["gt"],
                    dest, ctx["mid"], args.base_label, args.pick_label)
        manifest[f"a{i}_pick_retains"] = str(dest)
    clip_pools = () if args.pngs_only else (
        ("b_pick_tracker_failure", pick_fail), ("c_detector_ceiling", det_ceiling),
    )
    for key, pool in clip_pools:
        seq_name, s = pool[0]
        dest = viz_dir / f"{key}_{seq_name}_t{s.track_id}_gap{s.gap_length}.mp4"
        ctx = per_seq[seq_name]
        render_clip(seq_name, s, ctx["base"], ctx["pick"], ctx["dets"], ctx["gt"],
                    dest, ctx["mid"], args.base_label, args.pick_label)
        manifest[key] = str(dest)

    crop_png = viz_dir / "crops_assoc_scope.png"
    render_crop_grid(per_seq, crop_png, args.pick_label)
    manifest["crops_png"] = str(crop_png)

    # ---- summary PNG ----------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cat_code = {"RETAINED": 0, "SWITCHED": 1, "POST_NONE": 2, "PRE_NONE": 3}
    grid = np.array(
        [[cat_code[b.category], cat_code[p.category]] for _, b, p, _ in rows_for_grid]
    )
    cmap = ListedColormap(["#2e7d32", "#c62828", "#9e9e9e", "#424242"])
    fig, ax = plt.subplots(figsize=(7.5, max(8, len(grid) * 0.11)))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xticks([0, 1], [args.base_label, args.pick_label])
    boundaries, names, start = [], [], 0
    for seq_name, ctx in per_seq.items():
        n = len(ctx["segs"])
        boundaries.append(start + n - 0.5)
        names.append((seq_name.replace("MOT17-", "").replace("-FRCNN", ""), start + n / 2))
        start += n
    for b in boundaries[:-1]:
        ax.axhline(b, color="white", lw=1.2)
    for nm, pos in names:
        ax.text(-0.62, pos, nm, va="center", ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_title(
        f"Dev-half occlusion segments ({len(grid)}): retention outcome\n"
        f"baseline retention {np.mean(grid[:, 0] == 0):.1%} -> pick {np.mean(grid[:, 1] == 0):.1%}"
    )
    ax.legend(handles=[
        Patch(color="#2e7d32", label="retained"),
        Patch(color="#c62828", label="switched (tracker-side)"),
        Patch(color="#9e9e9e", label="no detection at re-emergence"),
        Patch(color="#424242", label="never tracked pre-gap"),
    ], loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    png = viz_dir / "summary_retention_grid.png"
    fig.savefig(png, dpi=150)
    manifest["summary_png"] = str(png)
    log.info("wrote %s", png)

    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for k, v in manifest.items():
        log.info("%s: %s", k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
