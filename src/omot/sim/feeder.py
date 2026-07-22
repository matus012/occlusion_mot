"""Scenario -> MOT17-style sequence rendering (P2, gate G4).

Backends:
  MockBackend  — analytic pinhole projection + rasterized visibility; renders schematic
                 frames (ground grid, occluder boxes, walker rectangles) and exact
                 MOT-format GT with per-frame visibility. No simulator needed; used by
                 tests, the G4 demo path, and as the reference implementation the CARLA
                 backend must agree with on projection/visibility math.
  CarlaBackend — drives the CARLA sim via a dedicated py3.10 venv subprocess (client
                 wheels ship for <=3.10; see D22). Stub until the sim env is provisioned.

Output layout per scenario (loadable by omot.data.mot.load_sequence):
  <out>/<scenario_id>/img1/000001.jpg...   gt/gt.txt   seqinfo.ini   scenario.json
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from omot.io.mot_format import N_COLS, write_mot
from omot.sim.scenario import OcclusionScenario, WalkerSpec

logger = logging.getLogger(__name__)

_SEQINFO = """[Sequence]
name={name}
imDir=img1
frameRate={fps}
seqLength={length}
imWidth={w}
imHeight={h}
imExt=.jpg
"""

BODY_HALF_W = 0.25  # meters
MIN_DEPTH = 1.5  # meters; closer than this -> no annotation
VIS_GRID = (48, 24)  # rows, cols rasterization of a walker box for visibility


@dataclass(frozen=True)
class _Cam:
    x: float
    y: float
    z: float
    f: float  # focal in pixels
    w: int
    h: int

    @staticmethod
    def from_scenario(s: OcclusionScenario) -> _Cam:
        f = (s.img_w / 2) / np.tan(np.deg2rad(s.cam_fov_deg) / 2)
        return _Cam(s.cam_pos[0], s.cam_pos[1], s.cam_pos[2], float(f), s.img_w, s.img_h)


def _walker_pos(w: WalkerSpec, t: float) -> tuple[float, float]:
    """Position along the piecewise-linear path at time t (clamped at the end)."""
    pts = np.asarray(w.waypoints, dtype=np.float64)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = min(w.speed * t, float(seg_len.sum()))
    for i, sl in enumerate(seg_len):
        if dist <= sl or i == len(seg_len) - 1:
            r = 0.0 if sl == 0 else min(dist / sl, 1.0)
            p = pts[i] + r * (pts[i + 1] - pts[i])
            return float(p[0]), float(p[1])
        dist -= sl
    return float(pts[-1][0]), float(pts[-1][1])


def _project_box(cam: _Cam, x: float, y: float, height: float) -> np.ndarray | None:
    """World upright body at (x, y) -> image tlwh, or None if behind/too close."""
    depth = x - cam.x
    if depth < MIN_DEPTH:
        return None
    u = cam.w / 2 + cam.f * (y - cam.y) / depth
    v_top = cam.h / 2 + cam.f * (cam.z - height) / depth
    v_bot = cam.h / 2 + cam.f * (cam.z - 0.0) / depth
    half_w_px = cam.f * BODY_HALF_W / depth
    box = np.array([u - half_w_px, v_top, 2 * half_w_px, v_bot - v_top])
    if box[0] + box[2] < 0 or box[0] > cam.w or box[1] + box[3] < 0 or box[1] > cam.h:
        return None
    return box


def _occluder_screen_rect(
    cam: _Cam, occ: tuple[float, float, float, float, float]
) -> tuple[float, np.ndarray] | None:
    """Occluder box -> (depth, image tlwh) using its front face; None if behind camera."""
    cx, cy, half_w, half_d, height = occ
    depth = (cx - half_d) - cam.x  # front face
    if depth < MIN_DEPTH:
        return None
    u_l = cam.w / 2 + cam.f * ((cy - half_w) - cam.y) / depth
    u_r = cam.w / 2 + cam.f * ((cy + half_w) - cam.y) / depth
    v_top = cam.h / 2 + cam.f * (cam.z - height) / depth
    v_bot = cam.h / 2 + cam.f * cam.z / depth
    return depth, np.array([u_l, v_top, u_r - u_l, v_bot - v_top])


def _coverage(target: np.ndarray, blockers: list[np.ndarray]) -> float:
    """Fraction of `target` (tlwh) covered by union of `blockers`, on a raster grid."""
    if not blockers:
        return 0.0
    rows, cols = VIS_GRID
    ys = target[1] + (np.arange(rows) + 0.5) / rows * target[3]
    xs = target[0] + (np.arange(cols) + 0.5) / cols * target[2]
    gx, gy = np.meshgrid(xs, ys)
    covered = np.zeros_like(gx, dtype=bool)
    for b in blockers:
        covered |= (gx >= b[0]) & (gx <= b[0] + b[2]) & (gy >= b[1]) & (gy <= b[1] + b[3])
    return float(covered.mean())


class MockBackend:
    """Analytic renderer — schematic frames + exact GT with visibility."""

    def render(self, scenario: OcclusionScenario, out_dir: Path) -> Path:
        import cv2

        cam = _Cam.from_scenario(scenario)
        seq_dir = out_dir / scenario.scenario_id
        (seq_dir / "img1").mkdir(parents=True, exist_ok=True)
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(scenario.seed)
        colors = {
            w.walker_id: tuple(int(c) for c in rng.integers(60, 230, 3))
            for w in scenario.walkers
        }

        occ_rects = []
        for occ in scenario.occluders:
            pr = _occluder_screen_rect(cam, occ)
            if pr is not None:
                occ_rects.append(pr)

        gt_rows: list[np.ndarray] = []
        for fidx in range(1, scenario.n_frames + 1):
            t = (fidx - 1) / scenario.fps
            img = np.full((cam.h, cam.w, 3), 235, dtype=np.uint8)
            for gy in range(0, cam.w, 80):  # schematic ground grid
                cv2.line(img, (gy, cam.h // 2), (gy, cam.h), (210, 210, 210), 1)

            # gather walker states this frame, back-to-front for painter's algorithm
            states: list[tuple[float, int, np.ndarray, float]] = []
            for w in scenario.walkers:
                x, y = _walker_pos(w, t)
                box = _project_box(cam, x, y, w.height)
                if box is None:
                    continue
                depth = x - cam.x
                states.append((depth, w.walker_id, box, x))
            states.sort(key=lambda s: -s[0])

            for depth, wid, box, _x in states:
                blockers = [r for d, r in occ_rects if d < depth]
                blockers += [b for d2, _wid2, b, _x2 in states if d2 < depth]
                vis = 1.0 - _coverage(box, blockers)
                row = np.zeros(N_COLS)
                row[:6] = [fidx, wid, *box]
                row[6:] = [1.0, 1.0, round(vis, 4)]
                gt_rows.append(row)
                x1, y1 = int(box[0]), int(box[1])
                x2, y2 = int(box[0] + box[2]), int(box[1] + box[3])
                cv2.rectangle(img, (x1, y1), (x2, y2), colors[wid], -1)

            for _d, r in sorted(occ_rects, key=lambda p: -p[0]):
                x1, y1 = int(r[0]), int(r[1])
                x2, y2 = int(r[0] + r[2]), int(r[1] + r[3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (128, 128, 128), -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), 2)

            cv2.imwrite(str(seq_dir / "img1" / f"{fidx:06d}.jpg"), img)

        write_mot(seq_dir / "gt" / "gt.txt", np.stack(gt_rows) if gt_rows else np.zeros((0, 9)))
        (seq_dir / "seqinfo.ini").write_text(
            _SEQINFO.format(name=scenario.scenario_id, fps=scenario.fps,
                            length=scenario.n_frames, w=cam.w, h=cam.h),
            encoding="utf-8",
        )
        (seq_dir / "scenario.json").write_text(scenario.to_json(), encoding="utf-8")
        logger.info("%s: rendered %d frames, %d gt rows",
                    scenario.scenario_id, scenario.n_frames, len(gt_rows))
        return seq_dir


class CarlaBackend:
    """Real-sim rendering via a dedicated py3.10 client venv (D22). Not yet provisioned."""

    def render(self, scenario: OcclusionScenario, out_dir: Path) -> Path:
        raise NotImplementedError(
            "CARLA backend requires the sim venv (scripts/setup_carla.ps1, D22); "
            "the MockBackend is the reference implementation until then."
        )


def render_scenario(
    scenario: OcclusionScenario, out_dir: Path, backend: MockBackend | CarlaBackend | None = None
) -> Path:
    backend = backend or MockBackend()
    return backend.render(scenario, out_dir)
