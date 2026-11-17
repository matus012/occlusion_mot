"""Synthetic multi-agent scene generator — dataset-free testing of the full pipeline.

Agents move with constant velocity across the image. An optional occluder rectangle
suppresses detections (visibility 0) while the agent's center is inside it, which
creates ground-truth occlusion segments with known re-emergence points/times.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from omot.io.mot_format import N_COLS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthConfig:
    width: int = 1280
    height: int = 720
    n_frames: int = 120
    n_agents: int = 6
    seed: int = 0
    box_w: float = 60.0
    box_h: float = 120.0
    det_noise_px: float = 2.0  # gaussian jitter on detection corners
    p_low_score: float = 0.15  # fraction of detections downgraded to low score
    p_false_positive: float = 0.05  # per-frame chance of a spurious detection
    occluder: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2
    # spawn/velocity ranges — pin these in tests that need guaranteed occluder crossings
    x0_range: tuple[float, float] = (-60.0, 320.0)
    vx_range: tuple[float, float] = (3.0, 7.0)
    vy_range: tuple[float, float] = (-1.0, 1.0)


@dataclass(frozen=True)
class SynthScene:
    config: SynthConfig
    gt: np.ndarray  # (N, 9) MOT rows: frame, id, x, y, w, h, 1, 1, vis
    detections: dict[int, np.ndarray] = field(repr=False)  # frame -> (M, 5) tlwh+score

    @property
    def n_frames(self) -> int:
        return self.config.n_frames


def _center_in(rect: tuple[float, float, float, float], cx: float, cy: float) -> bool:
    return rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]


def generate(cfg: SynthConfig) -> SynthScene:
    """Deterministic (seeded) scene: GT + noisy detections per frame."""
    rng = np.random.default_rng(cfg.seed)
    # Spawn along the left edge with rightward velocity; vertical spread + slight drift.
    x0 = rng.uniform(cfg.x0_range[0], cfg.x0_range[1], cfg.n_agents)
    y0 = rng.uniform(0, cfg.height - cfg.box_h, cfg.n_agents)
    vx = rng.uniform(cfg.vx_range[0], cfg.vx_range[1], cfg.n_agents)
    vy = rng.uniform(cfg.vy_range[0], cfg.vy_range[1], cfg.n_agents)

    gt_rows: list[np.ndarray] = []
    detections: dict[int, np.ndarray] = {}
    for f in range(1, cfg.n_frames + 1):
        frame_dets: list[np.ndarray] = []
        for a in range(cfg.n_agents):
            x = x0[a] + vx[a] * (f - 1)
            y = np.clip(y0[a] + vy[a] * (f - 1), 0, cfg.height - cfg.box_h)
            if x + cfg.box_w < 0 or x > cfg.width:
                continue
            cx, cy = x + cfg.box_w / 2, y + cfg.box_h / 2
            occluded = cfg.occluder is not None and _center_in(cfg.occluder, cx, cy)
            vis = 0.0 if occluded else 1.0
            row = np.zeros(N_COLS)
            row[:6] = [f, a + 1, x, y, cfg.box_w, cfg.box_h]
            row[6:] = [1.0, 1.0, vis]
            gt_rows.append(row)
            if occluded:
                continue
            jitter = rng.normal(0, cfg.det_noise_px, 4)
            score = (
                float(np.clip(rng.normal(0.35, 0.08), 0.15, 0.55))
                if rng.uniform() < cfg.p_low_score
                else float(np.clip(rng.normal(0.85, 0.05), 0.6, 0.99))
            )
            frame_dets.append(
                np.array(
                    [x + jitter[0], y + jitter[1],
                     cfg.box_w + jitter[2], cfg.box_h + jitter[3], score]
                )
            )
        if rng.uniform() < cfg.p_false_positive:
            fx = rng.uniform(0, cfg.width - cfg.box_w)
            fy = rng.uniform(0, cfg.height - cfg.box_h)
            frame_dets.append(np.array([fx, fy, cfg.box_w, cfg.box_h, rng.uniform(0.15, 0.45)]))
        detections[f] = (
            np.stack(frame_dets) if frame_dets else np.zeros((0, 5), dtype=np.float64)
        )

    gt = np.stack(gt_rows) if gt_rows else np.zeros((0, N_COLS))
    logger.info(
        "synth scene: %d frames, %d agents, %d gt rows", cfg.n_frames, cfg.n_agents, len(gt)
    )
    return SynthScene(config=cfg, gt=gt, detections=detections)
