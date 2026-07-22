"""Occlusion scenario specifications (P2). Pure data — backend-agnostic.

A scenario describes pedestrians (walkers) with piecewise-linear world paths, one or
more box occluders, and a fixed camera. Templates parameterize the archetypal occlusion
events measured on MOT17 (D14/D16: median gap 25 frames, p90 ~77): behind-static,
crossing-paths, and crowd-merge.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

TEMPLATES = ("behind_static", "crossing_paths", "crowd_merge")


@dataclass(frozen=True)
class WalkerSpec:
    walker_id: int
    waypoints: list[tuple[float, float]]  # world-plane (x, y) meters, piecewise-linear
    speed: float  # m/s
    height: float = 1.75  # meters


@dataclass(frozen=True)
class OcclusionScenario:
    scenario_id: str
    template: str
    seed: int
    duration_s: float
    fps: int
    walkers: list[WalkerSpec]
    # box occluders on the world plane: (cx, cy, half_w, half_d, height_m)
    occluders: list[tuple[float, float, float, float, float]]
    # camera: position (x, y, z), yaw deg, horizontal FOV deg, image size
    cam_pos: tuple[float, float, float]
    cam_yaw_deg: float
    cam_fov_deg: float = 90.0
    img_w: int = 1280
    img_h: int = 720

    @property
    def n_frames(self) -> int:
        return int(self.duration_s * self.fps)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> OcclusionScenario:
        d = json.loads(text)
        d["walkers"] = [WalkerSpec(**w) for w in d["walkers"]]
        d["occluders"] = [tuple(o) for o in d["occluders"]]
        d["cam_pos"] = tuple(d["cam_pos"])
        return OcclusionScenario(**d)


@dataclass(frozen=True)
class _TemplateParams:
    n_walkers: tuple[int, int]
    n_occluders: tuple[int, int]


_TEMPLATE_PARAMS: dict[str, _TemplateParams] = {
    "behind_static": _TemplateParams(n_walkers=(3, 6), n_occluders=(1, 3)),
    "crossing_paths": _TemplateParams(n_walkers=(4, 8), n_occluders=(0, 1)),
    "crowd_merge": _TemplateParams(n_walkers=(6, 10), n_occluders=(0, 2)),
}


def _make_walker(rng: np.random.Generator, wid: int, crossing: bool) -> WalkerSpec:
    # Walk across the camera frustum (camera looks along +x from origin area).
    y0 = float(rng.uniform(-8, 8))
    x0 = float(rng.uniform(8, 22))
    if crossing:
        # start left or right, cross to the other side at roughly constant depth
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        start = (x0, side * float(rng.uniform(6, 10)))
        end = (x0 + float(rng.uniform(-2, 2)), -side * float(rng.uniform(6, 10)))
    else:
        start = (x0, y0)
        end = (x0 + float(rng.uniform(-3, 3)), y0 + float(rng.uniform(-10, 10)))
    mid = (
        (start[0] + end[0]) / 2 + float(rng.uniform(-1, 1)),
        (start[1] + end[1]) / 2 + float(rng.uniform(-1, 1)),
    )
    return WalkerSpec(
        walker_id=wid,
        waypoints=[start, mid, end],
        speed=float(rng.uniform(0.8, 1.8)),
        height=float(rng.uniform(1.6, 1.9)),
    )


def make_scenario(
    template: str, seed: int, duration_s: float = 20.0, fps: int = 20
) -> OcclusionScenario:
    assert template in TEMPLATES, f"unknown template {template!r}"
    rng = np.random.default_rng(seed)
    params = _TEMPLATE_PARAMS[template]
    n_walk = int(rng.integers(params.n_walkers[0], params.n_walkers[1] + 1))
    n_occ = int(rng.integers(params.n_occluders[0], params.n_occluders[1] + 1))
    walkers = [
        _make_walker(rng, wid + 1, crossing=template != "behind_static")
        for wid in range(n_walk)
    ]
    occluders = [
        (
            float(rng.uniform(10, 18)),  # cx: mid-frustum depth
            float(rng.uniform(-4, 4)),  # cy
            float(rng.uniform(0.8, 2.0)),  # half width
            float(rng.uniform(0.4, 1.0)),  # half depth
            float(rng.uniform(1.8, 2.6)),  # height
        )
        for _ in range(n_occ)
    ]
    return OcclusionScenario(
        scenario_id=f"{template}_{seed:04d}",
        template=template,
        seed=seed,
        duration_s=duration_s,
        fps=fps,
        walkers=walkers,
        occluders=occluders,
        cam_pos=(0.0, 0.0, 2.5),  # ~eye level; a high cam with no pitch culls near walkers
        cam_yaw_deg=0.0,
    )


def generate_scenarios(n: int, seed: int = 0) -> list[OcclusionScenario]:
    """Round-robin the templates; deterministic in (n, seed)."""
    out = [
        make_scenario(TEMPLATES[i % len(TEMPLATES)], seed=seed * 1000 + i)
        for i in range(n)
    ]
    logger.info("generated %d scenario specs", n)
    return out


def save_scenarios(scenarios: list[OcclusionScenario], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in scenarios:
        (out_dir / f"{s.scenario_id}.json").write_text(s.to_json(), encoding="utf-8")
