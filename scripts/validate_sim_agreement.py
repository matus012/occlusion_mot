"""Mock-vs-CARLA agreement validation on shared scenario specs (phase 5b).

For each scenario rendered by BOTH backends, per (walker, frame) with annotations in
both: compare box centers (normalized by mock box height) and visibility profiles.
Agreement thresholds are loose by design — walkers have real silhouettes vs the mock's
0.5 m slab, and CARLA occluder props approximate the spec boxes — the check exists to
catch sign/axis/timing bugs, not to demand pixel equality.

Writes results/sim_agreement.json; exit 1 on any hard disagreement.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import load_sequence  # noqa: E402
from omot.eval.occlusion import extract_segments  # noqa: E402
from omot.io.mot_format import COL  # noqa: E402
from omot.sim.feeder import _Cam, _coverage, _occluder_screen_rect  # noqa: E402
from omot.sim.scenario import OcclusionScenario  # noqa: E402


def walker_only_vis_and_shadow(
    mock_gt: np.ndarray, scenario: OcclusionScenario
) -> dict[tuple[int, int], tuple[float, bool]]:
    """(frame, wid) -> (visibility predicted from OTHER WALKERS only, in_static_shadow).

    Walker trajectories/boxes are identical across backends (deterministic paths), so
    inter-walker coverage is exact shared physics; static occluders are excluded and
    their (inflated) screen shadows marked so gating can skip those frames.
    """
    cam = _Cam.from_scenario(scenario)
    shadows = []
    for occ in scenario.occluders:
        pr = _occluder_screen_rect(cam, occ)
        if pr is not None:
            _, r = pr
            cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
            w, h = r[2] * SHADOW_MARGIN, r[3] * SHADOW_MARGIN
            shadows.append(np.array([cx - w / 2, cy - h / 2, w, h]))

    out: dict[tuple[int, int], tuple[float, bool]] = {}
    for frame in np.unique(mock_gt[:, COL.FRAME]).astype(int):
        rows = mock_gt[mock_gt[:, COL.FRAME] == frame]
        boxes = {int(r[COL.ID]): r[COL.X : COL.H + 1] for r in rows}
        # nearer == larger box height (monotone in 1/depth for equal-height walkers)
        for wid, box in boxes.items():
            blockers = [b for w2, b in boxes.items() if w2 != wid and b[3] > box[3]]
            vis = 1.0 - _coverage(box, blockers)
            in_shadow = any(
                not (box[0] + box[2] < s[0] or box[0] > s[0] + s[2]
                     or box[1] + box[3] < s[1] or box[1] > s[1] + s[3])
                for s in shadows
            )
            out[(frame, wid)] = (vis, in_shadow)
    return out

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("validate_sim_agreement")

ROOT = Path(__file__).resolve().parents[1]
CENTER_TOL = 0.6  # |center delta| / mock box height, median over shared annotations
# Visibility gate: cross-backend correlation is structurally unsound wherever static
# occluder PROPS differ from the spec slabs (a walker lingering in a spec-box shadow is
# "always hidden" to the mock and visible in the sim). Instead we gate on the physics
# both backends share exactly — inter-walker occlusion: corr(sim_vis, walker-only
# predicted vis) over frames OUTSIDE static-occluder shadows (spec rect + margin).
# Two-level gate: the walker-only prediction still uses bbox SLABS, which overestimate
# coverage in dense crowds — a noise floor, not a bug signal. Sign/axis/timing bugs
# produce corr ~0 or negative (observed pre-fix); slab noise bottoms out near ~0.45.
WALKER_CORR_HARD_MIN = 0.35  # per-scenario: below this = real pipeline bug
WALKER_CORR_MEDIAN_MIN = 0.5  # across scenarios: guards systematic degradation
MIN_DIP_FRAMES = 8  # need this many shadow-free annotations with a predicted dip to gate
SHADOW_MARGIN = 1.3  # inflate spec occluder screen rects by this factor
SIM_P75_VIS_MIN = 0.85  # p75 of visibility — most annotations should be mostly visible
SIM_FRAC_DEEP_MAX = 0.45  # fraction of annotations with vis < 0.25


def per_walker(gt: np.ndarray) -> dict[int, np.ndarray]:
    return {int(t): gt[gt[:, COL.ID] == t] for t in np.unique(gt[:, COL.ID])}


def main() -> int:
    mock_root = ROOT / "data" / "sim" / "scenarios"
    carla_root = ROOT / "data" / "sim" / "carla_render"
    shared = sorted(
        p.name for p in carla_root.iterdir()
        if (p / "gt" / "gt.txt").exists() and (mock_root / p.name / "gt" / "gt.txt").exists()
    )
    if not shared:
        raise SystemExit("no scenarios rendered by both backends")

    report: dict[str, dict] = {}
    failures: list[str] = []
    for name in shared:
        mock = load_sequence(mock_root / name)
        sim = load_sequence(carla_root / name)
        assert mock.gt is not None and sim.gt is not None
        scenario = OcclusionScenario.from_json(
            (carla_root / name / "scenario.json").read_text(encoding="utf-8")
        )
        predicted = walker_only_vis_and_shadow(mock.gt, scenario)
        m_by, s_by = per_walker(mock.gt), per_walker(sim.gt)
        center_errs: list[float] = []
        vis_pairs: list[tuple[float, float]] = []  # (walker-only predicted, sim) shadow-free
        for wid, m_rows in m_by.items():
            s_rows = s_by.get(wid)
            if s_rows is None:
                continue
            s_idx = {int(r[COL.FRAME]): r for r in s_rows}
            for mr in m_rows:
                sr = s_idx.get(int(mr[COL.FRAME]))
                if sr is None:
                    continue
                mc = mr[COL.X : COL.H + 1]
                sc = sr[COL.X : COL.H + 1]
                m_center = np.array([mc[0] + mc[2] / 2, mc[1] + mc[3] / 2])
                s_center = np.array([sc[0] + sc[2] / 2, sc[1] + sc[3] / 2])
                center_errs.append(float(np.linalg.norm(m_center - s_center) / max(mc[3], 1)))
                pred_vis, in_shadow = predicted.get(
                    (int(mr[COL.FRAME]), wid), (1.0, False)
                )
                if not in_shadow:
                    vis_pairs.append((pred_vis, float(sr[COL.VIS])))

        med_center = float(np.median(center_errs)) if center_errs else float("inf")
        vis_arr = np.array(vis_pairs)
        dip_mask = (vis_arr[:, 0] < 0.8) if len(vis_arr) else np.zeros(0, bool)
        if (
            dip_mask.sum() >= MIN_DIP_FRAMES
            and np.std(vis_arr[:, 0]) > 1e-3
            and np.std(vis_arr[:, 1]) > 1e-3
        ):
            corr = float(np.corrcoef(vis_arr[:, 0], vis_arr[:, 1])[0, 1])
        else:
            corr = float("nan")  # not enough shadow-free inter-walker dips to gate
        n_seg_mock = len(extract_segments(mock.gt, min_len=3))
        n_seg_sim = len(extract_segments(sim.gt, min_len=3))
        sim_vis = sim.gt[:, COL.VIS]
        p75_vis = float(np.percentile(sim_vis, 75)) if len(sim_vis) else 0.0
        frac_deep = float(np.mean(sim_vis < 0.25)) if len(sim_vis) else 1.0
        entry = {
            "shared_annotations": len(center_errs),
            "center_err_med_boxnorm": round(med_center, 3),
            "walker_vis_corr": None if np.isnan(corr) else round(corr, 3),
            "n_shadowfree_dips": int(dip_mask.sum()),
            "segments_mock": n_seg_mock,
            "segments_sim": n_seg_sim,
            "sim_p75_vis": round(p75_vis, 3),
            "sim_frac_deep_occl": round(frac_deep, 3),
        }
        report[name] = entry
        log.info("%s: %s", name, entry)
        if med_center > CENTER_TOL:
            failures.append(f"{name}: center err {med_center:.2f} > {CENTER_TOL}")
        if not np.isnan(corr) and corr < WALKER_CORR_HARD_MIN:
            failures.append(f"{name}: walker-vis corr {corr:.2f} < {WALKER_CORR_HARD_MIN}")
        if p75_vis < SIM_P75_VIS_MIN:
            failures.append(f"{name}: sim p75 vis {p75_vis:.2f} < {SIM_P75_VIS_MIN}")
        if frac_deep > SIM_FRAC_DEEP_MAX:
            failures.append(f"{name}: sim deep-occl fraction {frac_deep:.2f} > {SIM_FRAC_DEEP_MAX}")

    corrs = [
        e["walker_vis_corr"] for e in report.values() if e["walker_vis_corr"] is not None
    ]
    if corrs and float(np.median(corrs)) < WALKER_CORR_MEDIAN_MIN:
        failures.append(
            f"median walker-vis corr {np.median(corrs):.2f} < {WALKER_CORR_MEDIAN_MIN}"
        )

    dest = ROOT / "results" / "sim_agreement.json"
    dest.write_text(
        json.dumps({"scenarios": report, "failures": failures}, indent=2), encoding="utf-8"
    )
    log.info("%d scenarios compared, %d failures -> %s", len(shared), len(failures), dest)
    for f in failures:
        log.info("  FAIL: %s", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
