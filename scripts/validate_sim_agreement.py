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

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("validate_sim_agreement")

ROOT = Path(__file__).resolve().parents[1]
CENTER_TOL = 0.6  # |center delta| / mock box height, median over shared annotations
CORR_MIN = 0.5  # visibility correlation gate — applied to behind_static ONLY: there the
# mock's slab-occluder model is trustworthy. In walker-on-walker scenes (crossing/crowd)
# the mock knowingly overestimates coverage (bbox slabs vs true silhouettes), so corr is
# reported but not gated. Instead, ALL scenarios face absolute sim-GT sanity checks:
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
        m_by, s_by = per_walker(mock.gt), per_walker(sim.gt)
        center_errs: list[float] = []
        vis_pairs: list[tuple[float, float]] = []
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
                vis_pairs.append((float(mr[COL.VIS]), float(sr[COL.VIS])))

        med_center = float(np.median(center_errs)) if center_errs else float("inf")
        vis_arr = np.array(vis_pairs)
        dip_mask = (vis_arr < 0.8).any(axis=1) if len(vis_arr) else np.zeros(0, bool)
        if dip_mask.sum() >= 8 and np.std(vis_arr[:, 0]) > 1e-3 and np.std(vis_arr[:, 1]) > 1e-3:
            corr = float(np.corrcoef(vis_arr[:, 0], vis_arr[:, 1])[0, 1])
        else:
            corr = float("nan")  # no meaningful occlusion overlap to correlate
        n_seg_mock = len(extract_segments(mock.gt, min_len=3))
        n_seg_sim = len(extract_segments(sim.gt, min_len=3))
        sim_vis = sim.gt[:, COL.VIS]
        p75_vis = float(np.percentile(sim_vis, 75)) if len(sim_vis) else 0.0
        frac_deep = float(np.mean(sim_vis < 0.25)) if len(sim_vis) else 1.0
        entry = {
            "shared_annotations": len(center_errs),
            "center_err_med_boxnorm": round(med_center, 3),
            "visibility_corr": None if np.isnan(corr) else round(corr, 3),
            "segments_mock": n_seg_mock,
            "segments_sim": n_seg_sim,
            "sim_p75_vis": round(p75_vis, 3),
            "sim_frac_deep_occl": round(frac_deep, 3),
        }
        report[name] = entry
        log.info("%s: %s", name, entry)
        if med_center > CENTER_TOL:
            failures.append(f"{name}: center err {med_center:.2f} > {CENTER_TOL}")
        if name.startswith("behind_static") and not np.isnan(corr) and corr < CORR_MIN:
            failures.append(f"{name}: visibility corr {corr:.2f} < {CORR_MIN}")
        if p75_vis < SIM_P75_VIS_MIN:
            failures.append(f"{name}: sim p75 vis {p75_vis:.2f} < {SIM_P75_VIS_MIN}")
        if frac_deep > SIM_FRAC_DEEP_MAX:
            failures.append(f"{name}: sim deep-occl fraction {frac_deep:.2f} > {SIM_FRAC_DEEP_MAX}")

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
