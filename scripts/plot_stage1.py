"""Stage-1 dose-response figure (perun_detector_v1.md section 3, D61)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector_launcher import ols_with_ci  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("plot_stage1")

ROOT = Path(__file__).resolve().parents[1]
G2B = 0.55


def main() -> int:
    S = json.loads((ROOT / "results/sweep/perun_detector/summary.json").read_text())
    rows = S["units"]
    tr = [r for r in rows if not r.get("reference")]
    ref = {r["model"]: r for r in rows if r.get("reference")}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    c = S["dose_response_ceiling"]

    for ax, key, title, ylab in (
        (ax1, "oracle_ceiling", "PRIMARY: detector quality drives the ceiling",
         "oracle_ceiling (recoverable fraction)"),
        (ax2, "id_retention", "…but every trained detector lands BELOW the baseline",
         "id_retention (end-to-end)"),
    ):
        xs_t = [r["map50_95"] for r in tr]
        ys_t = [r[key] for r in tr]
        ax.scatter(xs_t, ys_t, s=48, color="#E4844A", edgecolor="#7A3F14", zorder=3,
                   label="12 trained (MOT20/CARLA)")
        for tag, col, mark, lab in (("yolo11x", "#4C78A8", "s", "yolo11x baseline"),
                                    ("gtvis", "#2E8B57", "D", "GT visible (Stage 0)")):
            if tag in ref:
                ax.scatter([ref[tag]["map50_95"]], [ref[tag][key]], s=110, color=col,
                           marker=mark, edgecolor="#111", zorder=4, label=lab)
        fit = ols_with_ci([r["map50_95"] for r in rows], [r[key] for r in rows])
        x0, x1 = 0.15, 1.02
        ax.plot([x0, x1], [fit["intercept"] + fit["slope"] * x for x in (x0, x1)],
                color="#C03030", lw=2.0, zorder=2,
                label=(f"OLS slope {fit['slope']:.3f}\n"
                       f"95% CI [{fit['ci_lo']:.3f}, {fit['ci_hi']:.3f}]"))
        ax.set_xlabel("detector mAP50-95 on MOT17 dev-half")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(alpha=.25)
        ax.set_xlim(0.13, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper left")

    ax2.axhline(G2B, color="#C03030", ls="--", lw=1.6)
    ax2.text(1.03, G2B + .02, "G2b 0.55", color="#C03030", fontsize=9,
             weight="bold", ha="right")

    fig.suptitle(
        "Stage 1: the mechanism is CONFIRMED (slope "
        f"{c['slope']:.2f}, CI [{c['ci_lo']:.2f}, {c['ci_hi']:.2f}], R²={c['r2']:.3f}) — "
        "but the intervention FAILED (G2b not met)",
        fontsize=12.5, weight="bold", y=1.01)
    fig.tight_layout()
    out = ROOT / "viz" / "stage1_dose_response.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
