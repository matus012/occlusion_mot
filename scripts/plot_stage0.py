"""Stage-0 kill-gate figure (perun_detector_v1.md section 2, D55)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("plot_stage0")

ROOT = Path(__file__).resolve().parents[1]
GATE = 0.55

ARMS = [
    ("yolo11x", "yolo11x\n(current detector)", "#4C78A8"),
    ("gtvis", "GT visible\n(detector supremum)", "#2E8B57"),
    ("gtall", "GT all\n(not achievable)", "#BBBBBB"),
]


def main() -> int:
    data = {
        tag: json.loads((ROOT / f"results/hidden_dev_stage0_{tag}.json").read_text())["g2"]
        for tag, _, _ in ARMS
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    labels = [lab for _, lab, _ in ARMS]
    colors = [c for _, _, c in ARMS]
    x = range(len(ARMS))

    ceil = [data[t]["oracle_ceiling"] for t, _, _ in ARMS]
    e2e = [data[t]["id_retention"] for t, _, _ in ARMS]

    ax1.bar(x, ceil, color=colors, edgecolor="#222", linewidth=.7)
    for i, v in enumerate(ceil):
        ax1.text(i, v + .015, f"{v:.3f}", ha="center", fontsize=10, weight="bold")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("oracle_ceiling  (associable segments / all)")
    ax1.set_title("Detection quality sets the ceiling", fontsize=11.5, weight="bold")
    ax1.set_ylim(0, 1.12); ax1.grid(axis="y", alpha=.25)
    ax1.annotate("", xy=(1, ceil[1]), xytext=(0, ceil[0]),
                 arrowprops=dict(arrowstyle="<->", color="#C03030", lw=1.8))
    ax1.text(.5, (ceil[0] + ceil[1]) / 2 + .04, f"+{ceil[1]-ceil[0]:.3f}\nheadroom",
             ha="center", color="#C03030", fontsize=9.5, weight="bold")

    ax2.bar(x, e2e, color=colors, edgecolor="#222", linewidth=.7)
    for i, v in enumerate(e2e):
        ax2.text(i, v + .015, f"{v:.3f}", ha="center", fontsize=10, weight="bold")
    ax2.axhline(GATE, color="#C03030", ls="--", lw=1.8)
    ax2.text(2.42, GATE + .018, "G2b gate 0.55", color="#C03030",
             fontsize=9.5, weight="bold", ha="right")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("id_retention  (end-to-end)")
    ax2.set_title("A perfect detector clears G2b by a wide margin",
                  fontsize=11.5, weight="bold")
    ax2.set_ylim(0, 1.12); ax2.grid(axis="y", alpha=.25)

    fig.suptitle(
        "Stage 0 kill gate: PASSED — the detector is the binding constraint "
        f"(kill fires below {GATE}; measured {e2e[1]:.3f})",
        fontsize=12.5, weight="bold", y=1.01)
    fig.tight_layout()
    out = ROOT / "viz" / "stage0_kill_gate.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("wrote %s", out)

    print(f"{'arm':<10}{'pre_match':>11}{'oracle':>9}{'e2e':>9}{'assoc':>9}{'center_err':>12}")
    for tag, _, _ in ARMS:
        d = data[tag]
        print(f"{tag:<10}{d['pre_match_rate']:>11.4f}{d['oracle_ceiling']:>9.4f}"
              f"{d['id_retention']:>9.4f}{d['id_retention_assoc']:>9.4f}"
              f"{d['reemergence_center_err_med']:>12.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
