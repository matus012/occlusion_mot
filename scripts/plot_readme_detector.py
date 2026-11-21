"""README figure: end-to-end identity retention vs detector quality (pure matplotlib, our own metric JSON).

Same data as viz/stage1_dose_response.png (right panel), with plain-language labels for the README.

    python scripts/plot_readme_detector.py   ->  viz/readme_detector.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "sweep" / "perun_detector" / "summary.json"
OUT = ROOT / "viz" / "readme_detector.png"


def main() -> None:
    s = json.loads(SRC.read_text(encoding="utf-8"))
    units = s["units"]
    trained = [u for u in units if not u["reference"]]
    stock = next(u for u in units if u["reference"] and "yolo11x" in u["unit"])
    perfect = next(u for u in units if u["reference"] and "GT" in u["unit"])
    fit = s["dose_response_e2e"]

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=110)
    fig.patch.set_facecolor("white")
    xs = [0.15, 1.02]
    ax.plot(xs, [fit["intercept"] + fit["slope"] * x for x in xs], color="#c0392b", lw=1.5, ls="--", alpha=0.7,
            zorder=1, label="linear fit over all 14 points")
    ax.scatter([u["map50_95"] for u in trained], [u["id_retention"] for u in trained], s=55, color="#e67e22",
               edgecolor="black", lw=0.6, zorder=3, label="12 detectors trained here (non-MOT17 data)")
    ax.scatter([stock["map50_95"]], [stock["id_retention"]], s=110, marker="s", color="#2e6fb5",
               edgecolor="black", zorder=3, label="off-the-shelf detector (yolo11x)")
    ax.scatter([perfect["map50_95"]], [perfect["id_retention"]], s=130, marker="D", color="#27ae60",
               edgecolor="black", zorder=3, label="perfect boxes (upper bound)")
    ax.annotate(f"{stock['id_retention']:.2f}", (stock["map50_95"], stock["id_retention"]),
                xytext=(10, -16), textcoords="offset points", color="#2e6fb5", fontweight="bold")
    ax.annotate(f"{perfect['id_retention']:.2f}", (perfect["map50_95"], perfect["id_retention"]),
                xytext=(-44, 4), textcoords="offset points", color="#27ae60", fontweight="bold")
    ax.set_xlabel("detector quality on MOT17 (mAP50-95, development half)")
    ax.set_ylabel("people who keep their ID\nthrough an occlusion")
    ax.set_xlim(0.1, 1.06)
    ax.set_ylim(0, 0.9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper left", frameon=True)
    ax.text(0.62, 0.1, "the fit rests on the two reference points;\nthe 12 trained detectors alone show no significant trend",
            fontsize=9.5, color="#555555", ha="center")
    ax.set_title("Identity retention vs detector quality (development half)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
