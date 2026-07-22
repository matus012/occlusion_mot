"""Phase 4 tuning: grid-search HiddenConfig on the DEV half only (G2 metrics per combo).

Runs run_hidden.py per combo (tracking + segment eval, no TrackEval), collects G2 metrics,
ranks by (id_retention, center_err_coverage, -center_err) and writes
results/hidden_tuning.json. TrackEval G1-regression is checked afterwards on the top combos
only (separate invocation, --with-trackeval).

Usage: .venv/Scripts/python.exe scripts/tune_hidden.py
"""
from __future__ import annotations

import itertools
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("tune_hidden")

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

BUFFER = 90  # grid 1 showed buffer size is not the binding constraint (taxonomy: 0 expiries)
# grid 3 (phase 5a): appearance gates around the phase-4 pick (kf/noise1, d100, g15)
APP_GATES = [-1.0, 0.25, 0.35, 0.45]  # -1 = off; applied to both lost + recover vetoes
RECOVER_GATES = [1.5, 3.0]
OVERLAP = 0.25


def run_combo(
    buffer: int, damping: float, gate: float, overlap: float, tag: str,
    noise: float = 1.0, mode: str = "kf", app_gate: float = -1.0,
) -> dict:
    cmd = [
        str(PY), str(ROOT / "scripts" / "run_hidden.py"), "--half", "dev",
        "--skip-trackeval", "--tag", tag,
        "--occl-buffer", str(buffer), "--damping", str(damping),
        "--recover-gate", str(gate), "--overlap-thresh", str(overlap),
        "--noise-scale", str(noise), "--lowconf-mode", mode,
        "--app-gate-lost", str(app_gate), "--app-gate-recover", str(app_gate),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"{tag} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    blob = json.loads((ROOT / "results" / f"hidden_dev_{tag}.json").read_text(encoding="utf-8"))
    return blob


def main() -> int:
    results: list[dict] = []

    # Baseline-equivalent reference on dev (nothing classified occluded, no damping,
    # recovery effectively disabled by an impossible gate).
    base = run_combo(30, 1.0, 0.0001, 1.1, "devbase")
    log.info("DEV BASELINE g2: %s", base["g2"])
    results.append({"tag": "devbase", **base})

    for app_gate, gate in itertools.product(APP_GATES, RECOVER_GATES):
        tag = f"app{int(app_gate * 100)}_g{int(gate * 10)}"
        blob = run_combo(
            BUFFER, 1.0, gate, OVERLAP, tag, noise=1.0, mode="kf", app_gate=app_gate
        )
        g2 = blob["g2"]
        log.info(
            "%s: retention=%.3f cov=%.3f center=%.4f time=%.1f",
            tag, g2["id_retention"], g2["center_err_coverage"],
            g2["reemergence_center_err_med"], g2["reemergence_time_err_med"],
        )
        results.append({"tag": tag, **blob})

    ranked = sorted(
        results[1:],
        key=lambda r: (
            -r["g2"]["id_retention"],
            -r["g2"]["center_err_coverage"],
            r["g2"]["reemergence_center_err_med"],
        ),
    )
    out = {"dev_baseline": base, "ranked": ranked}
    dest = ROOT / "results" / "hidden_tuning.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("TOP 5:")
    for r in ranked[:5]:
        log.info("  %s: %s", r["tag"], r["g2"])
    log.info("-> %s", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
