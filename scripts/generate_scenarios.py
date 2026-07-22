"""P2 / G4: generate + render occlusion scenarios into MOT-style dirs.

Renders with the mock backend by default; --backend carla once the sim venv exists (D22).
Writes results/carla_feeder.json in the schema gates.yaml G4 expects, with demo_runs set
by actually loading a rendered sequence back through the standard loader + extractor.

Usage: .venv/Scripts/python.exe scripts/generate_scenarios.py [--n 24] [--backend mock]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import load_sequence  # noqa: E402
from omot.eval.occlusion import extract_segments  # noqa: E402
from omot.sim.feeder import CarlaBackend, MockBackend, render_scenario  # noqa: E402
from omot.sim.scenario import generate_scenarios, save_scenarios  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("generate_scenarios")

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["mock", "carla"], default="mock")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sim" / "scenarios")
    args = ap.parse_args()

    scenarios = generate_scenarios(args.n, seed=args.seed)
    save_scenarios(scenarios, args.out / "specs")
    backend = MockBackend() if args.backend == "mock" else CarlaBackend()

    n_segments_total = 0
    for s in scenarios:
        seq_dir = render_scenario(s, args.out, backend)
        seq = load_sequence(seq_dir)
        assert seq.gt is not None
        n_segments_total += len(extract_segments(seq.gt, min_len=3))

    # demo-first check: load one back and verify end-to-end usability
    first = load_sequence(args.out / scenarios[0].scenario_id)
    demo_runs = (
        first.gt is not None
        and len(first.gt) > 0
        and first.frame_path(1).exists()
        and float(first.gt[:, 8].min()) < 0.25
    )

    report = {
        "n_scenarios": len(scenarios),
        "backend": args.backend,
        "has_visibility_gt": True,
        "demo_runs": bool(demo_runs),
        "n_occlusion_segments": int(n_segments_total),
    }
    # G4 is a CARLA gate: only real-sim renders write the gates-facing file. The mock
    # run writes a separate artifact so the gate stays honestly PENDING until then.
    fname = "carla_feeder.json" if args.backend == "carla" else "carla_feeder_mock.json"
    dest = ROOT / "results" / fname
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("%s", report)
    logger.info("-> %s", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
