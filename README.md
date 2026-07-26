# occlusion-mot

Occlusion-aware multi-object tracking with **hidden-agent state prediction** (P1) and a
**CARLA occlusion-scenario data engine** for sim2real ablation (P2).

**Status: submission-ready, awaiting HPC access.** All tracker-level numbers below are
dev-half, pre-PERUN, placeholder-checkpoint results — honest by construction, final
claims come from the pre-registered val run and PERUN-scale sweep.

![CARLA occlusion scenario with per-walker GT visibility](demo/s5_carla_excerpt.gif)

*CARLA scenario with exact per-walker visibility ground truth (green ≥0.5, orange, red
<0.25) — the P2 data engine that labels occlusion for free. Full clip:
`demo/s5_carla.mp4`.*

## Problem

Trackers lose people behind people. On MOT17, ByteTrack retains the identity through
only **29% of ground-truth occlusion gaps** (dev-half, 168 segments) — 3 of 4 tracked
pedestrians come out of an occlusion as somebody else. This project (a) matches
ByteTrack on the standard metrics (HOTA/IDF1 parity gates), and (b) adds a
hidden-agent module: position estimate while occluded, re-emergence point/time, and
appearance-gated re-identification on reappearance, evaluated on occlusion segments
extracted from GT visibility.

## Architecture

- **Own ByteTrack** (`src/omot/track/`) — Kalman + two-stage IoU association,
  implementation-equivalence proven against a pinned reference (gate G0: ΔHOTA 0.52,
  ΔIDF1 0.07 at identical hyperparameters on identical detections).
- **Frozen-detections protocol** — the detector runs once; every tracker comparison
  consumes the identical cached detections. No comparison in this repo ever mixes
  detector outputs.
- **Hidden-state module** (`src/omot/hidden/`) — occlusion classification at track
  loss, extended coasting buffer with velocity damping (occluded pedestrians barely
  move: median gap displacement 0.02 image diagonals), a recovery association stage,
  and a cosine-distance appearance veto from a trained re-ID embedder
  (ResNet18/BN-neck, occlusion-upweighted triplet + CE, random-erasing = synthetic
  occlusion).
- **Occlusion taxonomy** — every eval emits a failure decomposition: pre-match rate,
  oracle association ceiling, association-scope vs end-to-end retention. This split
  is what separates *module* failures from *detector* failures (G2a vs G2b below).

## Methodology (the part that survives review)

| discipline | implementation |
|---|---|
| Gates G0–G4 | G0 repro **frozen PASS** · G1 parity pending val · G2 hidden-state: center-err + coverage + G2a-floor PASS on dev, G2a-paired pending scale, G2b contingent on detector arm · G3 quality (pytest+ruff+coverage, live) **PASS** · G4 CARLA feeder **PASS** |
| Val freeze | single pre-registered val event (`val_manifest.md`): configs, checkpoint SHA256s, pass/fail mapping, expected-power notes frozen BEFORE the run; no retune after |
| Multi-seed discipline | measured seed noise bounds (single-seed tracker deltas <6pt are noise — logged D36); arm comparisons use ≥3-seed means; the discriminating G2a criterion is a paired McNemar test, not a point delta |
| License guards | CI-enforced: zero dataset-derived pixels tracked (per-file visual allowlist with content classes), dataset/cache extensions blocked, 2MB default-deny, train/eval identity-disjointness verified per source |

## Results (dev-half; no single-seed claims — deltas within the ±6pt seed band are not claimed)

| configuration | e2e id retention | assoc-scope retention |
|---|---|---|
| ByteTrack baseline | 0.292 | 0.505 |
| + geometric hidden-state | 0.310 | 0.531 |
| + appearance veto (ImageNet null) | 0.327 | 0.556 |
| + trained embedder | **0.345** | **0.604** |

The baseline→full-stack progression (+5.3pt e2e, +9.9pt assoc) exceeds the seed-noise
band; individual embedder-variant differences do not, and are therefore not claimed —
trained-vs-ImageNet is p=0.26 (paired McNemar, n=96) pending PERUN-scale identities.
Re-emergence position error: 0.0055–0.0064 of the image diagonal (gate ≤0.015).

**Detector arm** *(dev-optimistic: detector finetuned ON dev-half GT and scored on
dev-half — the honest read is the pre-registered val run)*: oracle ceiling 0.583→0.845,
full-stack e2e 0.494.

![identity scaling](viz/scaling_curve_local.png)

*Identity count scales re-ID retrieval monotonically for both seeds; tracker-level
effects don't resolve at local scale — why the PERUN sweep uses log-scale pools and
≥3 seeds.*

![retention taxonomy](viz/summary_retention_grid.png)

## Data engine (P2)

| source | train ids | occluded-query eval ids | crops |
|---|---|---|---|
| MOT17 dev-half tracklets | 269 | 66 | 46.9k |
| CARLA renders (24 scenarios, exact visibility GT) | 36 | 9 | 58.2k |
| MOT20 tracklets (manifest-pinned 80/20 split) | 1,715 | 431 | 378.1k |
| Market-1501 (no vis GT → never in eval) | 1,500 | 0 | 26.0k |
| **total** | **3,520** | **506** | **509k** |

All sources license-verified with pinned provenance; nothing dataset-derived is in
the repo. A wild-clip qualitative showcase (10 permissive-stock clips, annotated) is
available locally / on request — regenerate with `scripts/showcase.py --src
showcase/sources --out showcase/renders`.

## PERUN sweep — designed, pre-registered, submission-ready

4-arm ablation (ImageNet-null / sim-only / real-only / sim+real; the P2 claim is
**sim+real > real-only**), log-scale identity pools {300, 1k, 2k, 3.5k} × 3 seeds,
detector finetunes at scale. Fully pre-registered in
[perun_sweep_v2.md](perun_sweep_v2.md) (selection rules, gate semantics, decision
criteria fixed before any run). Budget: **~30–36 H200h** (40h ceiling). The local
dry-run executed the identical entrypoint end-to-end. Submission, once
partition/account are filled in `configs/sweep/perun_full.yaml`:

```bash
python3 scripts/sweep_launcher.py --config configs/sweep/perun_full.yaml --mode slurm
sbatch results/sweep/perun_full/submit.sbatch
```

## Quickstart

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt -e .
.venv/Scripts/python.exe -m pytest -q          # full suite incl. license/disjointness guards
.venv/Scripts/python.exe scripts/demo_synthetic.py --render demo.mp4   # dataset-free demo
.venv/Scripts/python.exe scripts/render_demo.py --all                  # regen demo clips (needs MOT17)
```

**Guided tour:** [demo/README.md](demo/README.md) — 10 minutes, S0–S8, from the
failure mode to the gate scoreboard.

## Repo map

- `src/omot/` — loaders, MOT IO, detection/embedding caches, tracker, hidden-state module, eval (incl. paired test)
- `mission.md` / `gates.yaml` / `status.txt` / `context.md` — mission, executable gates, state, full decision log D1–D44
- `val_manifest.md` / `perun_sweep_v2.md` — frozen val pre-registration; approved sweep design
- `configs/sweep/` + `scripts/sweep_*.py` — config-driven sweep (identical entrypoint local/SLURM)
- `demo/` — guided tour + committed CARLA media; `scripts/showcase.py` — wild-clip renderer
- `runloop.ps1` — autonomous development loop (invokes `claude -p` per iteration against the gates)

License: AGPL-3.0-only (see context.md D9 — ultralytics dependency).
