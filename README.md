# occlusion-mot

Occlusion-aware multi-object tracking with hidden-agent state prediction (P1) and a CARLA
occlusion-scenario feeder for sim2real ablation (P2).

**Goal:** a tracker that (a) matches ByteTrack on MOT17 (HOTA/IDF1) and (b) predicts the state
of occluded agents — position while hidden, re-emergence point and time — evaluated on occlusion
segments extracted from MOT17/20 ground truth.

**Status: submission-ready, awaiting HPC access.**

## State (2026-07-24)

Dev-half results (MOT17 first-half, fixed cached yolo11x detections; val untouched
per protocol): occlusion-aware tracker + trained re-ID embedder lifts end-to-end id
retention through occlusion from 0.292 (ByteTrack baseline) to 0.345, association-
scope retention 0.505 -> 0.604; a dev-half-finetuned detector prototype lifts the
oracle ceiling 0.583 -> 0.845 (dev-optimistic — trained on dev; honest read comes at
val). Identity count, not training length, is the binding constraint on the embedder
(see context.md D29/D36); training pool now 3,520 identities across MOT17-dev / CARLA
/ MOT20 / Market-1501.

**Next stage — PERUN 4-arm sweep** (`perun_sweep_v2.md`, pre-registered): arms
ImageNet-null / sim-only / real-only / sim+real, log-scale identity pools, >= 3 seeds,
detector finetunes at scale. Budget: **~30-36 H200h** (40h ceiling). Dry-run passed
end-to-end locally through the identical entrypoint. Submission path (after filling
partition/account in `configs/sweep/perun_full.yaml`):

```bash
python3 scripts/sweep_launcher.py --config configs/sweep/perun_full.yaml --mode slurm
sbatch results/sweep/perun_full/submit.sbatch
```

## Layout
- `src/omot/` — package: data loaders, MOT IO, detection cache, tracker, hidden-state module, eval
- `mission.md` / `gates.yaml` / `status.txt` / `context.md` — mission, quality gates, state, decisions
- `val_manifest.md` / `perun_sweep_v2.md` — frozen val pre-registration; approved sweep design
- `demo/README.md` — 10-minute guided tour (S0-S8) with clips; `scripts/showcase.py` — wild-clip renderer
- `runloop.ps1` — autonomous development loop (invokes `claude -p` per iteration against the gates)
- `scripts/demo_synthetic.py` — dataset-free end-to-end demo (synthetic occlusion scene)

## Quickstart
```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt -e .
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/demo_synthetic.py --render demo.mp4
```

License: AGPL-3.0-only (see context.md D9 — ultralytics dependency).
