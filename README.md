# occlusion-mot

Occlusion-aware multi-object tracking with hidden-agent state prediction (P1) and a CARLA
occlusion-scenario feeder for sim2real ablation (P2).

**Goal:** a tracker that (a) matches ByteTrack on MOT17 (HOTA/IDF1) and (b) predicts the state
of occluded agents — position while hidden, re-emergence point and time — evaluated on occlusion
segments extracted from MOT17/20 ground truth.

## Layout
- `src/omot/` — package: data loaders, MOT IO, detection cache, tracker, eval
- `mission.md` / `gates.yaml` / `status.txt` / `context.md` — mission, quality gates, state, decisions
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
