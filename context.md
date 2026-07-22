# context.md — occlusion-mot (P1+P2)

## Scope
P1: occlusion-aware multi-object tracker on MOT17/20 — ByteTrack-parity association plus a
hidden-state module (position estimate while occluded, re-emergence point/time, re-ID gating).
P2: CARLA occlusion-scenario feeder with GT visibility, for sim2real ablation of the hidden-state
module. Mission details in mission.md; gates in gates.yaml.

## Architecture
```
data/MOT17/...            (gitignored; motchallenge.net download)
        │
src/omot/
  data/mot.py             MOT sequence loader: frames, GT boxes + visibility, half-split protocol
  io/mot_format.py        MOT-format (frame,id,x,y,w,h,conf,...) read/write
  detect/cache.py         ultralytics yolo11x person detections → data/cache/detections/*.npz
  track/kalman.py         constant-velocity Kalman filter (xywh state)
  track/bytetrack.py      our ByteTrack: two-stage IoU association (high/low score)
  hidden/                 (phase 4) occluded-state motion model, re-emergence prediction, re-ID gate
  eval/trackeval_runner.py TrackEval wrapper → HOTA/IDF1/IDsw JSON in results/
  eval/occlusion.py       (phase 3) occlusion-segment extraction from GT visibility
carla/                    (phase 5, P2) scenario generation + feeder mirroring data/mot.py API
scripts/                  demo_synthetic.py, check_gates.py, hook_postedit.py, download helpers
results/*.json            metric outputs consumed by check_gates.py
runloop.ps1               autonomous outer loop (claude -p per iteration, fresh context)
```
Data flow: video/frames → cached detections (run once) → tracker (baseline | occlusion-aware) →
MOT-format output → TrackEval → results/*.json → gates.

## Decisions
- **D1 (2026-07-22) Fixed-detections design.** Detector inference runs once, cached to
  `data/cache/detections/`. All tracker comparisons use identical cached detections —
  detector-independent, cheap iteration on 8GB VRAM. Invariant in CLAUDE.md.
- **D2 (2026-07-22) Detector = ultralytics yolo11x (COCO, person cls), not YOLOX.** Avoids YOLOX
  Windows build pain. Published ByteTrack used MOT-finetuned YOLOX-x, so absolute numbers will
  differ → the reproduction gate (G0) is implementation-equivalence vs a reference ByteTrack on
  identical detections. Optional later: MOT-finetune a detector on PERUN.
- **D3 (2026-07-22) Own ByteTrack implementation** in src/omot/track (Kalman + two-stage IoU,
  no learned parts) because the hidden-state module extends tracker internals. Cross-checked vs
  reference impl (boxmot or supervision) for G0.
- **D4 (2026-07-22) Eval protocol** = MOT17 train half-split (first half dev / second half val,
  ByteTrack ablation protocol). HOTA/IDF1/IDsw via TrackEval; motmetrics cross-check.
- **D5 (2026-07-22) numpy pinned 1.26.4** — TrackEval unmaintained, breaks on numpy 2.x.
- **D6 (2026-07-22) torch from cu126 index**, latest resolved by uv; CUDA availability verified
  post-install (see status.txt).
- **D7 (2026-07-22) Occlusion segment definition**: GT track visibility < 0.25 for ≥ 5 consecutive
  frames, reappearing with visibility ≥ 0.5. Encoded in gates.yaml G2.
- **D8 (2026-07-22) Multi-GPU-ready structure**: device always injected, no `cuda:0` literals,
  config-driven batch sizes; SLURM launcher added later without refactor (PERUN H200s).
- **D9 (2026-07-22) Licensing: repo is AGPL-3.0.** ultralytics is AGPL-3.0 and is imported by the
  detection-caching path; isolating it buys nothing since we open-source anyway. Decision: license
  the whole repo AGPL-3.0-only (LICENSE file + pyproject). [User amendment 2026-07-22]
- **D10 (2026-07-22) TrackEval pinned** to commit `12c8791b303e0a0b50f753af204249e622d0281a`
  (HEAD at scaffold time) via git URL in requirements.txt. [User amendment 2026-07-22]
- **D11 (2026-07-22) G2 freeze order**: reemergence position error + id_retention freeze first;
  reemergence_time_err_med is the LAST sub-gate to freeze and never hard-fails before the
  hidden-state module exists. [User amendment 2026-07-22]
- **D12 (2026-07-22) Any GitHub remote stays private**; making it public requires explicit user
  approval. [User amendment 2026-07-22]
