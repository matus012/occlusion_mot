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
  reference impl (supervision) for G0. NOTE: supervision deprecates ByteTrack at v0.30 —
  the exact pin supervision==0.29.1 in requirements.txt is what keeps the G0 reference
  reproducible; do not bump it.
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
- **D12 (2026-07-22, updated) GitHub remote.** Original rule: remotes stay private without
  explicit approval. 2026-07-22: user created public repo matus012/occulsion_mot and gave
  one-line approval to push publicly (AGPL-3.0 already in place). Remote `origin` set; repo
  is PUBLIC by explicit user approval.
- **D13 (2026-07-22) MOT17 mirror fallback.** motchallenge.net down all session. User directive:
  free reputable mirror allowed after the official-site poller window; verify structure/hashes
  vs official spec; log source + hashes here; no paid/credential-gated mirrors. Chosen mirror:
  HF `ling1016/MOT17` (ungated, full 21-dir train layout verified via API; fallback
  `Morrison1025/MOT17`). Official site publishes no checksums -> verification is structural
  (scripts/verify_mot17.py: official seqLength/resolution table, frame counts, GT ranges,
  cross-variant gt.txt equality) + SHA256 manifest recorded in results/mot17_verification.json.
  Baseline stays UNVERIFIED until G0 freezes numerically. Provenance (filled post-download):
  source=, revision=, manifest_sha256=
- **D14 (2026-07-22) Occlusion-segment definition revised after contact with real GT.**
  D7's rule (all gap frames vis < 0.25, anything in [0.25, 0.5) contaminates) yielded 1
  segment on MOT17-02 because MOT GT visibility decays gradually through the 0.25-0.5 band.
  Revised: anchors vis >= 0.5; gap frames stay < 0.5; at least one dip < 0.25 (unannotated
  frames count as occluded); min_len 5; pedestrian class + consider flag only. Calibration:
  MOT17-02 -> 68 full / 41 val-half segments, median gap 37 frames (n.b. > ByteTrack's
  default 30-frame track buffer — exactly the failure mode the hidden-state module targets).
  Same numeric thresholds kept (0.25 / 0.5 / 5); only the contamination semantics changed.
