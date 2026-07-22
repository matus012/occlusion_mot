# Mission — Occlusion-aware MOT + hidden-agent prediction (P1+P2)

## Goal
A multi-object tracker that:
- **(a)** matches ByteTrack on MOT17 — HOTA/IDF1 floor vs our reproduced baseline (gates G0/G1), and
- **(b)** adds hidden-agent state prediction: position estimate while occluded + re-emergence point/time, evaluated on occlusion segments extracted from GT (gate G2).

Plus **P2**: a CARLA feeder generating occlusion scenarios with GT visibility annotations (gate G4), enabling sim2real ablation.

## Phases (in order; P2 runs as a parallel workstream once phase 3+ is underway)
1. **eval-harness** — MOT sequence/GT loaders (incl. visibility), MOT-format IO, TrackEval wrapper, synthetic tests, runnable synthetic demo. Done: pytest green, `scripts/demo_synthetic.py` runs end-to-end.
2. **baseline** — detection caching (ultralytics yolo11x, person class), our ByteTrack implementation, reference-implementation cross-check on identical detections → freeze G0 numbers into gates.yaml.
3. **occlusion-segments** — extract occlusion-segment index from MOT17(/20) GT (visibility < 0.25 for ≥ 5 consecutive frames, reappears at ≥ 0.5); measure baseline ID-retention on segments → re-freeze G2 provisional thresholds.
4. **hidden-state** — occlusion-aware track state: motion model during occlusion, re-emergence point/time prediction, re-ID gating on reappearance → gates G1 + G2.
5. **carla-feeder (P2)** — scenario generation, visibility GT, feeder API mirroring the MOT loader → G4.
6. **sim2real-ablation** — hidden-state module trained/evaluated on CARLA vs MOT occlusion segments (heavy training on PERUN, not the 4060).
7. **demo** — demo video + README.

## Working loop (each runloop iteration)
1. Read status.txt (current phase) and the latest gate report.
2. Do the highest-leverage work for the current phase. Delegate to coder/eval-runner subagents; match model tier to task difficulty.
3. Reviewer subagent must approve diffs against CLAUDE.md rules. Failed review → rework in-loop, never surfaced to the user.
4. Update status.txt every cycle; commit at checkpoints; log decisions in context.md.
5. Never wait for user input except the four approval exceptions in CLAUDE.md.
