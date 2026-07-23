# Session handoff — 2026-07-23 (context rotation before local-max phase L2)

Read order for the next session: CLAUDE.md → mission.md → status.txt → gates.yaml →
context.md (D1–D28) → this file. This file holds ONLY what those do not.

## Where we are (one line)
L1 of the local-max directive is done (G2a audit → D28 recalibration proposal PENDING
user); next is L2 (visualization/demo pipeline), then L3–L6 per the user directive in
D27; PERUN access pending on the user side; D18 val protocol untouched.

## In-flight / pending
- D28-PROPOSAL (G2a >= 0.58) awaits user decision — do NOT touch gates.yaml G2a until then.
- L2–L6 queued (user directive, verbatim in D27 context entry + status.txt).
- CARLA server is STOPPED (GPU freed). Restart before any sim rendering:
  `tools\CARLA_0.9.15\WindowsNoEditor\CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe
  CarlaUE4 -RenderOffScreen -quality-level=Low -carla-rpc-port=2000`
  then probe with .venv-sim + scratchpad probe (SERVER_UP takes 60–120s; restart the
  server between LONG driver batches — it memory-thrashes after ~30 sequential runs).
- motchallenge.net was still down; D13 re-hash vs official zip remains queued.

## Environment gotchas (hard-won, not in governance files)
- PowerShell 5.1 mangles quotes in `python -c` one-liners and here-strings passed to
  native exes → ALWAYS write scratch .py files. `Out-File -Encoding utf8` adds a BOM →
  read with utf-8-sig.
- A Windows venv's python.exe is a LAUNCHER: every run shows two processes (launcher +
  base interpreter with identical cmdline). Not a duplicate job. (See D-incident-adjacent
  correction in status history; misdiagnosing this killed a healthy download once.)
- CARLA: instance-seg ids are renderer-internal (stable per scene arrangement, NOT actor
  ids); same-blueprint walkers SHARE an id → per-scenario blueprint uniqueness in
  sim_driver is LOAD-BEARING (seeded permutation). Sensors lag set_transform by 1 tick;
  walkers returning from underground need ~4 settle ticks (mesh streaming). Child
  pedestrian blueprints 0009–0014 never render under teleport control (excluded).
- The two DirectX DLLs (XINPUT1_3, X3DAudio1_7) live NEXT TO CarlaUE4-Win64-Shipping.exe
  (no admin install); if the exe "hangs at 6MB RAM", it's a hidden loader error dialog.
- .claude/settings.json post-edit hook runs ruff+pytest on every .py write — suite is
  ~47 tests and quick, but budget for it on bulk edits.
- supervision==0.29.1 pin is load-bearing (ByteTrack removed in 0.30). numpy 1.26.4 pin +
  np.float shim in trackeval_runner are load-bearing (TrackEval unmaintained).

## Open hypotheses (unverified — treat as leads, not facts)
- Proto embedder retrieval numbers (occ-rank1 0.888) are flattered by same-sequence
  near-duplicate galleries; tracker-level gain is the real signal (+1.8pt assoc over
  ImageNet). Full-convergence training (L3) expected to add ~2–4pt assoc-retention; if it
  lands >= 0.62 dev, the D28 threshold 0.58 has margin on val.
- Val-half assoc-scope n will be ~76 (133 x ~0.577) → noisier than dev; sigma ~0.057.
- The 22% never-tracked-pre-gap mass may be partially recoverable by the detector
  workstream (L5/G2b), not only the 20% post-miss mass.

## Dead ends already tried (do not re-litigate without new evidence)
- fuse_score with COCO detections (worse IDF1/IDsw), coast-only lowconf mode (regressed
  on real data), kf noise inflation (neutral), buffer size beyond 60 (zero expiries),
  semantic-tag self-calibration for instance ids (picked scenery), actor-id == instance-id
  (false), cross-backend visibility correlation as a gate (structurally unsound — see D23).

## Disposable artifacts
- results/raw/dev_half/trackers/hidden_* (~60 tagged grid dirs) and coasting twins:
  regenerable, delete-safe with user approval only (standing rule).
- data/sim/carla_render_10bp: v3 renders (10-blueprint era) — tracking GT valid,
  re-ID labels superseded; kept for comparison.

## Working tree
Clean at commit d7caee4 + this handoff commit; branch main, pushed to
github.com/matus012/occlusion_mot (private).
