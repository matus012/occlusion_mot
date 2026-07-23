# Project rules — occlusion-mot (P1+P2)

## Mission pointers
- mission.md — what to build and the phase order
- gates.yaml — when it's done (numeric gates; check_gates.py is the executable interpretation)
- status.txt — where we are (update every cycle)
- context.md — architecture + Decisions log

Read all four before acting. Log every nontrivial decision in context.md under "Decisions".

## Autonomy
- Fully autonomous loop: implement → test → review → document → commit. Never pause to ask "should I continue?".
- The ONLY actions requiring user approval (one line): force-push; deleting repos/branches/data; making anything public — **any GitHub remote must stay private**; spending money.
- If an instruction is technically wrong or a bad approach: say so and propose the fix in context.md Decisions; don't silently comply.

## One-writer rule
Exactly ONE writing session at a time (the orchestrator / current runloop iteration) owns all writes: files, commits, state-changing commands. Read-only consultant sessions may attach but MUST NOT write. If unsure whether you are the writer, you are not.

## Environment
- Win11, RTX 4060 Laptop 8GB VRAM — dev/debug/small runs only.
- All training-capable code must scale to SLURM multi-GPU (TUKE PERUN H200s) later: no hardcoded `cuda:0`, device always injected via config/arg, batch/world-size configurable, checkpointing rank-safe.
- Python 3.11 venv at `.venv` (uv-managed). NEVER install into system Python. Install: `uv pip install --python .venv/Scripts/python.exe <pkg>`, then re-freeze requirements.txt.

## Code rules (reviewer agent enforces; violations = REJECT)
- Type hints on every function signature.
- Every ML entry point: explicit seed setting (random/numpy/torch), explicit device selection with CPU fallback, module-level `logging` logger (no bare print in src/).
- Fail fast in dev code (assert/raise early). try/except only on production paths (demo scripts, runloop tooling).
- Every core module in src/omot has tests; synthetic-data tests preferred (no dataset dependency).
- ruff clean (config in pyproject.toml).

## Experimental design (invariants)
- **FIXED DETECTIONS**: tracker-vs-tracker comparisons MUST use identical cached detections from `data/cache/detections/`. Never compare trackers run on different detector outputs.
- Eval protocol: MOT17 train half-split — first half of each train sequence = dev, second half = val. HOTA/IDF1/IDsw via TrackEval (pinned commit in requirements.txt); MOTA/IDF1 cross-checked with motmetrics.
- Demo-first: every phase ends with something runnable end-to-end, verified by actually running it before claiming done.

## Commits
Commit at logical checkpoints (tests green, module complete, gate frozen) — not one giant commit. Imperative messages; reference gate IDs where relevant (e.g. "G0: freeze baseline numbers").

## Response protocol (D27, binding for every Builder response)
- **VISUAL-FIRST**: any milestone with viewable output ships the visual, not just metrics.
  Defaults: annotated video (track boxes, occlusion-state coloring, coasting/recovery
  markers, hidden-agent predicted pose), qualitative crop grids, plots. End of response:
  artifact path(s) + one-line open command. Metric deltas: short table (<= 8 rows),
  never prose lists of numbers.
- **CTA FOOTER**: the final line of every response is exactly one of:
    CTA: REVIEW — <what to look at>
    CTA: PICK — <options>
    CTA: WAIT — <running task> — ETA <hh:mm>
    CTA: STUCK — <blocker + what's needed from user>
    CTA: DOWNLOADING <what> — ETA <hh:mm>
    CTA: DONE — <next queued item>
  ETA is mandatory for WAIT/DOWNLOADING.
