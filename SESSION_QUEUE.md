# SESSION_QUEUE — resumable state for occlusion-mot

**Purpose: a fresh session with ZERO prior context can read this file and continue.**
Updated after every completed item. Last update: 2026-08-19 23:05 — SESSION COMPLETE.

Read alongside: `status.txt` (phase/blockers), `context.md` (decision log D1–D63),
`STAGE1_RUNBOOK.md` (how to finish Stage 1 if it ever needs re-running).

---

## Hard rules in force

**⚠ APPROVAL-GATED — never execute without an explicit user "yes":**
- any second val run
- any deletion of files/data
- force-push
- anything public-facing beyond this repo (the repo itself is intentionally public)

**Autonomy limits (standing):**
- never exceed `--array ...%8` SLURM concurrency
- never raise a wall cap or the 40 h H200 ceiling
- no config edits beyond levers already pre-registered in a frozen doc
- frozen docs (`gates.yaml`, `val_manifest.md`, `perun_sweep_v2.md`,
  `perun_detector_v1.md`) change ONLY via a dated amendment appended to the doc —
  never by rewriting history
- report anomalies; do not improvise around them silently
- gates before every commit: `ruff check` clean AND full `pytest` green

**Decisions received (binding):**
| decision | date | effect |
|---|---|---|
| Accept the embedder null; pursue G2b | 2026-08-19 | D51 |
| Val CLOSED for the detector workstream | 2026-08-19 | D53 d1 — Stages 0/1 dev-only |
| Stage 0 runs and reports alone; Stage 1 needs explicit go | 2026-08-19 | D53 d2 (both satisfied) |
| Grid frozen as drafted, no widening | 2026-08-19 | D53 d3 |
| MOT20 incremental transfer approved | 2026-08-19 | D53 d4 (done) |
| **PROPOSED_val2_manifest.md REJECTED — do not run val** | 2026-08-19 | draft stays on disk, marked rejected |
| **Stage-1 TRAINED-DETECTOR demo variant skipped** | 2026-08-19 | corrected scope: only the Stage-1-detector variant is dropped, NOT the demo item. Demo ships on the current best stack (yolo11x dets + hidden-state + conv embedder). |

---

## Queue

### 1. STAGE-1 WATCH + AGGREGATION — ✅ DONE
Array 77354/77367/77386, 12/12 units, zero failures, 10.72 H200-h.
- Verdict (D61): mechanism **SUPPORTED** (`oracle_ceiling ~ mAP50-95` slope 0.902,
  CI [0.784, 1.021], R² 0.949) but **G2b NOT MET** (best trained e2e 0.202 vs ≥0.55, and
  0.143 *below* the yolo11x baseline).
- Written into README, `context.md` D61, `status.txt`. Figure
  `viz/stage1_dose_response.png`. Summary `results/sweep/perun_detector/summary.json`.
- Acceptance: 12 result JSONs + summary.json committed locally ✅; slope/CI/table/budget
  reported ✅.

### 2. DEMO REBUILD — ✅ DONE
Scope correction 2026-08-19 22:05: only the *Stage-1 trained-detector variant* is dropped.
The demo item itself ships, on the **current best stack** — yolo11x cached detections +
geometric hidden-state + trained conv embedder @ app-gate 0.45, the exact config behind
the dev/val tables.
- **HERO CLIP REBUILT (D64)** ✅ — `render_hero` rewritten from a 4-stage montage with
  title cards + CARLA tail (~63 s) into what was actually asked for: **baseline vs full
  stack side by side, persistent ID labels, ~31 s**, two segments from two different
  sequences (MOT17-09, MOT17-04) so it cannot be read as one lucky pick. 2 s opener,
  3 s outro carrying the headline numbers. `render_pair_clip` gained a `canvas_size`
  parameter so multi-segment concat cannot fail on a size change.
- `demo/README.md` now opens with a **"Watch this one first"** hero section — the tour
  previously never mentioned the hero clip at all.
- Full demo suite re-rendered clean (13 artifacts) ✅; wild-clip showcase re-rendered
  against `reid_conv` (11 renders + contact sheet) ✅
- Hero tests updated to the new list contract, plus a new guard that the reel spans two
  sequences ✅
- Acceptance: clips regenerate from a clean checkout ✅; hero reads in <30 s without the
  README ✅; license rules held — `hero.mp4` holds MOT17 pixels, stays gitignored ✅

### 3. README FINAL PASS — ✅ DONE
Hero GIF → problem → three-line honest findings → how-it-was-measured table → detail,
plus a detector-workstream section carrying the Stage-0 and Stage-1 verdicts with their
caveats. Every honesty label retained.

### 6. TECHNICAL REPORT — ✅ DONE
`report/omot_report.md` (~4,100 words, arXiv-short-paper shape): abstract, problem, method,
protocol, results, limitations, reproducibility appendix. Every number cites a committed
result file by path.
- Reviewer pass for overclaiming found and fixed **three** issues:
  (a) the median occlusion gap was stated as 37 frames and used to argue it exceeds
      ByteTrack's 30-frame buffer — the committed index says **25.0**, which is *inside*
      the buffer. The buffer argument now rests on the tail (p90 = 97.3) where it actually
      holds, which is both correct and a stronger point.
  (b) CARLA segment count 223 → **225** per `results/carla_feeder.json`. (I first
      reported the stale value as living in README.md — wrong: README never carried it.
      The stale instances were `status.txt` and `context.md` D23, both handled in D66:
      status corrected outright, the dated D23 entry annotated rather than rewritten.)
  (c) segment-count sourcing split correctly: 168 dev-half comes from the run outputs,
      133 val-half from the segment index (they do not sum to the 328 full-sequence count).
- Verified independently: G0 ΔHOTA 0.522 / ΔIDF1 0.070 recomputed from
  `results/g0_equivalence/`; IDsw 359 → 298 is a 17.0 % reduction.
- Acceptance: every cited path exists and is git-tracked ✅; honesty labels carried verbatim
  (dev-optimistic, R6 prototype, MOT17-disjoint) ✅; claims table maps each claim to
  supported / not established / refuted ✅

### 4. REPO CLEANUP — ⏸ BLOCKED on ⚠ line-item approval
`DELETION_PROPOSAL.md` written. **Nothing moved or deleted.**
- Group A: ~36 GB of gitignored build artifacts (superseded bundle, staging tree, CARLA
  zip) — zero repo impact, the only material win.
- Groups B/C: stale renders and overlapping docs — tidiness only.
- Group D: what NOT to touch, with reasons (`results/`, `viz/`, detection caches, `logs/`).
- **Next action: wait.** Do not execute any group without per-line approval.

### 5. PROJECT DOCS — ✅ DONE
- `occlusion_mot_plain.md` — Stage-1 verdict folded in ✅
- `context.md` D1–D63 current ✅ · `status.txt` current ✅
- `STAGE1_RUNBOOK.md` ✅ · `SESSION_QUEUE.md` (this file) ✅ · `session_handoff.md` ✅
- `PROPOSED_val2_manifest.md` marked REJECTED in-file and in `context.md` D63 ✅
- Acceptance: a fresh reader can answer "what was found, what it means, what's next"
  from `occlusion_mot_plain.md` alone ✅

---

## SESSION COMPLETE — nothing is running

No SLURM jobs, no background renders, no monitors armed. Working tree clean, all gates
green (ruff + full pytest).

**The only open item is #4 (repo cleanup), which waits on your line-item approval.**
A successor session can start cold from this file.

### If you approve DELETION_PROPOSAL.md later
Execution plan is in that file's "How I would execute" section: create `attic/` +
`attic/MANIFEST.md`, move (never delete) approved Group B/C items, hard-delete only the
Group A gitignored build artifacts you explicitly mark, run the full suite after EACH
group, commit per group referencing the line items.

---

## Where the project actually stands (the 60-second version)

1. **Parity holds.** The occlusion module clears every ByteTrack-parity gate on held-out
   val and costs nothing in ordinary tracking quality.
2. **Appearance scaling is a closed negative result.** 23 PERUN units, 3 seeds,
   pre-registered: retrieval scales with identities, tracker association does not.
3. **The detector is the bottleneck — proven, not exploitable yet.** Perfect visible GT
   boxes take the ceiling 0.571 → 0.970 and e2e 0.345 → 0.786. Detector quality drives the
   ceiling at slope ~0.90. But every MOT17-disjoint-trained detector came out *worse* than
   the off-the-shelf baseline, so G2b (0.55) is not met.
4. **Open problem:** the next step needs a better *MOT17-domain* detector, which conflicts
   with the disjointness that made the Stage-1 read honest. Unresolved.
5. **Budget:** 18.46 h of the 40 h H200 ceiling spent; 21.54 h remain. Ceiling never moved.

## Cluster state (PERUN)

- Repo: `/mnt/project/perun26011488/omot/omot_hpc/repo` — bundle `8421645` + patch
  `4c97f1d` + Stage-1 files. Provenance table in `READY.md` there.
- Queue empty. All Stage-1 results pulled locally and committed.
- Outputs on Lustre via symlinks (`data/models`, `data/reid`, `results/sweep/perun_full`,
  `.cache`); bundle and venv on project storage.
