# Session handoff — 2026-08-19 (autonomous evening session)

> **PROJECT CLOSED 2026-09-04 (D67).** This file is retained as the narrative record of
> the session that executed both PERUN workstreams. It is no longer a handoff: there is
> nothing to pick up. Final state is `status.txt` (`state: CLOSED`), the write-up is
> `report/omot_report.md`, and the summary is the `## Headline` section of `README.md`.
> Final compute 18.46 / 40 H200-h. The "what I would pick up next" section below has been
> rewritten as future work that was **not pursued**.

**Start here, then read `SESSION_QUEUE.md`.** That file is the resumable state: queue
items, acceptance criteria, decisions received, and the approval-gated list. This file is
the narrative of what happened in this session and why.

## What this session did

Took the project from "PERUN sweep submission-ready" to "both PERUN workstreams executed,
analysed and written up".

1. **Embedder sweep (array 77150)** — 23/23 units, 7.74 H200-h. Result: **a null**. G2a
   fails both limbs (floor 0.5779 vs 0.58; best paired McNemar p=0.143 vs <0.05). Accepted
   as the finding rather than tuned away (D50/D51).
2. **Detector workstream** — pre-registered fresh (`perun_detector_v1.md`, frozen D53).
   - **Stage 0** (kill gate, ~0 H200-h, ran locally): perfect visible GT boxes take the
     ceiling 0.571 → 0.970 and e2e 0.345 → 0.786. **Passed by +0.236.** The detector is
     the binding constraint (D55).
   - **Stage 1** (12 units, 10.72 H200-h): mechanism **CONFIRMED** — ceiling scales with
     detector mAP at slope 0.902, CI [0.784, 1.021], R² 0.949 — but **G2b NOT MET**,
     because every MOT17-disjoint-trained detector came out worse than the off-the-shelf
     baseline (D61).
3. **Docs/demo** — README restructured for a 90-second reader, demo suite and showcase
   re-rendered, `occlusion_mot_plain.md` written, cleanup proposal drafted.

## Four defects I introduced and caught — read these before trusting my code

1. **D49 `timeout` format** — the emitted array passed SLURM `HH:MM:SS` to coreutils
   `timeout`, which rejects it. All 23 tasks of array 77126 died in under a second. The
   suite had only ever asserted the `#SBATCH --time` line. Fixed + mutation-tested guard.
2. **D57 dataset-prep race** — concurrent `os.link` → `FileExistsError` → `copy2` →
   `SameFileError` killed 10/12 Stage-1 tasks. Fixed with a tolerant link *and* a prep
   lock, which also closed a worse latent race (a peer `rmtree`-ing a dataset mid-build).
3. **D60 wrong-dataset x-axis** — the dose-response x-axis was read from ultralytics'
   `results.csv`, which validates on the *MOT20 training split*, not MOT17. **Its failure
   mode was a false null**: near-zero x-variance would have produced a wide CI that the
   pre-registered kill criterion reads as "mechanism not established". Fixed and
   instrument-proved (`gtvis` scores exactly 1.0000).
4. **Wall caps derived instead of transcribed** — my launcher computed `01:00:00` where
   the frozen doc says `01:15:00`, i.e. tighter than the pre-registration, which would
   have killed units the doc permits to finish. Caps now come verbatim from the doc.

The pattern worth noting: three of the four would have produced a *plausible-looking wrong
answer* rather than an obvious crash.

## State of the frozen record

`gates.yaml`, `val_manifest.md`, `perun_sweep_v2.md`, `perun_detector_v1.md` are all
intact. Two amendments were appended to `perun_sweep_v2.md` (8: occ-rank1 demoted to a
within-arm diagnostic; 9: UNIT_EST_H recalibration) and two to `perun_detector_v1.md`
(1: gtvis/gtall split; 2: the MOT17 x-axis correction). **No threshold was ever moved and
no history was rewritten.**

## Val

Read exactly once (D45, 2026-08-09). A second event was drafted when the Stage-1 slope came
back positive, with a recommendation against running it — **user rejected it** (D63).
`PROPOSED_val2_manifest.md` is retained, marked REJECTED, and authorises nothing.

## Future work — not pursued (recorded at closure, D67)

The project closed with these three forks open. None is blocked; each was simply out of
scope at closure, and none is claimed as a result.

1. **A better MOT17-domain detector.** The actionable next step, and it conflicts with the
   MOT17-disjointness that made the Stage-1 read honest. That tension is unresolved and is
   the real frontier.
2. **The tracker-side residual.** ~19% of recoverable segments are lost inside the tracker
   even with perfect detections. No detector work can close it.
3. **Sim-to-real transfer (P2) beyond the data engine.** *sim+real > real-only* is **not
   established** by its own pre-registered rule (assoc holds, occ-rank1 fails and is
   cross-arm-invalid anyway).

`DELETION_PROPOSAL.md` Group A — the repo-cleanup item this file previously listed — was
executed at closure on owner approval: 28.81 GiB of gitignored build artifacts removed
(A1, A2, A4), with A3 and A5 kept as the proposal recommended. Groups B/C/D untouched.
