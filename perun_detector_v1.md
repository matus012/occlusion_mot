# perun_detector_v1 — G2b detector workstream, pre-registration (FROZEN D53)

**Status: FROZEN as of 2026-08-19 (D53). Section 7 records the user's four decisions;
they are binding. Stage 0 is authorized to run. Stage 1 is NOT — it is blocked on the
Stage-0 verdict plus explicit user go.**

Authorized by D51 (user decision: accept the embedder null, pursue G2b). This document is
written to the same standard as `perun_sweep_v2.md`: selection rules, gate semantics and
decision criteria are fixed *before* any run, and the kill criteria are written to fire.

---

## 0. What this workstream tests, and the honest prior

**G2b (frozen, D26):** end-to-end `id_retention >= 0.55`. Current state: **0.345 dev /
0.278 val**. The claim under test is D25/D26's mechanism — that end-to-end retention is
*detector-capped*, so a better detector raises `oracle_ceiling` and pulls e2e up with it.

**The prior is not favourable, and this document says so up front rather than discovering
it later.** The D45 val event already produced one honest detector read (R6): a yolo11s
finetuned on dev-half GT lifted dev `oracle_ceiling` 0.583 -> 0.845, but on val the lift
**did not transfer** (0.632 -> 0.586), while e2e moved only 0.278 -> 0.338. The dev lift
was dev-optimism — the detector had trained on the half it was scored on. So the single
existing honest measurement of this mechanism is **negative-to-weak**.

That is why Stage 0 is a cheap falsification gate placed *before* any training budget is
committed, and why this design front-loads the honest read onto dev instead of reaching
for val.

## 1. The design fault this document exists to fix

Every detector number in the project so far carries the `dev-optimistic` label (D43
amendment 3): the detector was finetuned on **MOT17 dev-half GT** and scored on **MOT17
dev-half**. That is why R6 needed val to be read honestly at all — and val is now spent
(section 6).

**Fix: train the detector only on sources DISJOINT from the evaluation half.** MOT20 and
CARLA share no frames, sequences or scenes with MOT17. A detector trained on MOT20 +
CARLA and evaluated on MOT17 dev-half is **honest by construction**, and consumes no val.
This buys back an unlimited, un-contaminated dev read of exactly the mechanism G2b
depends on.

### FIXED-DETECTIONS rule: deliberately inverted here, and how

`CLAUDE.md` requires tracker-vs-tracker comparisons to use identical cached detections.
This workstream varies the detector *as the independent variable*, so the rule is
inverted, not broken:

> **Tracker config is FROZEN across every unit** at the D28-final canonical setting
> (b90 / d1.0 / g1.5 / overlap 0.25 / kf 1.0 / app-gate 0.45, conv embedder). Only the
> detection source varies. Every comparison in this document is detector-vs-detector at a
> fixed tracker, never tracker-vs-tracker.

No gate in `gates.yaml` moves. G2a is closed (D50/D51) and is not re-opened by any result
here.

## 2. Stage 0 — perfect-detector kill gate (runs FIRST, ~0.2 H200-h)

Before spending any training budget, establish the ceiling that detector improvement is
chasing, by substituting ground-truth boxes for detections at the top of an otherwise
unchanged pipeline.

- **Arm GT:** MOT17 dev-half GT boxes as the detection stream (class/confidence filled to
  satisfy the detection contract), frozen tracker, conv embedder.
- Reports the full `aggregate()` set, `oracle_ceiling` and e2e `id_retention`.

**KILL CRITERION, written to fire:**

> If **arm GT e2e `id_retention` < 0.55**, then no detector — however good — can reach
> G2b at this tracker configuration, because GT boxes are the supremum of any detector's
> output. The detector workstream is then **STOPPED**, its remaining budget is not spent,
> and G2b is reported as unreachable-by-detector with arm GT as the evidence. The gate is
> NOT lowered and the finding is not re-scoped; the next question becomes a
> tracker/association question, which is outside this document.

This costs ~0.2 h and can falsify a ~21 h workstream. It runs alone and its result is
reported before Stage 1 is submitted.

Informative-only from the same run: `oracle_ceiling` under GT boxes bounds how much of the
current 0.58 ceiling is detector-attributable at all.

## 3. Stage 1 — detector dose-response (only if Stage 0 clears)

The question is not "is detector X better than detector Y". It is **whether end-to-end
retention responds to detector quality at all**. Stage 1 therefore measures a
*dose-response curve* over 12 trained detectors spanning a wide quality range, plus two
free reference points.

### Grid

| factor | levels | n |
|---|---|---|
| model | yolo11s (48.9 GFLOPs @960), yolo11m (154.2) | 2 |
| train mix (MOT17-disjoint) | `mot20` (8,931 fr, 40 ep), `mot20_carla` (18,531 fr, 20 ep) | 2 |
| seed | 0, 1, 2 | 3 |
| **trained units** | | **12** |

Epochs are set per mix to hold frame-passes roughly constant (~357k vs ~371k), so the
mixes differ in *data composition*, not in gradient budget.

**Free reference points (no training cost):** the frozen `yolo11x` cached detections (the
D1 baseline, dev `oracle_ceiling` 0.583) and arm GT from Stage 0. With the 12 trained
points that gives **14 detector-quality levels**.

### Pre-registered analysis

1. **Primary (dose-response):** OLS of `oracle_ceiling` on detector `mAP50-95` measured on
   MOT17 dev-half, across all 14 points; report slope, 95% CI and R². Seeds enter as
   replicates, not as separate fits.
   **The mechanism claim is supported only if the slope is positive with a CI excluding
   zero.** A flat curve means detector quality does not bind the ceiling, and is reported
   as exactly that rather than tuned around.
2. **Secondary:** same regression with e2e `id_retention` as the response.
3. **G2b verdict:** `id_retention >= 0.55` (frozen D26 semantics, unchanged), evaluated at
   the best single detector and reported with its seed spread. **Single-seed deltas < 6pt
   are noise (D36/D50) and are never claimed** — the lesson the embedder sweep paid for,
   binding here.
4. **Every** trained detector's mAP and every arm's retention are reported. No
   cherry-picking, no post-hoc arm selection.

### Kill criterion

> If the Stage-1 primary slope's 95% CI includes zero, the detector-ceiling mechanism is
> **not established**, G2b is reported as not achieved by this route, and no further
> detector budget is requested. Thresholds do not move.

## 4. Budget — rebuilt from measured H200 throughput

`perun_sweep_v2.md` amendment 9 records that `UNIT_EST_H` was 4.3x conservative because it
was derived from RTX 4060 timings. This document does **not** reuse those numbers. It uses
a cost model fitted to the executed sweep:

```
cost_h = frames x epochs x GFLOPs(model, imgsz=960) x k
k      = 3.7315e-8   h per frame-epoch-GFLOP
```

calibrated on the measured point `yolo11s / mot17dev / 100 ep = 0.485 h` and **validated
out-of-sample** on the second measured point:

| unit | predicted | measured | error |
|---|---|---|---|
| yolo11s / mot17dev / 100 ep | 0.485 h (fit) | 0.485 h | — |
| yolo11s / mot17dev_carla / 100 ep | 2.237 h | 2.423 h | **-7.7%** |

The model under-predicts by ~8%, so every figure below carries a **1.15x safety factor**.
This is a documented recalibration with its evidence attached, not a silent edit of a
pre-registered constant.

| unit class | n | est/unit (incl. 1.15x) | subtotal |
|---|---|---|---|
| Stage 0 arm GT | 1 | 0.20 h | 0.20 h |
| yolo11s / mot20 / 40 ep | 3 | 0.75 h | 2.25 h |
| yolo11m / mot20 / 40 ep | 3 | 2.37 h | 7.11 h |
| yolo11s / mot20_carla / 20 ep | 3 | 0.78 h | 2.34 h |
| yolo11m / mot20_carla / 20 ep | 3 | 2.45 h | 7.35 h |
| per-detector eval (cache dets + embed + track) | 12 | 0.20 h | 2.40 h |
| **total** | | | **21.65 h** |

**Against 32.26 H200-h remaining under the unchanged 40 h ceiling** (7.74 h spent by sweep
77150). Headroom 10.6 h. The ceiling does not move.

### Wall caps — applied PER UNIT, not per model class

Amendment 9's lesson: detector cost is bimodal by **mix**, not by model (0.485 h vs
2.423 h for the *same* model), so amendment 7's model-level lever was coarser than the
cost structure. Caps here are per `(model x mix)` unit:

| unit | est | wall cap |
|---|---|---|
| yolo11s / mot20 | 0.75 h | 01:15:00 |
| yolo11m / mot20 | 2.37 h | 03:15:00 |
| yolo11s / mot20_carla | 0.78 h | 01:15:00 |
| yolo11m / mot20_carla | 2.45 h | 03:15:00 |

Worst case at the caps: 0.2 + 3(1.25) + 3(3.25) + 3(1.25) + 3(3.25) + 2.4 = **29.6 h**,
still under the 32.26 h remaining. A unit that overruns its cap is killed and leaves no
result JSON (existing resume rule); it is reported as a missing dose-response point, and
the cap is not raised.

**yolo11x is excluded by budget, pre-registered:** 5.9-6.1 h/unit x 3 seeds x 2 mixes is
~36 h alone, over the remaining ceiling. It is not a candidate, and no result will be
narrated as "yolo11x would have".

## 5. Implementation work required before submission (not yet done)

1. `finetune_detector.py`: `VALID_MIXES` is currently `("mot17dev", "mot17dev_carla")`.
   Add `mot20` and `mot20_carla` with the same prep/caching contract and the same
   dataset-dir-name-encodes-the-mix rule, plus a guard asserting **no MOT17 frame can
   enter a `mot20*` train split** — the entire honesty argument rests on that.
2. Stage-0 GT-detection source: a detection stream built from MOT17 dev-half GT satisfying
   the existing detection-cache contract.
3. Bundle: **MOT20 frames are not on the cluster.** `data/MOT20` is 3.2 GB / 8,931 frames
   and was never staged (the bundle carries MOT20 *crops* for re-ID, not frames). At the
   measured ~1 MB/s uplink that is a **~53 min transfer** — acceptable, and far cheaper
   than a bundle rebuild. Ship it as an incremental payload with its own sha256 manifest
   entry.
4. Tests: mix-disjointness guard, cost-model regression, and a `timeout`-grammar-class
   guard for any newly emitted script (the D49 bug class).

## 6. Val protocol — FLAGGED, not assumed

**A G2b val event is NOT covered by any existing pre-registration, and this document does
not assume one.**

What the record says:

- `val_manifest.md` binds **"the single D18 val event"**, executed exactly once at D45
  (2026-08-09). It is written for one pass and contains no provision for re-use.
- **D18.2:** exactly ONE config advances; the single canonical val run requires explicit
  user approval.
- **D18.3:** if val misses, re-tuning and **re-running require explicit approval** — the
  contamination decision is explicitly user-owned.
- **D46** accepted the G2a-floor val miss and deferred the margin to the PERUN embedder.
  Nothing in D45 or D46 authorizes a second val pass, for G2b or anything else.
- R6 **already spent** a detector-arm read at val (e2e 0.278 -> 0.338; the ceiling lift did
  not transfer).

So a second val event needs its **own justification and its own frozen manifest**, and it
carries a statistical cost that must be stated rather than glossed:

> Val has now been observed once. A second val event is no longer a virgin read: the D45
> outcome is part of the knowledge that shaped this design. Any G2b val number must be
> reported as a **second-look** result with the first look disclosed, or its nominal error
> rate is not what it appears to be.

**Proposed protocol, for user decision — not adopted here:**

1. Stages 0 and 1 run **entirely on dev**, with MOT17-disjoint training data. They are
   honest by construction and consume no val. This is sufficient to establish or refute
   the dose-response mechanism.
2. **Only if** Stage 1 establishes a positive slope AND the best detector reaches
   `id_retention >= 0.55` on dev does a val event become worth proposing.
3. That event would require: a new `val_manifest_g2b.md` frozen before the run, a single
   pre-committed detector (no selection at val), explicit user approval per D18.2, and
   mandatory second-look disclosure per the paragraph above.

If you prefer that G2b never touch val again, **the design above still stands entirely** —
it simply terminates at the dev dose-response, reported as such. Nothing in Stages 0-1
depends on a val event.

## 7. Frozen decisions (D53, 2026-08-19 — user; binding)

These four were the open questions in the draft. They are now decided and this section is
the authority on them.

1. **VAL IS CLOSED.** Stages 0 and 1 are **dev-only**. No val event is authorized by this
   document and none may be run under it. A positive Stage-1 slope earns exactly one
   thing: the right to **draft** a second frozen val manifest carrying the second-look
   disclosure of section 6, for **separate** user approval. Drafting is not approval, and
   approval of this document is not approval of that one. Nothing else about val is
   licensed by any Stage-1 outcome.

2. **STAGE 0 RUNS AND REPORTS ALONE.** The perfect-detector kill gate is submitted by
   itself, its verdict is reported to the user, and **Stage 1 submission is blocked on
   both that verdict AND an explicit user go**. No Stage-1 unit may be queued in
   anticipation of a favourable Stage-0 result, and the two stages may not be chained with
   a SLURM dependency — the gate is a human decision point, not a scheduler edge.

3. **GRID AS DRAFTED — NO WIDENING.** 2 models x 2 mixes x 3 seeds = 12 trained units,
   plus the yolo11x baseline and arm GT for 14 dose-response levels. No additional model,
   mix, seed or pool may be added without a new dated amendment. The 10.6 h of headroom
   under the ceiling is **not** an invitation to widen the grid; it is margin.

4. **MOT20 TRANSFER APPROVED** as an incremental payload (3.2 GB / 8,931 frames, ~53 min
   at the measured uplink), launched in background parallel to Stage 0 rather than
   blocking it.

### What Stage 0 may and may not conclude

Stage 0 answers exactly one question — whether a perfect detector can reach G2b at the
frozen tracker config. It is authorized to **stop** the workstream (section 2 kill
criterion). It is **not** authorized to start Stage 1; only decision 2 above can do that.

---

## Amendment 1 (D54, 2026-08-19 — specification gap in section 2, resolved before the run)

Section 2 specifies the Stage-0 arm as "MOT17 dev-half GT boxes as the detection stream"
and justifies the kill criterion with "GT boxes are the supremum of any detector's
output". **Those are two different sets, and the difference is material**: MOT17 GT
annotates targets that are *fully occluded* (visibility 0). On this data that is
**10,485 boxes, 9.3% of all consider-flagged pedestrian annotations** (112,297 total vs
101,812 with visibility > 0).

A detection stream that includes visibility-0 rows is not a detector — it sees through
occluders, which is precisely the capability whose absence creates the occlusion problem
in the first place. Feeding it to the tracker would hand the hidden-state module the
answer it is supposed to infer.

**Resolution, fixed before the run and before any number was seen:**

| arm | contents | role |
|---|---|---|
| `gtvis` | consider-flagged pedestrians with **visibility > 0** | **PRIMARY — drives the section 2 kill criterion** |
| `gtall` | consider-flagged pedestrians, any visibility | informative upper-**upper** bound, reported alongside; never gate-bearing |

`gtvis` is the faithful reading of section 2's own stated rationale: it is the supremum of
what a *detector* can emit. `gtall` is reported because the gap between the two is itself
the interesting quantity — it measures how much of the ceiling is unreachable by any
detector, however good.

Why the choice is not free: using `gtall` for the kill would be **too permissive**. It
could clear 0.55 in a world where no real detector ever could, and the workstream would
then spend ~21 h chasing an unreachable ceiling — exactly the outcome the kill gate exists
to prevent. The stricter arm is therefore the correct one, even though it makes the gate
more likely to fire against the workstream this document proposes.

No threshold, gate, arm, seed or budget changes. This amendment records an
under-specification in section 2 and the reading adopted for it, with the numbers that
make the distinction concrete. Implementation: `scripts/cache_gt_detections.py`
(6 synthetic-data tests, including a containment guard asserting `gtall` is a superset of
`gtvis`, so the two arms provably bracket the truth).

## Amendment 2 (D60, 2026-08-19 — implementation defect in the section 3 x-axis, caught
## on the first four units and corrected before any verdict was formed)

Section 3 fixes the dose-response x-axis as *"detector `mAP50-95` measured on MOT17
dev-half"*. The first Stage-1 implementation did not do that: it read `mAP50-95` straight
out of ultralytics' `results.csv`, which reports validation on the **training mix's own
held-out split** — MOT20 frames. The x-axis and the y-axis were therefore measured on two
different datasets.

It surfaced as a symptom rather than by inspection. Across the first four completed units:

| quantity | measured on | spread across units |
|---|---|---|
| `mAP50-95` from `results.csv` | MOT20 val split | 0.6011 – 0.6060 (**0.005**) |
| `oracle_ceiling` | MOT17 dev-half | 0.250 – 0.333 (0.083) |

Regressing the second on the first would have produced a slope with essentially no
x-variance behind it: a meaningless estimate with a huge CI, which the pre-registered kill
criterion would then have read as "CI includes zero → mechanism not established". **The
defect would have manufactured a null.**

**Correction:** `scripts/eval_detector_map.py` computes mAP50-95 on MOT17 dev-half from
the *same cached detections the tracker consumes*, so both axes describe one detector on
one dataset. No re-training was required — detections are already cached per detector tag.

Instrument proof, before any Stage-1 number was read through it:

| detection source | mAP50 | mAP50-95 | expected |
|---|---|---|---|
| `gtvis` (built *from* MOT17 GT) | **1.0000** | **1.0000** | exactly 1.0 by construction |
| `yolo11x` (baseline) | 0.6827 | 0.3969 | plausible off-the-shelf value |

The GT arm scoring exactly 1.0 is the check that matters: it proves the matcher, the IoU
sweep and the GT population agree.

This changes no threshold, gate, arm, seed or budget. It repairs an implementation that
did not match what section 3 already specified. The superseded MOT20-split number is
retained per unit as `map50_95_mot20_split` rather than discarded.
