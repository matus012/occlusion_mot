# PERUN sweep design v2 (D37 item 6 — user reviews BEFORE submission approval)

Supersedes the implicit v1 ("train the embedder longer on PERUN"). v1 is dead: D29/L3
showed epochs saturate; D36 showed identities are the lever but tracker-level effects
need multi-seed aggregation. v2 = fast sweeps over ENLARGED identity pools + detector
finetune at scale (D33).

## Data state (post-D37/D38 integration, all local, license-guarded)

| source | train ids | eval ids (occluded-query) | crops | notes |
|---|---|---|---|---|
| mot17_dev | 269 | 66 | 46,859 | dev-half tracklets only (D18-safe) |
| sim (CARLA) | 36 | 9 | 58,159 | blueprint ceiling 45 exhausted (D34) |
| mot20 | 1,715 | 431 | 378,062 | 80/20 manifest-pinned (D38) |
| market1501 | 1,500 | 0 | 26,038 | no vis GT -> banned from eval side |
| **total** | **3,520** | **506** | **509,118** | train pool 11.5x the D36 curve range |

Occluded-query eval (D38 bounds q [0.10,0.50), g >= 0.60): 422 answerable identities
(>=1 query AND >=1 gallery crop), vs 58 pre-integration. ~~Optional extension: MSMT17
(+4,101 ids -> ~7.6k pool) is already mirrored at the pinned aveocr HF repo; requires
a separate user approval (license form semantics) — NOT in this sweep's baseline plan.~~
(STRUCK by amendment 4, D43: MSMT17 via mirror is a license-hygiene violation class;
official request form only, post-val, user-gated.)

## Arms (D37 redesign, mission.md phase 6)

- **A** ImageNet null — no training; embedding-cache pass only.
- **B** sim-only — 36 train ids (fixed; cannot scale — that is itself a finding slot).
- **C** real-only — mot17_dev + mot20 + market1501 = 3,484 ids.
- **D** sim+real — 3,520 ids. **The P2 claim is D > C**, not B > A.

## Sweep grid

1. **Ablation core (the claim):** arms B, C, D at FULL pool x 3 seeds (identity-seed
   AND init-seed varied together) = 9 training runs. Arm A adds 1 cache pass.
2. **Identity-scaling curve (PERUN edition of D36):** arm D at pools
   {300, 1000, 2000, 3520} x 3 seeds — nested seeded subsets via the committed
   scaling_study machinery (per-source proportional fractions). Full-pool points shared
   with (1) -> 9 additional runs.
3. **Detector arm at scale (G2b workstream, D26/D31):** yolo11s AND yolo11m finetuned
   on MOT17 dev-half GT + CARLA renders (two data mixes), ~100 epochs each = 4 runs.
   Val-clean by construction (no val frames in training).

Total: 18 embedder runs + 4 detector runs + cache passes.
(Exact array enumeration: **25 units** — see amendment 5, D48. SUPERSEDED 2026-08-19 by amendment 7, D49: the detector lever fired on measured smoke throughput, dropping yolo11m and taking the grid to **23 units**.)

## Metrics & decision rules (pre-registered)

- Per embedder run: occ-rank1 / occ-mAP under D38 bounds (primary at sweep scale;
  resolves identity effects — D36), plus standard rank1/mAP.
- Tracker-level (dev half, FIXED yolo11x detections): per-arm best gate from a 3-point
  gate probe {0.40, 0.45, 0.50}; assoc retention reported as MEAN +/- spread over the
  3 seeds. Single-seed deltas < 6pt are noise (D36) — never claimed.
- Paired McNemar (frozen D28 code path) per seed: D vs A, D vs C, C vs A on the
  intersection denominator; report all p-values, no cherry-picking.
- D > C verdict: mean assoc(D) > mean assoc(C) AND mean occ-rank1(D) > (C) with
  non-overlapping seed ranges; anything weaker is reported as "not established".

## Budget (H200, single-GPU jobs; model is ResNet18 @ 64x128 — small)

CORRECTED at D43 close-out (dry-run-measured throughput; the original <= 0.5 h/run
embedder estimate did not survive measurement, and the first corrected draft
silently dropped the cache/detector blocks from the total — both errors fixed here):

CORRECTED AGAIN at D48 (2026-08-10) to match the executable interpretation in
`sweep_common.UNIT_EST_H` — which is now what the sbatch generator and its budget
assertion actually read. Two accounting changes, no design change: the cache pass and
gate probe are billed INSIDE their embedder unit (they run in the same array task, not
as separate jobs), and arm A is billed as its own cheap class. Detector runs are billed
at their **wall cap**, not their estimate high (amendment 6).

| block (= array unit class) | units | est. per unit (measured basis) | wall limit | subtotal |
|---|---|---|---|---|
| embedder unit: 60 ep x 400 batches @ eval-every-5 (~1.2 h) + embedding cache (~0.2 h) + 3-point gate probe (~0.15 h) | 18 | ~1.55 h | 02:10 | ~27.9 h |
| arm-A null unit: retrieval eval + cache + gate probe, no training | 3 | ~0.45 h | 00:40 | ~1.35 h |
| detector finetunes (~100 ep, yolo11s + yolo11m x 2 mixes) | 4 | 1.5-3 h, **billed at the 2.5 h cap** | 02:30 | 6-10 h |
| eval/tracker runs | — | CPU, inside the units above | — | — |
| **honest total** | **25** | | | **35.25-39.25 H200h vs the 40 h ceiling — worst case now fits by construction** |

SLURM shape: job array over (arm, pool, seed); device/seed/batch injected (D8);
data staged as the re-ID crop trees (~1.4 GB) + caches; checkpoints + result JSONs
synced back. STAGING NOTE (D43 close-out, pre-registered): the requirement to stage
PRE-RESIZED 64x128 crops is ALREADY SATISFIED BY CONSTRUCTION — all four extractors
write 64x128 at extraction time (Market-1501 is natively 64x128); verified on disk
2026-07-24 (all sources 64x128, 1.38 GB total). Residual loader cost is
single-threaded tiny-jpg decode, untouched by resizing; if more margin is needed the
levers are a parallel DataLoader or decode-free uint8 .npy shards — optional, not
required for the 40h ceiling.

## Pre-registration amendments (D43 — user verdict 2026-07-24: APPROVED as written
## plus these four; appended verbatim in substance, no design changes)

1. **Detector-arm selection rule (pre-registered):** the R6 candidate among the 4
   detector finetunes is picked by **dev oracle-ceiling, tie-break e2e at the fixed
   gate**. Stated here BEFORE any run; no post-hoc selection.
2. **Gate probe semantics:** per arm, the gate is chosen by MEAN assoc over the 3
   seeds; ONE gate per arm applied to all its seeds. No per-seed gate picking.
3. **Dev-optimistic label:** mandatory on every table/figure/log line where a
   dev-half-GT-trained detector is evaluated on dev-half. Only R6 at val is the
   honest number.
4. **MSMT17:** the HF-mirror acquisition path is STRUCK from this doc. If ever
   approved (post-val, user-gated), acquisition is via the official request form
   ONLY. Mirror acquisition is a license-hygiene violation class (D39/D41).

## Pre-registration amendments (D48 — 2026-08-10, pre-submission; enumeration and
## budget mechanics only, no arm/pool/seed/metric changes)

5. **Correct array enumeration: 25 units, not 18.** The "Sweep grid" section counts
   18 embedder TRAINING runs and then, separately, "Arm A adds 1 cache pass" — but the
   "Metrics & decision rules" section pre-registers **paired McNemar per seed** for
   D vs A and C vs A. That protocol consumes one arm-A tracker output *per seed*, so
   arm A is a 3-unit cell (A:full:{0,1,2}), not a 1-unit one; the grid section
   undercounts it. The config enumerates the honest requirement:

   | cell | units |
   |---|---|
   | ablation core — B, C, D @ full pool x 3 seeds (D's full pool = 3520) | 9 |
   | identity-scaling curve — D @ {300, 1000, 2000} x 3 seeds (3520 shared with core) | 9 |
   | **embedder training subtotal (the doc's "18")** | **18** |
   | arm A ImageNet null x 3 seeds — no training; per-seed McNemar denominator | 3 |
   | detector finetunes — {yolo11s, yolo11m} x {mot17dev, mot17dev_carla} | 4 |
   | **total array units** | **25** |

   The 3 arm-A units are *not* 3 extra embedder trainings: ImageNet weights are
   seed-independent, so all three produce the same embedding and are billed as the
   cheap `embedder_null` class (0.45 h, +0.9 h over a single pass). Deduplicating them
   to one run would require special-casing the per-seed pairing in
   `sweep_launcher.aggregate()` — 0.9 h of H200 time is not worth a code fork in the
   frozen D28 McNemar path on HPC day. Enumeration is **kept at 25**; this amendment,
   not the config, is the correction.

6. **Detector wall cap 02:30:00 (pre-registered kill, not an estimate).** Each
   detector array task is killed by `timeout` at 2.5 h. The class's measured spread is
   1.5-3.0 h/run, so this cap sits *inside* the spread deliberately: it bounds the
   grid's worst case at 39.25 h under the 40 h ceiling instead of 41.25 h over it.
   Consequence, stated before the fact: a detector unit that would have needed >2.5 h
   dies leaving no result JSON, and is retried by re-submitting the array (the
   existing resume rule). If both yolo11m units die at the cap, the pre-registered
   R6 selection rule (amendment 1) is applied over whichever detector units completed,
   and the shortfall is reported — never silently. The 40 h ceiling itself does not
   move (CLAUDE.md: thresholds never move to make a gate pass).

## Val interaction (hard constraint)

val_manifest.md (D35) pins the CURRENT conv checkpoint by SHA256. If the PERUN sweep
produces a better embedder, running it at val REQUIRES a user-approved amendment to
val_manifest.md BEFORE the val event (D35 freeze rule). Default path: val runs with
the pinned checkpoint; the PERUN winner becomes a pre-registered secondary claim or a
manifest amendment — user's call, made before val, never after.

## Pre-registration amendment (D49 — 2026-08-19, post-smoke, pre-array; fires a
## lever that was itself pre-registered in amendment 6 — no new design freedom)

7. **Detector lever fired: `detector.models` reduced to `[yolo11s]`; grid 25 -> 23 units.**
   Amendment 6 pre-registered the response to a detector class that cannot fit the
   02:30:00 hard cap: *drop to one detector model rather than burn the remaining budget,
   and never raise the cap.* The smoke job (SLURM 77122, gpu04, 2026-08-19) supplied the
   measurement that triggers it.

   Measured: **yolo11s = 88 s/epoch** at the sweep's `imgsz: 960` (smoke epoch 2; epoch 1
   discarded as warmup-contaminated). At 100 epochs that is ~2.44 h — inside the cap, but
   only by ~2 s/epoch of margin.

   Projected for yolo11m by GFLOPs ratio at the same imgsz: **154.2 / 48.9 = 3.15x**, so
   88 s x 3.15 = **~277 s/epoch -> ~7.7 h per 100-epoch run**. That is ~3x the 02:30 cap,
   so both yolo11m units would be killed leaving no result JSON — 5 h of allocation spent
   for nothing. The projection is compute-scaling only and ignores yolo11m's larger
   activation memory and dataloader pressure, so ~277 s is a floor, not a point estimate;
   the verdict does not depend on the precision.

   Enumeration after the lever: 18 embedder + 3 arm-A ImageNet-null + 2 detector =
   **23 units**, `--array=0-22%8`, grid total **32.25-34.25 H200-h** against the unchanged
   **40 h ceiling**.

   Stated before the fact, as amendment 6 requires: the two surviving yolo11s detector
   units sit only ~2 s/epoch under the cap and are therefore themselves marginal. If one
   or both die at 02:30, the R6 selection rule (amendment 1) is applied over whichever
   detector units completed and the shortfall is reported in the selection — the cap is
   not raised, the ceiling does not move, and no further model substitution is made
   without a new dated amendment.

   What did NOT change: arms, pools, seeds, epochs, gate_probe, the McNemar pairs, the
   embedder classes, the 40 h ceiling, and every wall cap. The detector arm is a G2b
   *workstream* feed (D26/D31), not a G2a claim input, so narrowing it does not touch any
   pre-registered hypothesis test.
