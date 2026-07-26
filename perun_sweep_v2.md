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

| block | runs | est. per run | subtotal |
|---|---|---|---|
| embedder training (60 ep x 400 batches) | 18 | <= 0.5 h | <= 9 h |
| embedding-cache passes (7 MOT17 seqs) | ~13 | ~0.2 h | ~2.6 h |
| detector finetunes (~100 ep) | 4 | 1.5-3 h | <= 12 h |
| eval/tracker runs | — | CPU | — |
| **total** | | | **<= 25 H200h** (ceiling 40) |

SLURM shape: job array over (arm, pool, seed); device/seed/batch injected (D8);
data staged as jpg trees (~1.5 GB) + caches; checkpoints + result JSONs synced back;
no code changes needed beyond a launcher script (to be written after this doc is
approved).

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

## Val interaction (hard constraint)

val_manifest.md (D35) pins the CURRENT conv checkpoint by SHA256. If the PERUN sweep
produces a better embedder, running it at val REQUIRES a user-approved amendment to
val_manifest.md BEFORE the val event (D35 freeze rule). Default path: val runs with
the pinned checkpoint; the PERUN winner becomes a pre-registered secondary claim or a
manifest amendment — user's call, made before val, never after.
