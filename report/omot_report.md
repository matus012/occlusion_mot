# Occlusion-aware multi-object tracking with hidden-agent state prediction

**A pre-registered study, including two negative results**

Matus Filo · 2026-08-19 · [github.com/matus012/occulsion_mot](https://github.com/matus012/occulsion_mot)

---

## Abstract

Multi-object trackers lose identities through occlusion. On MOT17 dev-half, our ByteTrack
reimplementation retains the correct identity through only **29.2 %** of ground-truth
occlusion gaps (`results/hidden_dev_audit_base.json`). We add a *hidden-state module* that
maintains a motion estimate for occluded targets and gates re-identification on appearance,
raising retention to **34.5 %** while *improving* standard tracking quality — on held-out
val, HOTA 49.98 → 51.07, IDF1 58.28 → 60.09, ID switches 359 → 298
(`results/baseline_ours.json`, `results/val/val_report.md`).

We then report two pre-registered experiments that did **not** work, because they locate
the actual bottleneck. (i) Scaling the appearance embedding to 3,520 identities on 23
H200 units improves *retrieval* monotonically (occ-rank1 0.408 → 0.610) but leaves
*tracker association* inside seed noise (0.523 → 0.578, per-pool seed spread up to 0.078);
the pre-registered paired test against an ImageNet null fails at every seed (best
p = 0.143) (`results/sweep/perun_full/summary.json`). (ii) Substituting ground-truth boxes
for detections lifts the recoverable-occlusion ceiling 0.571 → 0.970 and end-to-end
retention 0.345 → 0.786 (`results/hidden_dev_stage0_gtvis.json`), and a 12-detector
dose-response confirms the ceiling scales with detector quality at slope **0.902**
(95 % CI [0.784, 1.021], R² 0.949) (`results/sweep/perun_detector/summary.json`).

The mechanism is therefore established: end-to-end retention is detector-capped. The
intervention nonetheless failed — every detector we trained on MOT17-disjoint data scored
*below* the off-the-shelf baseline it was meant to beat. We call this the honesty tax and
report it rather than retrying on contaminated data. All gates, thresholds and decision
rules were frozen before the runs; no threshold was moved afterwards.

---

## 1. Problem

A tracker assigns a persistent ID to each person across frames. Occlusion breaks this: when
a pedestrian passes behind an obstacle or another pedestrian, detections stop, the track is
terminated or drifts, and re-appearance is registered as a new identity. Every downstream
use — counting, flow estimation, trajectory analysis, person-following — degrades when
identity resets.

We quantify the failure directly rather than through aggregate tracking metrics, which
average it away. From MOT17 ground-truth visibility we extract **occlusion segments**: a
run of frames for one identity bounded by two solidly-visible annotations (visibility
≥ 0.5), during which visibility stays below 0.5 and dips below 0.25 at least once. This
yields **168 dev-half** segments (`n_segments` in every `results/hidden_dev_*.json`) and
**133 val-half** (`results/occlusion_segments.json:stats_val_half`); the index also records
328 over full sequences, which exceeds the halves' sum because some segments straddle the
midpoint. Definition in `context.md` D14. The primary quantity
is `id_retention`: the fraction of segments where the identity before the gap equals the
identity after it.

**Baseline: 0.2917** (`results/hidden_dev_audit_base.json`). Roughly three of four tracked
pedestrians emerge from an occlusion as somebody else.

The gap-length distribution explains why a longer buffer is the right lever, and it is the
*tail* that does so, not the centre: median gap **25.0** frames but 90th percentile **97.3**
(full sequences; 76.8 on val-half) — `results/occlusion_segments.json:stats_full`. The
median sits *inside* ByteTrack's default 30-frame track buffer; the top decile sits at
roughly three times it. A buffer sized on the median would miss exactly the segments that
matter.

## 2. Method

### 2.1 Base tracker

An own ByteTrack implementation (`src/omot/track/`): Kalman filter with two-stage IoU
association (high-confidence, then low-confidence). Implementation equivalence against a
pinned reference was frozen as gate G0 before any of the work below: ΔHOTA 0.52,
ΔIDF1 0.07, both within the ≤ 1.0 bound (`results/g0_equivalence/`, `context.md` D15).
This matters because every comparison here is *ours-vs-ours*; the baseline row is our own
tracker, not a number copied from a paper.

### 2.2 Hidden-state module

For a track that loses its detection, the module maintains a *hidden state* rather than
deleting the track:

1. **Occlusion classification** — a lost track is distinguished from a departed one using
   overlap with other tracks and frame geometry.
2. **Damped motion prediction** — the Kalman prediction continues with damping (1.0 in the
   frozen configuration), producing a position estimate while the target is invisible.
3. **Extended buffer** — 90 frames rather than 30, sized against the measured gap *tail*
   (p90 = 97.3 frames), not the median (25.0).
4. **Appearance-gated recovery** — on re-appearance, a candidate detection is accepted only
   if its embedding distance to the stored appearance falls under a gate (0.45), preventing
   the buffer from becoming an ID-donation machine.

The frozen canonical configuration is `b90 / damping 1.0 / recover-gate 1.5 /
overlap 0.25 / lowconf kf / app-gate 0.45` with the trained `conv` embedder. It is
byte-identical between the val manifest and the detector study
(`val_manifest.md` R1; `scripts/detector_unit.py:FROZEN_TRACKER`).

### 2.3 Data engine

Appearance re-ID needs occluded-query training data with visibility labels, which MOT
datasets do not provide directly. We built a CARLA 0.9.15 scenario generator producing
MOT-format sequences with **exact per-walker visibility ground truth** by construction:
**24 scenarios, 225 occlusion segments**, validated against an independent analytic
backend with **zero failures across all 24 scenarios**
(`results/carla_feeder.json`, `results/sim_agreement.json:failures`; gate G4 PASS).

The combined re-ID corpus is 3,520 training identities / 509k crops from four sources —
MOT17 dev-half tracklets, CARLA renders, MOT20 tracklets, Market-1501 — with an
identity-disjoint split and pinned licence provenance. Market-1501 carries no visibility
ground truth and is therefore **never** used in evaluation, only training.

## 3. Experimental protocol

The protocol is the part of this work we would defend first.

**Half-split.** MOT17 train sequences are split in half. The first half (dev) carries *all*
tuning, model selection and ablation. The second half (val) is touched **once**. This is
binding (`context.md` D18).

**Executable gates.** `gates.yaml` holds every threshold as a machine-readable bound, and
`scripts/check_gates.py` is its executable interpretation. Relevant here:

| gate | bound | meaning |
|---|---|---|
| G1 parity | HOTA/IDF1 ≥ baseline − 0.5, IDsw ≤ baseline | the module must not damage ordinary tracking |
| G2 center-err | ≤ 0.015 of image diagonal | hidden-state position estimate is usable |
| G2 conditional coverage | ≥ 0.90 | predictions exist when they can |
| G2a-floor | `id_retention_assoc` ≥ 0.58 | regression floor on association scope |
| G2a-paired | McNemar p < 0.05 | **the discriminating criterion** — trained embedder vs ImageNet null |
| G2b | `id_retention` ≥ 0.55 | end-to-end retention |

**Pre-registration.** Each large experiment was fully specified before execution:
`val_manifest.md` (the single val event), `perun_sweep_v2.md` (the embedder sweep),
`perun_detector_v1.md` (the detector study). Each fixes arms, pools, seeds, selection rules,
analysis and *kill criteria* in advance. Changes after freezing are appended as dated
amendments; **history is never rewritten**, and no threshold was ever moved to make a gate
pass.

**Seed discipline.** Local experiments established that single-seed tracker deltas below
~6 pt are noise at n ≈ 97 (`context.md` D36). Every claim at scale therefore uses ≥ 3 seeds,
and differences inside the seed band are not claimed.

**Fixed detections.** Tracker-vs-tracker comparisons use identical cached detections
(`data/cache/detections/*__yolo11x.npz`), so only the tracker varies. The detector study
deliberately *inverts* this: the tracker is frozen and the detection source varies. Both
directions are stated where used.

## 4. Results

### 4.1 Parity holds, and the module works on held-out data

Dev-half progression, n = 168 segments (each row a committed file):

| configuration | e2e retention | assoc-scope | cond. coverage | center err | source |
|---|---|---|---|---|---|
| ByteTrack baseline | 0.2917 | 0.5052 | 0.8527 | 0.00690 | `results/hidden_dev_audit_base.json` |
| + geometric hidden-state | 0.3095 | 0.5306 | 0.8931 | 0.00718 | `results/hidden_dev_audit_geom.json` |
| + ImageNet appearance veto | 0.3274 | 0.5556 | 0.8939 | 0.00706 | `results/hidden_dev_audit_in45.json` |
| + trained embedder | **0.3452** | **0.6042** | 0.9141 | 0.00639 | `results/hidden_dev_conv_app45.json` |

The baseline → full-stack progression (+5.3 pt e2e, +9.9 pt assoc) exceeds the seed-noise
band. Individual embedder-variant differences do not, and are not claimed.

**Val (single pre-registered pass, executed exactly as frozen; `results/val/val_report.md`,
`context.md` D45):**

| verdict | criterion | val | bound |
|---|---|---|---|
| **PASS** | G1 HOTA / IDF1 | 51.07 / 60.09 | ≥ 49.98 / 58.28 − 0.5 |
| **PASS** | G1 ID switches | 298 | ≤ 359 |
| **PASS** | center err | 0.0090 | ≤ 0.015 |
| **PASS** | conditional coverage | 0.907 | ≥ 0.90 |
| **MISS** | G2a-floor | 0.440 | ≥ 0.58 |
| not met | G2a-paired | p = 0.4073 (n = 83) | < 0.05 |

Parity is the load-bearing pass: the identity gains cost nothing in standard tracking
quality — ID switches *fall* by 17 %. The G2a-floor miss is a genuine dev→val
generalisation gap (0.604 → 0.440), reported and not tuned away; D18.3 halted work on
val-gated claims pending an explicit decision.

The val event also carried a detector-arm secondary, whose mandatory label we reproduce
verbatim from `val_manifest.md` R6:

> "detector-arm prototype — yolo11s finetuned 10 epochs on DEV-HALF GT only; never saw val
> frames; val eval clean by construction; local prototype of the D26 G2b detector
> workstream, not the headline claim."

It replicated the e2e direction (0.278 → 0.338) but the dev oracle-ceiling lift did **not**
transfer (0.632 → 0.586) — the first sign that the detector story was weaker than dev
suggested.

### 4.2 Negative result I: scaling the appearance embedding does not help the tracker

**Pre-registered as `perun_sweep_v2.md`.** Four arms (A: ImageNet null, no training;
B: sim-only; C: real-only; D: sim+real), identity pools {300, 1000, 2000, 3520} × 3 seeds,
23 units on TUKE PERUN H200s, 7.74 H200-h. All results:
`results/sweep/perun_full/summary.json`.

Arm-D identity curve (gate 0.45, mean of 3 seeds, from the 12 committed `D_p*.json` units):

| training identities | occ-rank1 (retrieval) | assoc retention (tracker) | seed spread |
|---|---|---|---|
| 300 | 0.4081 | 0.5228 | 0.0736 |
| 1000 | 0.5143 | 0.5395 | 0.0722 |
| 2000 | 0.5771 | 0.5626 | 0.0269 |
| 3520 | 0.6095 | 0.5779 | 0.0776 |

Retrieval climbs monotonically across a 12× identity range and is **still rising** at the
pool ceiling. Tracker association moves 0.055 in total — against a per-pool seed spread
reaching **0.078**, i.e. the noise is larger than the entire effect.

The discriminating criterion, McNemar on the intersection denominator, per seed
(n ≈ 95), all values reported, no cherry-picking:

| pair | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| D vs A (the module claim) | 0.1431 | 0.6612 | 0.5000 |
| D vs C | 0.1445 | 0.6128 | 0.7734 |
| C vs A | 0.3388 | 0.6682 | 0.3318 |

Nothing approaches 0.05. G2a-floor also fails at scale: arm D 0.5779 vs the frozen 0.58.
**Both limbs fail.** The trained embedder is not distinguishable from an ImageNet-pretrained
ResNet18 at tracker level, at full scale, with three seeds.

**A caveat that invalidates one of our own metrics.** `sweep_unit.py` sets
`eval_sources = arm_sources(cfg, arm)`, so each arm is scored on a retrieval gallery drawn
from *its own* sources. Arm B (36 sim identities) posts the sweep's **highest** retrieval
(occ-rank1 0.8751) together with its **lowest** tracker association (0.5487 — below the
ImageNet null's 0.5567). That is small-gallery inflation, not a result. Consequently
**cross-arm occ-rank1 comparisons are invalid**, which demotes the metric that
`perun_sweep_v2.md` had named "primary at sweep scale" (recorded as amendment 8). The
within-arm identity curve above remains valid — fixed sources, only pool size varies — and
the paired criterion is untouched, being tracker-level on a shared denominator.

By the same rule, the project's P2 claim (*sim+real > real-only*) required
`mean assoc(D) > (C)` **and** `mean occ-rank1(D) > (C)`. The first holds (0.5779 > 0.5674);
the second fails (0.6095 < 0.6180) and is cross-arm-invalid anyway. **Reported as not
established.**

### 4.3 Locating the bottleneck: the perfect-detector ceiling

**Pre-registered as `perun_detector_v1.md` §2, with a kill criterion written to fire.**
Substitute ground-truth boxes for detections at the top of an otherwise unchanged pipeline;
the tracker is frozen. GT is the supremum of any detector's output, so if it cannot reach
the 0.55 target, no detector can.

| detection source | pre-match | oracle ceiling | e2e retention | assoc | source |
|---|---|---|---|---|---|
| yolo11x (current) | 0.7619 | 0.5714 | 0.3452 | 0.6042 | `results/hidden_dev_stage0_yolo11x.json` |
| **GT, visible only** | 0.9940 | **0.9702** | **0.7857** | 0.8098 | `results/hidden_dev_stage0_gtvis.json` |
| GT, all annotations | 1.0000 | 1.0000 | 0.9524 | 0.9524 | `results/hidden_dev_stage0_gtall.json` |

The kill fires below 0.55; measured **0.7857**. Detection quality accounts for almost the
entire ceiling gap with the tracker untouched.

The `visible only` split is load-bearing and was fixed *before any number was seen*
(amendment 1). MOT17 annotates fully-occluded targets — 10,485 boxes, 9.3 % of all
consider-flagged pedestrian annotations. A stream including them is not a detector: it sees
through occluders, handing the hidden-state module the very answer it exists to infer. The
gate-bearing arm is therefore `gtvis`; `gtall` is reported as an upper-*upper* bound only.

### 4.4 Negative result II: the dose-response, and the honesty tax

**Pre-registered as `perun_detector_v1.md` §3.** 12 detectors (yolo11s/yolo11m × two
MOT17-disjoint mixes × 3 seeds), 10.72 H200-h, trained **only** on MOT20 + CARLA so the
MOT17 dev read is honest by construction and consumes no val. Per-unit label, verbatim from
each result file:

> "MOT17-disjoint training (perun_detector_v1.md s1) -- NOT dev-optimistic"

Primary analysis, OLS across 14 detector-quality levels (12 trained + yolo11x baseline +
GT), `results/sweep/perun_detector/summary.json`:

| response | slope | 95 % CI | R² |
|---|---|---|---|
| `oracle_ceiling ~ mAP50-95` | **0.9021** | **[0.7837, 1.0205]** | 0.9489 |
| `id_retention ~ mAP50-95` | 0.7738 | [0.7313, 0.8163] | 0.9907 |

The CI excludes zero: **the detector-ceiling mechanism is supported**, at near 1:1 — a point
of detector mAP buys about a point of recoverable-occlusion ceiling.

**And G2b is not met.** Best trained unit `yolo11s:mot20_carla:2` reaches e2e **0.2024**
against the frozen 0.55 — and **0.1428 below the yolo11x baseline's 0.3452**. Every trained
detector is worse on MOT17 than the off-the-shelf detector it was meant to improve on:
mAP50-95 0.1891–0.2282 vs the baseline's 0.3969; ceiling 0.2321–0.3452 vs 0.5714.

**The honesty tax.** The decision that made this measurement trustworthy — training only on
MOT17-disjoint sources — is the same decision that sank it. MOT20 is a far denser,
different-domain benchmark; the domain gap cost more accuracy than finetuning gained. The
trained detectors over-fire badly: 72k–226k detections over MOT17 dev-half against the
baseline's 80k for the same 53,678 ground-truth boxes. We report this rather than retrying
on MOT17-trained detectors, which would restore the `dev-optimistic` contamination the
design existed to remove.

**Sensitivity, reported because it qualifies the headline.** The pre-registered fit spans
mAP 0.189 → 1.000, but the 12 trained units occupy only 0.0391 of that range; the lever arm
comes from the two reference points. Restricted to trained units alone, the ceiling
relationship survives (slope 2.2374, CI [0.7556, 3.7192], R² 0.467) but the end-to-end one
does **not**: slope 0.4375, CI **[−0.3808, 1.2558]**, R² 0.099 — includes zero. The
pre-registered verdict stands as defined over all 14 levels; this is recorded so it is not
read as more than it is.

## 5. Limitations

**A tracker-side residual no detector can close.** Even with perfect visible detections,
assoc-scope retention is **0.8098**, not 1.0 (`results/hidden_dev_stage0_gtvis.json`):
~19 % of *associable* segments are still lost inside the tracker. This bounds G2b from
above even under a flawless detector, and no amount of detector work addresses it. It is the
clearest target we can name for future work.

**The disjointness tension is unresolved.** §4.4 leaves a genuine conflict: the actionable
next step is a better *MOT17-domain* detector, but training on MOT17 is exactly what makes
the read dev-optimistic. Resolving it needs either a third held-out split (we do not have
one — MOT17 train is already halved) or a detector trained on a domain closer to MOT17 than
MOT20 while remaining disjoint from it. We state the problem rather than picking whichever
option flatters the numbers.

**Underpowered paired tests.** Val carries n ≈ 83 association-scope segments and dev n ≈ 95.
Non-significance at these sizes was pre-registered as the expected outcome and is not
evidence of absence — though at PERUN scale, with three seeds, the effect being sought is
demonstrably smaller than the seed noise.

**Val is spent.** It was read once (D45). A second read was drafted when the Stage-1 slope
came back positive, and **rejected** — there was no candidate worth reading, since every
trained detector fell below the baseline (`PROPOSED_val2_manifest.md`, marked REJECTED;
`context.md` D63). Any future val event must be reported as a second-look result with the
first look disclosed, or its nominal error rate is not its true error rate.

**Single benchmark.** Everything here is MOT17 (with MOT20/CARLA/Market-1501 as training
sources only). Generalisation to other benchmarks is untested.

**occ-rank1 is not cross-arm comparable** (§4.2). Any reader tempted to compare arms on it
should not.

## 6. Reproducibility appendix

**Compute budget.** A 40 H200-h ceiling was pre-registered and never moved.

| workstream | units | H200-h | source |
|---|---|---|---|
| Embedder sweep | 23 | 7.74 | `results/sweep/perun_full/*.json` |
| Detector Stage 0 | 3 arms | ~0 (ran on a local RTX 4060) | `results/hidden_dev_stage0_*.json` |
| Detector Stage 1 | 12 | 10.72 | `results/sweep/perun_detector/*.json` |
| **Total** | | **18.46 / 40** | 21.54 h unspent |

Both grids came in far under their estimates: the embedder sweep was projected at
32.25–34.25 h and cost 7.74 h, because `UNIT_EST_H` was derived from RTX 4060 timings and
runs ~4.3× conservative on H200. The pre-registered numbers were **not** retro-edited; the
corrected basis is recorded as amendment 9 and carried forward into the detector study,
whose cost model was validated out-of-sample at −7.7 % error.

**Seeds.** Every scale claim uses seeds {0, 1, 2}. Seeding covers `random`, `numpy` and
`torch` at each entry point. Per-pool seed spreads are reported alongside every mean; the
6-pt noise band (D36) is applied as a claiming rule, not a footnote.

**Amendments to frozen documents** — all appended, dated, never rewriting history:

| doc | amendment | substance |
|---|---|---|
| `perun_sweep_v2.md` | 5–6 (D48) | array enumeration; detector wall cap as a pre-registered kill |
| `perun_sweep_v2.md` | 7 (D49) | detector lever fired on measured throughput; grid 25 → 23 |
| `perun_sweep_v2.md` | 8 (D51) | occ-rank1 demoted to a within-arm diagnostic |
| `perun_sweep_v2.md` | 9 (D51) | `UNIT_EST_H` recalibration, recorded not retro-applied |
| `perun_detector_v1.md` | 1 (D54) | `gtvis`/`gtall` split; gate-bearing arm fixed before any number |
| `perun_detector_v1.md` | 2 (D60) | dose-response x-axis corrected to MOT17-side mAP |
| `val_manifest.md` | post-hoc note (D46) | hash remap after a history purge; no semantic edit |

**Two defects worth disclosing**, both caught before they reached a result:

1. The emitted SLURM array passed `HH:MM:SS` to coreutils `timeout`, which rejects it; all
   23 tasks died in under a second, consuming no GPU time. The test suite had only ever
   asserted the `#SBATCH --time` line. Fixed with a mutation-tested grammar guard
   (`context.md` D49).
2. The dose-response x-axis initially read mAP from ultralytics' `results.csv`, which
   validates on the *MOT20 training split* rather than MOT17. Its failure mode was a
   **false null**: near-zero x-variance would have produced a wide CI that the
   pre-registered kill criterion reads as "mechanism not established". Corrected and
   instrument-proved before use — the GT arm scores mAP50-95 = **1.0000** exactly, and the
   yolo11x baseline 0.3969 (`results/detmap_*.json`, `context.md` D60).

**Environment.** Python 3.11, PyTorch 2.13 + CUDA 12.6, ultralytics 8.4.104. Dependencies
pinned in `requirements-lock.txt`. Cluster runs used an offline, hash-verified transfer
bundle (`scripts/make_hpc_bundle.py`); the deployed cluster tree is recorded as
`bundle 8421645 + patch 4c97f1d` with a sha256 table in `READY.md`.

**Everything is re-derivable.** Gates: `python scripts/check_gates.py`. Sweep aggregation:
`python scripts/sweep_launcher.py --config configs/sweep/perun_full.yaml --mode local`.
Detector aggregation: `python scripts/detector_launcher.py --config
configs/sweep/perun_detector.yaml --mode local`. Figures: `scripts/plot_stage0.py`,
`scripts/plot_stage1.py`. Full decision log: `context.md` D1–D64.

---

## Summary of claims

| claim | status | evidence |
|---|---|---|
| Occlusion module improves identity retention without harming tracking quality | **supported**, held-out | `results/val/val_report.md` |
| Retention reaches the 0.55 end-to-end target | **not met** (0.345 dev / 0.278 val) | `results/hidden_dev_conv_app45.json` |
| Trained embedding beats an ImageNet null at tracker level | **not established** at any scale tested | `results/sweep/perun_full/summary.json` |
| sim+real beats real-only (P2) | **not established** by its own pre-registered rule | ibid. |
| End-to-end retention is detector-capped | **supported** (slope 0.902, CI [0.784, 1.021]) | `results/sweep/perun_detector/summary.json` |
| A better detector is achievable by MOT17-disjoint training | **refuted** — all 12 fell below baseline | ibid. |
