# occlusion-mot

Occlusion-aware multi-object tracking with **hidden-agent state prediction** (P1) and a
**CARLA occlusion-scenario data engine** for sim2real ablation (P2).

## Headline

**Built:** an own ByteTrack-equivalent tracker (implementation equivalence frozen as G0),
a hidden-agent state module (damped motion prediction while occluded, a 90-frame buffer
sized on the measured gap *tail*, appearance-gated recovery), a CARLA occlusion-scenario
data engine that emits exact per-walker visibility ground truth, and an offline,
hash-verified HPC pipeline that executed 35 pre-registered units on TUKE PERUN H200s for
**18.46 of a 40 H200-h** budget that was never moved. **Proven:** the module clears every
ByteTrack-parity gate on held-out val (HOTA 49.98 -> 51.07, IDF1 58.28 -> 60.09, ID
switches 359 -> 298) while raising occlusion retention 0.292 -> 0.345, and end-to-end
retention is **detector-capped** -- substituting ground-truth visible boxes at the top of
an unchanged pipeline lifts retention **0.345 -> 0.786** and the recoverable-occlusion
ceiling 0.571 -> 0.970, with a 12-detector dose-response fixing the ceiling on detector
mAP at slope **0.902** (95% CI [0.784, 1.021], R2 0.949). **Null:** the trained appearance
embedder never beat an ImageNet null at tracker level at any scale tested (best paired
p = 0.143 against a frozen 0.05); all 12 detectors trained on MOT17-disjoint data scored
*below* the off-the-shelf yolo11x baseline, leaving the 0.55 retention gate **not met**;
and *sim+real > real-only* is **not established** by its own pre-registered rule.
**Bottom line:** no model trained in this project improved end-to-end tracking -- the
mechanism behind the ceiling is established, the route to exploiting it is not. **Project
closed 2026-09-04**; the three remaining forks -- a MOT17-domain detector that keeps the
disjointness that made this read honest, the ~19% tracker-side residual that survives
perfect detections, and sim-to-real transfer -- are recorded as future work and were not
pursued.

**Status (2026-08-19): pre-registered val executed as frozen (D45); the PERUN-scale
embedder sweep executed and returned a NULL (D50); the detector bottleneck located and
quantified (D55).** Dev-half tables are tuning-time numbers; the val section is the honest
held-out read. Misses are reported, not tuned away — the largest experiment in this repo
is a negative result, and it is written up as one.

![CARLA occlusion scenario with per-walker GT visibility](demo/s5_carla_excerpt.gif)

*CARLA scenario with exact per-walker visibility ground truth (green ≥0.5, orange, red
<0.25) — the P2 data engine that labels occlusion for free. Full clip:
`demo/s5_carla.mp4`.*

> **Want the 30-second version?** `demo/clips/hero.mp4` — ByteTrack baseline on the left,
> this project on the right, same video and same detections, through two real MOT17
> occlusions. Watch the ID number survive the gap on the right and reset on the left.
> (Not committed: it contains MOT17 pixels. Regenerate with
> `python scripts/render_demo.py --only hero`.)

## Problem

Trackers lose people behind people. On MOT17, ByteTrack retains the identity through
only **29% of ground-truth occlusion gaps** (dev-half, 168 segments) — 3 of 4 tracked
pedestrians come out of an occlusion as somebody else. This project (a) matches
ByteTrack on the standard metrics (HOTA/IDF1 parity gates), and (b) adds a
hidden-agent module: position estimate while occluded, re-emergence point/time, and
appearance-gated re-identification on reappearance, evaluated on occlusion segments
extracted from GT visibility.

## Findings in three lines

1. **Parity holds.** The occlusion module clears every ByteTrack-parity gate on held-out
   val — HOTA 51.07 / IDF1 60.09 / 298 ID switches, all better than baseline, so the
   identity gains below cost nothing in ordinary tracking quality.
2. **Scaling the appearance model does NOT close the gap — a pre-registered null.** 23
   PERUN units, 3 seeds, criteria frozen before the run: identity count scales *retrieval*
   monotonically (occ-rank1 0.408 → 0.610 over a 12x identity range) while tracker
   association barely moves (0.523 → 0.578) against a seed spread of 0.078 — larger than
   the whole effect. Best paired McNemar p = 0.143 against a required 0.05.
3. **The detector is the bottleneck — confirmed quantitatively, but not yet exploitable.**
   Perfect (visible) GT boxes take the recoverable-occlusion ceiling 0.571 → **0.970** and
   end-to-end retention 0.345 → **0.786**. A 12-detector dose-response then measured the
   relationship directly: ceiling scales with detector mAP at slope **0.90** (95% CI
   [0.78, 1.02], R² 0.949) — near 1:1. **But** every detector trained on MOT17-disjoint
   data came out *worse* than the off-the-shelf baseline, so G2b (0.55) is **not met**.
   The mechanism is proven; the route to exploiting it is not.

## How it was measured

| | |
|---|---|
| Benchmark | MOT17 train, half-split — first half dev (all tuning), second half val (held out) |
| Occlusion segments | extracted from GT visibility (D14): 168 dev / 133 val |
| Detections | FIXED across every tracker comparison (cached yolo11x), so only the tracker varies |
| Seeds | ≥3 on every claim at scale; single-seed deltas < 6pt are treated as noise and never claimed |
| Pre-registration | gates, selection rules and kill criteria frozen in `gates.yaml`, `val_manifest.md`, `perun_sweep_v2.md`, `perun_detector_v1.md` *before* the runs |
| Discipline | thresholds never move to make a gate pass; every deviation is a dated amendment |

## Architecture

- **Own ByteTrack** (`src/omot/track/`) — Kalman + two-stage IoU association,
  implementation-equivalence proven against a pinned reference (gate G0: ΔHOTA 0.52,
  ΔIDF1 0.07 at identical hyperparameters on identical detections).
- **Frozen-detections protocol** — the detector runs once; every tracker comparison
  consumes the identical cached detections. No comparison in this repo ever mixes
  detector outputs.
- **Hidden-state module** (`src/omot/hidden/`) — occlusion classification at track
  loss, extended coasting buffer with velocity damping (occluded pedestrians barely
  move: median gap displacement 0.02 image diagonals), a recovery association stage,
  and a cosine-distance appearance veto from a trained re-ID embedder
  (ResNet18/BN-neck, occlusion-upweighted triplet + CE, random-erasing = synthetic
  occlusion).
- **Occlusion taxonomy** — every eval emits a failure decomposition: pre-match rate,
  oracle association ceiling, association-scope vs end-to-end retention. This split
  is what separates *module* failures from *detector* failures (G2a vs G2b below).

## Methodology (the part that survives review)

| discipline | implementation |
|---|---|
| Gates G0–G4 | G0 repro **frozen PASS** · G1 parity **PASS on val** · G2 hidden-state: center-err + coverage **PASS on val**, G2a-floor **MISSED on val** (accepted, not re-tuned — D46), G2a-paired pending scale, G2b contingent on detector arm · G3 quality (pytest+ruff+coverage, live) **PASS** · G4 CARLA feeder **PASS** |
| Val freeze | single pre-registered val event (`val_manifest.md`): configs, checkpoint SHA256s, pass/fail mapping, expected-power notes frozen BEFORE the run; executed 2026-08-09 exactly as frozen (D45); no retune after — misses reported, not tuned away |
| Multi-seed discipline | measured seed noise bounds (single-seed tracker deltas <6pt are noise — logged D36); arm comparisons use ≥3-seed means; the discriminating G2a criterion is a paired McNemar test, not a point delta |
| License guards | CI-enforced: zero dataset-derived pixels tracked (per-file visual allowlist with content classes), dataset/cache extensions blocked, 2MB default-deny, train/eval identity-disjointness verified per source |

## Results (dev-half; no single-seed claims — deltas within the ±6pt seed band are not claimed)

| configuration | e2e id retention | assoc-scope retention |
|---|---|---|
| ByteTrack baseline | 0.292 | 0.505 |
| + geometric hidden-state | 0.310 | 0.531 |
| + appearance veto (ImageNet null) | 0.327 | 0.556 |
| + trained embedder | **0.345** | **0.604** |

The baseline→full-stack progression (+5.3pt e2e, +9.9pt assoc) exceeds the seed-noise
band; individual embedder-variant differences do not, and are therefore not claimed —
trained-vs-ImageNet is p=0.26 (paired McNemar, n=96) — resolved at PERUN scale and
still not significant (best p=0.143 over 3 seeds); see the sweep section below.
Re-emergence position error: 0.0055–0.0064 of the image diagonal (gate ≤0.015).

**Detector arm** *(dev-optimistic: detector finetuned ON dev-half GT and scored on
dev-half — the honest read is the pre-registered val run)*: oracle ceiling 0.583→0.845,
full-stack e2e 0.494.

## Val results (pre-registered, executed as frozen — D45, 2026-08-09)

The single val event ran `val_manifest.md` exactly: frozen configs, pinned checkpoint
SHA256s, identical cached detections, no re-runs. **Misses are reported, not tuned
away** — that discipline is the result this section exists to demonstrate.

| verdict | criterion | val | bound |
|---|---|---|---|
| **PASS** | G1 HOTA / IDF1 | 51.07 / 60.09 | ≥ baseline − 0.5 (49.98 / 58.28) |
| **PASS** | G1 ID switches | 298 | ≤ 359 (baseline) |
| **PASS** | re-emergence center err | 0.0090 | ≤ 0.015 |
| **PASS** | conditional coverage | 0.907 | ≥ 0.90 |
| **MISS** | G2a-floor assoc retention | 0.440 | ≥ 0.58 (dev: 0.604) |
| pending | G2a-paired McNemar | p = 0.41 (n = 83) | < 0.05 |

The G2a-floor miss is the finding: a **dev→val generalization gap** — assoc retention
0.604 → 0.440, and the trained-vs-ImageNet margin shrinks from +4.7pt to +2.6pt. The
paired test's non-significance was pre-registered as the expected outcome at this
sample size (underpowered pre-PERUN); both G2a criteria defer to the PERUN-scale
embedder (D46). The occlusion tracker still clears every ByteTrack-parity gate on val
while retaining more identities than the baseline (e2e 0.278 vs 0.263, assoc 0.440
vs 0.412).

**Detector arm on val** *(mandatory label: detector-arm prototype — yolo11s finetuned
10 epochs on DEV-HALF GT only; never saw val frames; val eval clean by construction;
local prototype of the D26 G2b detector workstream, not the headline claim)*: e2e
direction replicated (0.278 → 0.338, assoc 0.577), but the dev oracle-ceiling lift
did **not** transfer (0.632 → 0.586) — confirming the dev-optimism caveat above.

Full report: `results/val/val_report.md`.

![identity scaling](viz/scaling_curve_local.png)

*Identity count scales re-ID retrieval monotonically for both seeds; tracker-level
effects don't resolve at local scale — why the PERUN sweep uses log-scale pools and
≥3 seeds.*

![retention taxonomy](viz/summary_retention_grid.png)

## Data engine (P2)

| source | train ids | occluded-query eval ids | crops |
|---|---|---|---|
| MOT17 dev-half tracklets | 269 | 66 | 46.9k |
| CARLA renders (24 scenarios, exact visibility GT) | 36 | 9 | 58.2k |
| MOT20 tracklets (manifest-pinned 80/20 split) | 1,715 | 431 | 378.1k |
| Market-1501 (no vis GT → never in eval) | 1,500 | 0 | 26.0k |
| **total** | **3,520** | **506** | **509k** |

All sources license-verified with pinned provenance; nothing dataset-derived is in
the repo. A wild-clip qualitative showcase (10 permissive-stock clips, annotated) is
available locally / on request — regenerate with `scripts/showcase.py --src
showcase/sources --out showcase/renders`.

## PERUN sweep — executed. The null is the finding.

4-arm ablation (ImageNet-null / sim-only / real-only / sim+real), log-scale identity
pools {300, 1k, 2k, 3.5k} x 3 seeds, plus detector finetunes. Fully pre-registered in
[perun_sweep_v2.md](perun_sweep_v2.md) *before* any run — selection rules, gate
semantics and decision criteria all fixed in advance. Executed 2026-08-19 on TUKE PERUN
(H200): **23/23 units, zero failures, 7.74 H200-h** against a 40 h ceiling.

**Both G2a criteria fail, and that is the reported result.**

| verdict | criterion | sweep | bound |
|---|---|---|---|
| **FAIL** | G2a-floor assoc retention | 0.5779 | >= 0.58 |
| **FAIL** | G2a-paired McNemar (the module claim) | best p = 0.143 | < 0.05 |
| **not established** | P2 claim: sim+real > real-only | see below | pre-registered conjunction |
| — | detector arm (dev-optimistic) | mAP50 0.936 | informs G2b |

McNemar p-values, all seeds, no cherry-picking — D vs A: 0.143 / 0.661 / 0.500;
D vs C: 0.145 / 0.613 / 0.773; C vs A: 0.339 / 0.668 / 0.332 (n ~ 95).

![identity scaling at PERUN scale](viz/perun_sweep_identity_curve.png)

**The mechanism, which is why this is a finding and not just a miss.** Identity count
scales *retrieval* cleanly and monotonically — occ-rank1 0.408 -> 0.514 -> 0.577 ->
0.610 across a 12x identity range, still climbing at the pool ceiling. Tracker
*association* does not follow: 0.523 -> 0.540 -> 0.563 -> 0.578, an 0.055 total effect
against a per-pool seed spread reaching **0.078**. The seed noise is larger than the
entire effect. More identities buy a better embedder and not a better tracker, because
association is not what the embedder is failing at — the detector ceiling is.

That claim is not new here; it was predicted at local scale (D29, D36) and this sweep
is the properly-powered test of it. It survived.

**Caveat that constrains what may be read off this sweep — stated prominently because
it limits our own headline metric.** occ-rank1 is computed on a **per-arm gallery**
(`eval_sources = arm_sources(cfg, arm)`), so **cross-arm occ-rank1 comparisons are
invalid**. The sim-only arm trains on 36 identities and posts the sweep's *highest*
retrieval (0.875) together with its *lowest* association (0.549 — below the ImageNet
null). That is small-gallery inflation, not a result. Consequences, applied honestly:

- The within-arm identity curve above **is** valid (fixed sources, only pool varies).
- The pre-registered P2 verdict required `mean assoc(D) > mean assoc(C)` **and**
  `mean occ-rank1(D) > (C)`. The first holds (0.5779 > 0.5674); the second does not
  (0.6095 < 0.6180) and is cross-arm-invalid anyway. Per the rule as written, **"sim+real
  beats real-only" is not established.**
- G2a-paired is untouched by this: it is tracker-level on a fixed intersection
  denominator shared by both configs under test.

Recorded as [perun_sweep_v2.md](perun_sweep_v2.md) amendment 8, which annotates the
original "primary at sweep scale" line in place rather than rewriting it.

**Budget estimates were 4.3x conservative** (7.74 h actual vs 32.25-34.25 h projected;
`UNIT_EST_H` was RTX 4060-derived). The pre-registered numbers are *not* edited after
the fact — they bounded the run correctly. The measured basis is carried into the next
pre-registration as a documented recalibration (amendment 9).

**Where this leaves the project.** Appearance re-ID at PERUN scale does not deliver
G2a; that line is closed and reported as a negative result. The remaining headroom is
the detector/oracle-ceiling workstream (G2b), pre-registered separately in
[perun_detector_v1.md](perun_detector_v1.md).

## Detector workstream (G2b) — bottleneck located, dose-response in flight

Pre-registered in [perun_detector_v1.md](perun_detector_v1.md) (frozen D53) *before* any
run, with the kill criterion written to fire.

**Stage 0 — perfect-detector kill gate.** Substitute ground-truth boxes for detections at
the top of an otherwise unchanged pipeline. GT is the supremum of any detector, so if it
cannot reach the 0.55 target, no detector can and the workstream stops. ~0.2 GPU-h to
falsify a ~21 h commitment.

| detection source | pre-match | oracle ceiling | end-to-end retention |
|---|---|---|---|
| yolo11x (current) | 0.762 | 0.571 | 0.345 |
| **GT, visible only** (detector supremum) | 0.994 | **0.970** | **0.786** |
| GT, all annotations (*not* achievable) | 1.000 | 1.000 | 0.952 |

**Passed by +0.236.** Detection quality accounts for nearly the whole ceiling gap with the
tracker untouched. Two caveats kept attached: 0.786 is an *upper bound*, and even at
perfect detection ~19% of recoverable segments are still lost **inside the tracker** — a
residual no detector can close.

The `visible only` distinction is load-bearing and was fixed before any number was seen
(amendment 1): MOT17 annotates fully-occluded targets, 9.3% of all boxes. Feeding those in
would be a detector that sees through occluders, handing the module the answer it exists
to infer.

**Stage 1 — dose-response: mechanism CONFIRMED, intervention FAILED.** 12 detectors
(2 models x 2 mixes x 3 seeds) trained **only on MOT17-disjoint sources** (MOT20 + CARLA),
so the read is honest by construction and consumed no val. 12/12 units, zero failures,
10.72 H200-h.

![detector dose-response](viz/stage1_dose_response.png)

| | result |
|---|---|
| **Primary** `oracle_ceiling ~ mAP50-95` (14 levels) | slope **0.902**, 95% CI **[0.784, 1.021]**, R² 0.949 |
| Secondary `id_retention ~ mAP50-95` | slope 0.774, CI [0.731, 0.816], R² 0.991 |
| **Mechanism** (CI excludes zero) | **SUPPORTED** |
| **G2b** `id_retention >= 0.55` | **NOT MET** — best trained 0.202 |

**The mechanism is confirmed and the intervention still failed, which is the whole
result.** Detector quality drives the recoverable-occlusion ceiling almost 1:1 — the
strongest confirmation yet that end-to-end retention is detector-capped. But every one of
the 12 trained detectors is **worse on MOT17 than the off-the-shelf yolo11x** it was meant
to beat (mAP 0.189–0.228 vs 0.397; best e2e 0.202 vs the baseline's 0.345).

**Why — the honesty tax.** Training only on MOT17-disjoint data is what made this read
trustworthy, and it is also what sank it: MOT20 is a far denser, different-domain
benchmark, and the domain gap cost more than finetuning gained. The trained detectors
over-fire badly (72k–226k detections vs the baseline's 80k for the same 53,678 GT boxes).
The design bought honesty at the price of the quality it was trying to demonstrate.

**Caveat that qualifies the headline.** The 12 trained units span only 0.039 of the mAP
axis; the fit's lever arm comes from the two reference points. Restricted to trained units
alone the ceiling relationship survives (slope 2.24, CI [0.76, 3.72]) but the end-to-end
one does **not** — slope 0.44, CI **[−0.38, 1.26]**, which includes zero. The
pre-registered verdict stands as defined over all 14 levels; this is recorded so it is not
read as more than it is.

**Open problem, stated plainly:** the actionable next step is a better *MOT17-domain*
detector, which conflicts with the disjointness that made this read honest. That tension
is unresolved.

## Quickstart

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt -e .
.venv/Scripts/python.exe -m pytest -q          # full suite incl. license/disjointness guards
.venv/Scripts/python.exe scripts/demo_synthetic.py --render demo.mp4   # dataset-free demo
.venv/Scripts/python.exe scripts/render_demo.py --all                  # regen demo clips (needs MOT17)
```

**Guided tour:** [demo/README.md](demo/README.md) — 10 minutes, S0–S8, from the
failure mode to the gate scoreboard.

## Repo map

- `src/omot/` — loaders, MOT IO, detection/embedding caches, tracker, hidden-state module, eval (incl. paired test)
- `mission.md` / `gates.yaml` / `status.txt` / `context.md` — mission, executable gates, state, full decision log D1–D67
- `val_manifest.md` / `perun_sweep_v2.md` / `perun_detector_v1.md` — frozen val pre-registration; executed sweep design + post-execution amendments; frozen G2b detector pre-registration
- `occlusion_mot_plain.md` — one-page plain-English overview: what was found, what it means, what's next
- `configs/sweep/` + `scripts/sweep_*.py` — config-driven sweep (identical entrypoint local/SLURM)
- `demo/` — guided tour + committed CARLA media; `scripts/showcase.py` — wild-clip renderer
- `runloop.ps1` — autonomous development loop (invokes `claude -p` per iteration against the gates)

## Licensing & data

License: **AGPL-3.0-only**, because the direct dependency `ultralytics` is itself
AGPL-3.0 (see `context.md` D9). Datasets (MOT17, MOT20, CrowdHuman) are used under
their non-commercial research licences for evaluation and for detector
fine-tuning / re-ID experiments only — **no weights trained on them are released**;
trained checkpoints stay local (gitignored) and are never committed or attached to
a release. Full dependency list with licences, and dataset/simulator/compute notes:
[THIRD_PARTY.md](THIRD_PARTY.md).
