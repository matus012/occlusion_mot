# PROPOSED second val event — DRAFT, NOT APPROVED, NOT RUN

**Status: PROPOSAL ONLY. Nothing here has been executed. Val has NOT been touched.**

Drafted 2026-08-19 because D53 decision 1 says a positive Stage-1 slope earns exactly one
thing: *the right to draft* this document for separate approval. **Drafting is not
approval.** The slope came out positive (D61), so this draft exists. Whether it is ever
run is your call and yours alone.

---

## 0 — My recommendation: DO NOT RUN THIS

I am drafting it because I was told to, and I am telling you plainly that I do not think
it should be executed as things stand.

The Stage-1 result that triggered this draft is **split**:

| | |
|---|---|
| Mechanism (`oracle_ceiling ~ mAP`, 14 levels) | slope 0.902, CI [0.784, 1.021] — **supported** |
| G2b (`id_retention >= 0.55`) | best trained detector **0.202** — **not met** |
| Best trained detector vs the baseline it must beat | **0.143 worse** end-to-end |

A val event exists to give an honest held-out read of a candidate. **There is currently no
candidate worth reading.** Every detector Stage 1 produced is worse on MOT17 than the
off-the-shelf yolo11x already in use. Spending the project's second — and, under D18, its
last uncontaminated — look at val to confirm that a worse detector is worse would burn a
scarce, non-renewable resource for a number nobody needs.

The mechanism finding does not need val either: it is a *dev-side* relationship between
detector quality and ceiling, measured on data the detectors never trained on, and it is
already honest by construction (perun_detector_v1.md section 1).

**Run this only if** a future detector clears `id_retention >= 0.55` on dev AND beats the
yolo11x baseline. Neither holds today.

## 1 — Second-look disclosure (mandatory, per D52)

If this is ever run, the following must appear alongside every number it produces. It is
not a footnote.

> **This is a SECOND val event.** The val half was already observed once, on 2026-08-09
> (D45), under `val_manifest.md`. That first look returned G1 parity PASS, a G2a-floor MISS
> (0.440 vs 0.58), and a detector-arm read (R6) in which the oracle-ceiling lift did not
> transfer. **The design being tested here was shaped by knowledge of that outcome.**
> Consequently this is not a virgin read: the nominal error rate of any test reported from
> it is not its true error rate, and it must be described as a second-look result wherever
> it appears — in the README, in any report, and in any external communication.

D18.3 makes re-running val a user-owned contamination decision. This draft does not and
cannot discharge that.

## 2 — Frozen state this manifest would bind to

Filled in at freeze time, before any run, never after. Left deliberately blank here — a
manifest that names its checkpoint after the fact is not a pre-registration.

- code commit: `<FILL — must be the commit the run executes from>`
- detector checkpoint: `<FILL path>` sha256 `<FILL>`
- embedder checkpoint: `data/models/reid_conv.pt` sha256
  `7bb4fb3c6f333fdc4fb0131e2c1b44f50c3e20c0127851e30f4c25e2fc3d801d`
- detections: cached from the detector checkpoint above, over full sequences
- segment index: `results/occlusion_segments.json` (133 val-half segments, D14)

**Selection rule:** exactly ONE detector advances, chosen on **dev** evidence and named in
this file *before* the run. No selection at val, no "best of" across val outputs — that is
the D18.2 rule and it is what makes a single pass meaningful.

## 3 — Runs (single pass each, in this order)

| id | run | purpose |
|---|---|---|
| V1 | frozen canonical tracker + the pre-committed detector, `--half val` | the G2b claim |
| V2 | identical tracker on the **yolo11x baseline** detections, `--half val` | the comparison the claim needs; without it V1 is uninterpretable |
| V3 | `paired_test.py` V1 vs V2 on the intersection denominator | paired significance, frozen D28 path |

Every run emits the full `aggregate()` set (D26 standing requirement): `n_segments`,
`id_retention`, `id_retention_assoc`, `n_assoc_scope`, `pre_match_rate`, `oracle_ceiling`,
`cov_prematched`, `center_err_coverage`, `time_err_coverage`,
`reemergence_center_err_med`, `reemergence_time_err_med`.

## 4 — Pass/fail mapping (frozen semantics, unchanged)

| criterion | key | bound | gate-bearing |
|---|---|---|---|
| G2b end-to-end | `id_retention` | >= 0.55 | YES (D26 semantics, unchanged) |
| G1 parity | HOTA / IDF1 / IDsw vs baseline | >= base − 0.5 / <= base | YES |
| center err | `reemergence_center_err_med` | <= 0.015 | YES |
| conditional coverage | `cov_prematched` | >= 0.90 | YES |
| detector-vs-baseline | V1 vs V2 paired | report p, no threshold | report-only |

## 5 — Pre-registered expectations (so outcomes cannot be re-narrated)

- Stage 1 measured a dev-side `e2e ~ mAP` slope whose **trained-only** CI includes zero
  ([−0.38, 1.26]). A val e2e improvement that exceeds what that slope predicts should be
  treated as **suspicious**, not as good news.
- The 0.55 gate does not move. A miss is reported and work halts (D18.3).
- Val n≈133 segments: this is underpowered for small effects, exactly as it was at D45.
  Non-significance is an expected outcome and is not grounds for re-running anything.

## 6 — What approving this would cost

Val has been read once. Approving a second read means the project can no longer claim an
uncontaminated held-out estimate for *anything* measured after it, and there is no third
look available to repair that. That is the whole price, stated up front.

---

**Nothing in this file is authorised. It is a draft prepared for your review under D53
decision 1, and my recommendation is to leave it unexecuted until a detector exists that
is worth spending val on.**
