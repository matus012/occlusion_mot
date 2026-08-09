# D45 — D18 val event, executed per val_manifest.md (frozen f2e0102)

Date: 2026-08-09. User approval: D45 directive. Zero manifest edits, zero config
deviations. All runs consumed the identical frozen yolo11x detection cache
(D1, full-sequence, verified pre-flight) — R6 consumed the pre-registered
yolo11s_ft cache. NO re-runs, NO tuning were performed.

## Pre-flight (all green before any run)

- reid_conv.pt sha256 `7bb4fb3c…d801d` — MATCH; det best.pt sha256 `07b45511…ea003` — MATCH
- MOT17 data re-verified: manifest sha256 `49c218a3…` == D13 pin, 0 failures
- occlusion_segments.json: 133 val-half segments (== manifest pin)
- pytest -q: exit 0, 240 tests (incl. license + disjointness guards); CUDA probe ok
- Code drift vs frozen commit: `bc46f72` is the pre-purge hash (D41 rewrite);
  rewritten equivalent `a8e4da0` (D28-final). Only val-critical diff since:
  `src/omot/detect/cache.py` +9/−2 (L5 cache_tag routing, required by the
  manifest's own R6, present before the manifest freeze). Paired-test path
  byte-identical.

## Standing taxonomy report (D26) — all runs, val half, n_segments 133

| run | pre_match | oracle_ceiling | assoc retention (n) | e2e retention | cov_prematched | center_err | time_err |
|---|---|---|---|---|---|---|---|
| R1 canonical (conv) | 0.805 | 0.632 | **0.440** (84) | 0.278 | 0.907 | 0.0090 | 26.0 |
| R2 null (ImageNet) | 0.805 | 0.654 | 0.414 (87) | 0.271 | 0.925 | 0.0169 | 26.0 |
| R3 proto (report-only) | 0.805 | 0.639 | 0.435 (85) | 0.278 | 0.916 | 0.0136 | 26.0 |
| R4 ByteTrack ref (report-only) | 0.812 | 0.639 | 0.412 (85) | 0.263 | 0.926 | 0.0155 | 25.5 |
| R6 detector-arm (see label) | 0.782 | 0.586 | 0.577 (78) | 0.338 | 0.894 | 0.0072 | 25.5 |

R1 G1 (TrackEval, val half): HOTA 51.07 / IDF1 60.09 / MOTA 45.28 / IDsw 298
(baseline_ours: 49.98 / 58.28 / 45.23 / 359).

## R5 — G2a-paired (McNemar, valcanon vs valnull)

n_intersection = 83, retained 37 vs 35 (0.446 vs 0.422), discordant 10/8,
**p = 0.4073** (criterion p < 0.05: NOT MET). Pre-registered expectation held:
underpowered pre-PERUN (manifest predicted non-significance at n_assoc ~76;
actual 84/87 → intersection 83); per manifest this is NOT grounds for re-tuning —
the criterion stays pending until the PERUN-scale embedder.

## Gate mapping (per manifest table; evaluated on R1 unless noted)

| criterion | value | bound | verdict |
|---|---|---|---|
| center err | 0.0090 | <= 0.015 | **PASS** |
| conditional coverage | 0.907 | >= 0.90 | **PASS** |
| G2a-floor | 0.440 | >= 0.58 | **FAIL** (−0.16, ~2.4σ below — NOT within noise) |
| G2a-paired (R5) | p 0.4073 | < 0.05 | **NOT MET** (pre-registered expected; pending PERUN) |
| G2b end-to-end | 0.278 | >= 0.55 | contingent, non-blocking pre detector-at-scale (R6 informs) |
| time err | 26.0 | <= 5 | not gate-bearing (provisional, freeze_order last) |
| segment count | 133 | >= 100 | **PASS** |
| G1 HOTA/IDF1 | 51.07 / 60.09 | >= 49.48 / 57.78 | **PASS** |
| G1 IDsw | 298 | <= 359 | **PASS** |

check_gates.py: G0 PASS, G1 PASS, G2 **FAIL**, G3 PASS, G4 PASS.

## R6 — mandatory label (verbatim per manifest)

"detector-arm prototype — yolo11s finetuned 10 epochs on DEV-HALF GT only; never
saw val frames; val eval clean by construction; local prototype of the D26 G2b
detector workstream, not the headline claim."

Claim tested: G2b e2e retention direction + oracle-ceiling lift replicating on
val (dev showed 0.583 → 0.845). Outcome: **direction replicated, lift did NOT**:
e2e 0.278 → 0.338 (+6.0pt, direction holds), assoc 0.440 → 0.577 (+13.7pt),
but oracle_ceiling 0.632 → 0.586 (DOWN) and pre_match 0.805 → 0.782 (down) —
the dev oracle-ceiling lift (detector trained on dev GT) did not transfer.
Dev-optimistic caveat pre-registered in the manifest; this is the first honest
read of the arm.

## Dev → val generalization (context, not a gate)

R1 assoc 0.604 (dev) → 0.440 (val); null 0.557 → 0.414; conv−null margin
+4.7pt (dev) → +2.6pt (val). Val oracle ceiling 0.632 vs dev 0.577.

## Manifest-mandated consequence

G2a-floor MISSED → per D18.3 and the manifest header: reported here, NO
re-tuning, NO re-runs, NO config changes — **work HALTS pending user decision.**
