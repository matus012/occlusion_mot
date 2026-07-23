# Val manifest — frozen pre-registration of the single D18 val event (D35)

Committed as VAL-INVARIANT. The val run executes EXACTLY this manifest; any deviation
requires a user-approved amendment to this file BEFORE the run. Val remains blocked on
explicit user approval (D18). After val numbers are seen: NO re-tuning, NO re-runs, NO
config changes — misses are reported and work HALTS pending user decision (D18.3).

Frozen state this manifest binds to:
- code: commit bc46f72 (paired-test path frozen val-invariant since D28-final, 6112250)
- detections (primary arm): data/cache/detections/*__yolo11x.npz (D1 cache, untouched)
- embedder checkpoint: data/models/reid_conv.pt
  sha256 7bb4fb3c6f333fdc4fb0131e2c1b44f50c3e20c0127851e30f4c25e2fc3d801d
- detector-arm checkpoint: data/models/det_finetune/y11s_proto/weights/best.pt
  sha256 07b45511407e4058197d60727e4bee919e86f0bcadbe025531724d2656eea003
- segment index: results/occlusion_segments.json (133 val-half segments, D14)

## Runs (val half, single pass each, in this order)

R1 CANONICAL — the G2 module claim. Occlusion-aware tracker, conv embedder:
    run_hidden.py --half val --canonical --occl-buffer 90 --damping 1.0
      --recover-gate 1.5 --overlap-thresh 0.25 --noise-scale 1.0 --lowconf-mode kf
      --app-gate-lost 0.45 --app-gate-recover 0.45 --embedder-tag conv --tag valcanon
    Writes results/tracker_ours.json (G1) + results/hidden_state.json (G2).
    Dev reference: assoc 0.604, e2e 0.345, cov_prematched 0.914, center 0.0064.

R2 NULL INSTRUMENT — ImageNet-R18 embedder, otherwise identical config:
    run_hidden.py --half val --occl-buffer 90 --damping 1.0 --recover-gate 1.5
      --overlap-thresh 0.25 --noise-scale 1.0 --lowconf-mode kf
      --app-gate-lost 0.45 --app-gate-recover 0.45 --embedder-tag imagenet
      --tag valnull --skip-trackeval
    NOT a candidate (D18 intact): measurement instrument for G2a-paired only.

R3 SECONDARY TRACKER (report-only): proto embedder at its dev-best gate
    (b90/d1.0/g1.5/kf1.0/app0.40, embedder proto — dev assoc 0.598). Robustness
    read; NOT selectable, NOT gate-bearing regardless of outcome.

R4 BASELINE REFERENCE (report-only): ByteTrack parameters (occl-buffer 30,
    recover-gate 0.0001, overlap-thresh 1.1, no appearance) for the val taxonomy
    floor context.

R5 PAIRED TEST (G2a-paired criterion):
    paired_test.py --half val --tag-a hidden_valcanon --tag-b hidden_valnull --canonical
    Merges g2a_paired_p + discordant counts into hidden_state.json; per-segment
    outcome vectors land in results/paired_g2a_val_*.json.

R6 DETECTOR-ARM SECONDARY (separate pre-registered claim, honest label):
    cache_detections.py --model yolo11s_ft --weights <detector ckpt above> already
    covers full sequences; embeddings: cache_embeddings.py --model yolo11s_ft
    --weights reid_conv.pt --tag conv (already cached). Run:
    run_hidden.py --half val --occl-buffer 90 --damping 1.0 --recover-gate 1.5
      --overlap-thresh 0.25 --noise-scale 1.0 --lowconf-mode kf
      --app-gate-lost 0.45 --app-gate-recover 0.45 --model yolo11s_ft
      --embedder-tag conv --tag valftdet --skip-trackeval
    LABEL (mandatory in all reporting): "detector-arm prototype — yolo11s finetuned
    10 epochs on DEV-HALF GT only; never saw val frames; val eval clean by
    construction; local prototype of the D26 G2b detector workstream, not the
    headline claim." Claim tested: G2b e2e retention direction + oracle-ceiling
    lift replicating on val (dev showed 0.583 -> 0.845).

## Metrics emitted (every run)

Full aggregate() (D26 standing requirement): n_segments, id_retention,
id_retention_assoc, n_assoc_scope, pre_match_rate, oracle_ceiling,
center_err_coverage, cov_prematched, time_err_coverage,
reemergence_center_err_med, reemergence_time_err_med. Taxonomy guard
(pre_match_rate + oracle_ceiling) is reported ALONGSIDE every gate verdict.

## Pass/fail mapping (gates.yaml G1/G2, evaluated on R1 unless stated)

| criterion | key (hidden_state.json unless noted) | bound | gate-bearing |
|---|---|---|---|
| center err | reemergence_center_err_med | <= 0.015 | YES (frozen D16) |
| conditional coverage | cov_prematched | >= 0.90 | YES (D26) |
| G2a-floor | id_retention_assoc | >= 0.58 | YES (D28-final, regression-floor) |
| G2a-paired | g2a_paired_p (R5) | < 0.05 | YES (D28-final, module claim) |
| G2b end-to-end | id_retention | >= 0.55 | contingent (D26; non-blocking pre detector-at-scale; R6 informs) |
| time err | reemergence_time_err_med | <= 5 | NO (provisional, freeze_order last) |
| segment count | n_segments | >= 100 | YES |
| G1 HOTA/IDF1 | tracker_ours.json vs baseline_ours.json | >= base - 0.5 | YES |
| G1 IDsw | tracker_ours.json | <= base | YES |

## Pre-registered expectations (so outcomes cannot be re-narrated)

- n_assoc on val ~ 76 (133 x ~0.577): G2a-paired is UNDERPOWERED pre-PERUN
  (dev p = 0.26 at n = 97). Non-significance is the expected outcome and is NOT
  grounds for re-tuning; the criterion stays pending until the PERUN-scale embedder.
- G2a-floor 0.58 on val carries ~0.057 binomial sigma; dev best 0.604 gives ~0.4
  sigma margin — a miss within noise triggers report + halt, not adjustment.
- R6 caveat pre-registered: dev-half numbers for the detector arm were optimistic
  (trained on dev); val is the first honest read of that arm.
