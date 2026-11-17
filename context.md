# context.md — occlusion-mot (P1+P2)

## Scope
P1: occlusion-aware multi-object tracker on MOT17/20 — ByteTrack-parity association plus a
hidden-state module (position estimate while occluded, re-emergence point/time, re-ID gating).
P2: CARLA occlusion-scenario feeder with GT visibility, for sim2real ablation of the hidden-state
module. Mission details in mission.md; gates in gates.yaml.

## Architecture
```
data/MOT17/...            (gitignored; motchallenge.net download)
        │
src/omot/
  data/mot.py             MOT sequence loader: frames, GT boxes + visibility, half-split protocol
  io/mot_format.py        MOT-format (frame,id,x,y,w,h,conf,...) read/write
  detect/cache.py         ultralytics yolo11x person detections → data/cache/detections/*.npz
  track/kalman.py         constant-velocity Kalman filter (xywh state)
  track/bytetrack.py      our ByteTrack: two-stage IoU association (high/low score)
  hidden/                 (phase 4) occluded-state motion model, re-emergence prediction, re-ID gate
  eval/trackeval_runner.py TrackEval wrapper → HOTA/IDF1/IDsw JSON in results/
  eval/occlusion.py       (phase 3) occlusion-segment extraction from GT visibility
carla/                    (phase 5, P2) scenario generation + feeder mirroring data/mot.py API
scripts/                  demo_synthetic.py, check_gates.py, hook_postedit.py, download helpers
results/*.json            metric outputs consumed by check_gates.py
runloop.ps1               autonomous outer loop (claude -p per iteration, fresh context)
```
Data flow: video/frames → cached detections (run once) → tracker (baseline | occlusion-aware) →
MOT-format output → TrackEval → results/*.json → gates.

## Hidden-state module design (phase 4, v1 — geometric, no learned parts)
Measured failure modes it targets (D16): 26.3% retention (buffer expiry: gap p90 77f > 30f
buffer; IoU re-match fails after coasting drift), coasting coverage 75%, stay-put-like motion
(median gap displacement 0.02 diag — occluded pedestrians barely move).
Design — `OcclusionAwareTracker(ByteTracker)` via 4 protected hooks added to the base
(no behavior change when unused; verified by byte-identical baseline reruns):
1. `_classify_lost`: at loss time, mark track `occluded=True` if its box overlaps any other
   tracked box (IoU >= occl_overlap_thresh) — inter-object occlusion; else normal loss.
2. `_buffer_for`: occluded tracks live `occl_buffer` frames (default 90) vs 30 baseline.
3. `_predict_pool`: velocity damping `vel_damping^t` for LOST tracks — interpolates between
   pure CV coasting and stay-put, matching the measured displacement distribution.
4. `_recover`: recovery association stage before new-track spawning — unclaimed high dets vs
   occluded lost tracks, cost = center distance / predicted box scale (gate `recover_gate`),
   catches re-emergences where IoU is zero after drift but geometry still identifies the track.
Re-emergence outputs: per-frame coasting boxes of occluded tracks (position while hidden,
G2 center-err channel); re-emergence time prediction deferred to v2 (time-err sub-gate is
last-frozen per user amendment). Tuning: grid on dev-half only (occl_buffer, vel_damping,
recover_gate), objective = G2 retention/center-err/coverage, G1-regression check via
TrackEval on top candidates; val-half touched exactly once for final numbers.

## Decisions
- **D48 (2026-08-10) Sweep enumeration reconciled (21 vs 18), detector wall capped at
  02:30, bundle rebuilds incrementally.** Pre-submission close-out; no arm/pool/seed/metric
  change. Three parts.
  (a) **21 vs 18 embedder units — KEPT AT 21, doc amended, config unchanged.** The gap is
  not 3 extra trainings: `enumerate_units` yields 18 embedder TRAININGS (B/C/D @ full x 3
  seeds = 9; D @ {300,1000,2000} x 3 = 9; D's 3520 point IS the full-pool point, shared)
  plus 3 arm-A ImageNet-null units. perun_sweep_v2.md's grid section says "Arm A adds 1
  cache pass", but its OWN metrics section pre-registers McNemar paired PER SEED for D-vs-A
  and C-vs-A, which consumes one arm-A tracker output per seed — so the grid table
  undercounts arm A as a 1-unit cell when the protocol needs 3. Trimming to 1 would mean
  special-casing the per-seed pairing in `sweep_launcher.aggregate()`; 0.9 h of H200 time
  (arm A is the 0.45 h `embedder_null` class, and ImageNet weights are seed-independent so
  all 3 are identical) does not justify a code fork in the frozen D28 McNemar path on HPC
  day. Recorded as perun_sweep_v2.md **amendment 5** with the full 25-unit enumeration
  table (18 + 3 + 4 detector); config comment rewritten to state it.
  (b) **Detector wall cap 02:30:00 (`sweep_common.UNIT_TIME_CAP_H`), amendment 6.** A cap is
  a budget instrument, distinct from D47's estimate x margin rule, and it is NOT free: the
  detector class's measured spread is 1.5-3.0 h/run, so 2.5 h sits deliberately INSIDE that
  spread and a unit above it is killed by `timeout`, leaving no result JSON (retried by
  re-submitting the array — the existing resume rule). Bought: the grid's worst case drops
  from 41.25 h (over the 40 h ceiling) to 39.25 h (under it), and the array-wide `--time`
  from 04:05 to 02:30. The ceiling did not move; the failure mode is pre-registered, and
  R6 selection over a short detector set is reported, never silent. `budget_table` now bills
  `billed_high_h = min(est_high, wall)` — a unit cannot cost more than its own kill limit.
  Test invariant updated accordingly: uncapped classes must have limit >= est_high; a capped
  class must have limit == cap > est_low, and a new test asserts `fits_high`.
  (c) **`make_hpc_bundle.py build --reuse`.** A code/doc-only change moves repo.tar and
  nothing else, but the old build rmtree'd the stage and re-tarred 8 GB of payload. `--reuse`
  keeps a staged tar only when all four hold: not the git archive; spec sources/arcnames
  match the prior manifest (new field, so pre-D48 manifests are never reusable); the tar
  still hashes to its manifest entry; no source file is newer than the tar. Fails closed —
  any doubt rebuilds. Orphan staged files not in the current plan are dropped with a warning
  so a removed spec cannot ride along invisibly.
  (d) **Repo-visibility contradiction resolved [user, 2026-08-10].** CLAUDE.md's Autonomy
  rule said "any GitHub remote must stay private" while `matus012/occlusion_mot` has been
  PUBLIC by user approval since D12/D17 — so the D48 push (13 commits: D45 val event, D46
  adjudication, D47 readiness, D48) halted on a rule the state already contradicted. User
  ruled: public is intentional, the logged approval supersedes the privacy line. CLAUDE.md
  rewritten to say so explicitly, so a routine `push origin main` no longer stalls; changing
  visibility in EITHER direction and force-push still require approval.
- **D47 (2026-08-09) PERUN-readiness: HPC day reduced to "fill 2 values, transfer, sbatch".**
  Infrastructure only; no training logic touched. Six parts.
  (a) **Transfer bundle** `scripts/make_hpc_bundle.py` (build/verify/unpack). The input set
  is DERIVED from the sweep config (`required_inputs(cfg)`), never hardcoded, and a pytest
  guard asserts the bundle plan covers it — so adding an arm/source/detector model fails on
  the dev box, not on an offline compute node. Hashing rule: loose files hashed individually;
  large trees ride in per-tree tars that are hashed individually. A tar's sha256 covers every
  member byte, so integrity is strictly stronger than a per-file listing while the manifest
  stays a few KB instead of ~50 MB (data/reid alone is 509,122 files). The nesting also lets
  the operator place those inodes on scratch — cluster home dirs commonly cap at 100-500k.
  (b) **Offline env.** `requirements-lock.txt` resolved for x86_64-manylinux_2_28 + cu126.
  Two deliberate deviations from the dev-box requirements.txt, both HPC-motivated:
  opencv-python -> opencv-python-headless (compute nodes lack libGL.so.1) and trackeval
  dropped (git dep; the sweep runs `--skip-trackeval` so it is never imported). polars
  dropped — nothing imports it and polars-runtime-32 is yanked upstream.
  (c) **Runtime assets pre-fetched.** `resnet18(weights=IMAGENET1K_V1)` is the trunk
  initialiser for EVERY train_reid.py run and the entirety of arm A; ultralytics
  `check_amp()` loads yolo26n.pt before every detector train; yolo11m.pt was never on disk.
  All staged into `assets/` + repo root, pointed at by `scripts/hpc_env.sh`
  (TORCH_HOME / YOLO_CONFIG_DIR / MPLCONFIGDIR / HF_HUB_OFFLINE).
  (d) **Budget enforcement made executable.** SLURM's `--time` is uniform across an array,
  so a per-unit ceiling cannot come from it: wall limits are per unit CLASS and enforced with
  `timeout` inside each task. The grid total is asserted at generation time against
  `budget_ceiling_h` — hard fail if the LOW estimate busts it, loud warning if the HIGH one
  does. Measured outcome for perun_full: 25 units, **35.25-41.25 H200-h vs the 40 h ceiling**
  — the low end fits, the high end does not, driven entirely by the detector arm's 1.5-3 h/run
  spread. Reported, not tuned away (thresholds never move to make a gate pass); the operator's
  documented lever is dropping to one detector model after reading the smoke job's throughput.
  (e0) **THE find: `src/omot/data/` was never committed.** `.gitignore`'s `data/` pattern is
  unanchored, so it matched `src/omot/data/` as well as the root data dir — the package
  holding `MOTSequence` / `load_split` / `half_split_frames` has been untracked since the
  repo was created. Every local run worked because the files exist in the worktree; the
  checked-in tree (and therefore the public repo, and `git archive HEAD`) could not
  `import omot.data` at all. Surfaced by the deliverable-5 requirement to run the entrypoint
  from the UNPACKED BUNDLE rather than the dev tree: `ModuleNotFoundError: No module named
  'omot.data'`. Nothing else was lost (audited every ignored path). Fixed by anchoring
  `/data/ /logs/ /runs/ /weights/` and committing the package. Two tests now guard the class:
  every `src/**/*.py` on disk must be git-tracked, and no unanchored .gitignore directory
  pattern may name a directory that exists under `src/`. No measurement is affected — all
  results were produced from the worktree, which always had the file.
  (e) **Two more defects found and fixed, both fatal on HPC day.** (1) Emitted `submit.sbatch`
  had CRLF endings — `Path.write_text` translates `\n` to `\r\n` on Windows, so the script
  would reach the cluster as `#!/bin/bash\r` and die with "bad interpreter". Fixed with
  `write_text_lf` + `.gitattributes eol=lf`; a test now asserts LF on every emitted script.
  (2) The two detector units sharing a data mix would RACE on building
  `data/cache/det_finetune_<mix>/` (harmless in sequential local mode, corrupting under an
  array). Fixed by prepping both mixes in the smoke job, which the runbook chains ahead of
  the array with `--dependency=afterok`.
  (f) **Grid size correction.** perun_sweep_v2.md says 18 embedder runs; the config
  enumerates 21 (A/B/C at full pool x3 = 9, plus D's 4-point scaling curve x3 = 12) because
  arm A is run once per seed for schema uniformity. 21 + 4 detector = 25 units. The doc's
  "~13 cache passes" line is likewise superseded — every embedder unit carries its own cache
  pass. Budget above uses the enumerated 25, not the doc's 18.
  Cannot be verified without cluster access, and stated as such in the report: PERUN's real
  partition/account names, node CPU/RAM shape (cpus_per_task 8 / mem 64G are conservative
  guesses), glibc >= 2.28, availability of a Python 3.11 module, actual Linux `pip install`
  of the wheelhouse, and every wall-clock estimate on H200 (all extrapolated from RTX 4060).
- **D1 (2026-07-22) Fixed-detections design.** Detector inference runs once, cached to
  `data/cache/detections/`. All tracker comparisons use identical cached detections —
  detector-independent, cheap iteration on 8GB VRAM. Invariant in CLAUDE.md.
- **D2 (2026-07-22) Detector = ultralytics yolo11x (COCO, person cls), not YOLOX.** Avoids YOLOX
  Windows build pain. Published ByteTrack used MOT-finetuned YOLOX-x, so absolute numbers will
  differ → the reproduction gate (G0) is implementation-equivalence vs a reference ByteTrack on
  identical detections. Optional later: MOT-finetune a detector on PERUN.
- **D3 (2026-07-22) Own ByteTrack implementation** in src/omot/track (Kalman + two-stage IoU,
  no learned parts) because the hidden-state module extends tracker internals. Cross-checked vs
  reference impl (supervision) for G0. NOTE: supervision deprecates ByteTrack at v0.30 —
  the exact pin supervision==0.29.1 in requirements.txt is what keeps the G0 reference
  reproducible; do not bump it.
- **D4 (2026-07-22) Eval protocol** = MOT17 train half-split (first half dev / second half val,
  ByteTrack ablation protocol). HOTA/IDF1/IDsw via TrackEval; motmetrics cross-check.
- **D5 (2026-07-22) numpy pinned 1.26.4** — TrackEval unmaintained, breaks on numpy 2.x.
- **D6 (2026-07-22) torch from cu126 index**, latest resolved by uv; CUDA availability verified
  post-install (see status.txt).
- **D7 (2026-07-22) Occlusion segment definition**: GT track visibility < 0.25 for ≥ 5 consecutive
  frames, reappearing with visibility ≥ 0.5. Encoded in gates.yaml G2.
- **D8 (2026-07-22) Multi-GPU-ready structure**: device always injected, no `cuda:0` literals,
  config-driven batch sizes; SLURM launcher added later without refactor (PERUN H200s).
- **D9 (2026-07-22) Licensing: repo is AGPL-3.0.** ultralytics is AGPL-3.0 and is imported by the
  detection-caching path; isolating it buys nothing since we open-source anyway. Decision: license
  the whole repo AGPL-3.0-only (LICENSE file + pyproject). [User amendment 2026-07-22]
- **D10 (2026-07-22) TrackEval pinned** to commit `12c8791b303e0a0b50f753af204249e622d0281a`
  (HEAD at scaffold time) via git URL in requirements.txt. [User amendment 2026-07-22]
- **D11 (2026-07-22) G2 freeze order**: reemergence position error + id_retention freeze first;
  reemergence_time_err_med is the LAST sub-gate to freeze and never hard-fails before the
  hidden-state module exists. [User amendment 2026-07-22]
- **D12 (2026-07-22, updated) GitHub remote.** Original rule: remotes stay private without
  explicit approval. 2026-07-22: user created public repo matus012/occulsion_mot and gave
  one-line approval to push publicly (AGPL-3.0 already in place). Remote `origin` set; repo
  is PUBLIC by explicit user approval.
- **D13 (2026-07-22) MOT17 mirror fallback.** motchallenge.net down all session. User directive:
  free reputable mirror allowed after the official-site poller window; verify structure/hashes
  vs official spec; log source + hashes here; no paid/credential-gated mirrors. Chosen mirror:
  HF `ling1016/MOT17` (ungated, full 21-dir train layout verified via API; fallback
  `Morrison1025/MOT17`). Official site publishes no checksums -> verification is structural
  (scripts/verify_mot17.py: official seqLength/resolution table, frame counts, GT ranges,
  cross-variant gt.txt equality) + SHA256 manifest recorded in results/mot17_verification.json.
  Baseline stays UNVERIFIED until G0 freezes numerically. Provenance (2026-07-22):
  source=hf:ling1016/MOT17, revision=024c7873a46e0944ba711726bdfd9fc58bc8c2f1,
  manifest_sha256=49c218a399e616b9abe40d5d7a24556075483e41b1d90c9dd444580a6a9c2c27.
  Structural verification PASSED (0 failures, hf-dedup layout). Independent cross-mirror GT
  check vs Lekim89/MOT17: ALL 7 sequences match=1.0 with clean id bijection
  (results/mot17_gt_crosscheck.json). Re-hash vs official zip when motchallenge.net returns.
- **D15 (2026-07-22) G0 frozen — equivalence at 0.6, operating baseline at 0.25.**
  Equivalence vs supervision 0.29.1 required IDENTICAL hyperparameters (supervision defaults
  to activation 0.25; official ByteTrack 0.6 — first comparison was apples-to-oranges).
  Matched at 0.6: dHOTA 0.52, dIDF1 0.07 -> PASS. At 0.25 the ports diverge slightly
  (dIDF1 1.24; implementation tie-breaking details) — not chased further: G1/G2 compare our
  occlusion-aware tracker against OUR OWN baseline (same implementation, same detections),
  so cross-port residuals cannot contaminate the science. Operating baseline = ours at
  activation 0.25 (COCO score distribution: 0.6 drops real pedestrians): HOTA 49.98,
  IDF1 58.28, MOTA 45.23, IDsw 359. fuse_score measured WORSE with COCO scores
  (IDF1 -0.3, IDsw +64) -> TrackerConfig.fuse_score default False. All runs deterministic
  across repeats.
- **D19 (2026-07-22) Val run DEFERRED [user approval] — phase 5 proceeds.** Dev evidence:
  geometric mechanisms ceiling out at +1.8pt retention (0.292 -> 0.310 over 55 dev runs);
  42% of segment mass is detector-ceiling (no det at re-emergence / never tracked pre-gap).
  Path: CARLA feeder (P2) + appearance re-ID gate; val-half stays untouched until the module
  credibly targets all frozen G2 subs. STANDING NOTE [user]: if appearance re-ID measures out
  below the 0.55 retention target, PROPOSE a target amendment WITH the ceiling analysis
  attached — never a silent re-freeze.
- **D20 (2026-07-22) Appearance re-ID v0 = cached embeddings, no new heavy deps.** Embeddings
  are precomputed per cached detection (one GPU pass, stored alongside the detection cache)
  so the tracker stays image-free at association time — same invariant as fixed detections
  (D1). v0 embedder: torchvision ResNet18 (ImageNet) penultimate features — already
  installed, deterministic; upgrade path: CARLA/PERUN-trained embedding (phase 6 sim2real,
  the thesis ablation). Gate design: cosine-distance veto/rescue in lost-track matching and
  recovery.
- **D21 (2026-07-22) Appearance gate v0 measured (grid 3, dev).** Best: cos-dist gate 0.45
  on both lost/recover vetoes + recovery gate 1.5 -> retention 0.327 (geometric-only 0.310,
  baseline 0.292); center-err 0.0071 (passes frozen 0.015), coverage 0.702. Tighter gates
  (0.25/0.35) REGRESS — ImageNet ResNet18 features veto correct re-matches under partial
  occlusion. Conclusion: mechanism validated, embedder is the bottleneck; phase 6 trained
  re-ID embedding (CARLA + PERUN) is the designed remedy. D19 amendment trigger NOT fired
  yet — "appearance re-ID measures out" means after the trained embedder, not v0.
- **D22 (2026-07-22) CARLA client isolation.** The 0.9.15 Windows zip ships ONLY a cp37
  client wheel; PyPI carries carla==0.9.15 win wheels up to cp310. Sim driving therefore
  runs in a dedicated py3.10 venv behind a subprocess boundary (feeder CLI); the main
  py3.11 env never imports carla. Sim binary: tools/CARLA_0.9.15/WindowsNoEditor.
  Module named omot.sim (not carla) to avoid namespace collision. MockBackend is the
  projection/visibility reference implementation the CarlaBackend must agree with; mock
  runs write results/carla_feeder_mock.json — G4 stays PENDING until real-sim renders.
  SERVER BRING-UP (2026-07-23): the CARLA zip lacks UE4PrereqSetup; on this machine
  XINPUT1_3.dll + X3DAudio1_7.dll (DirectX Jun2010 redist) were missing -> exe dies with
  STATUS_DLL_NOT_FOUND. Launched from the agent harness tree this shows as a SILENT
  loader freeze (6MB RAM, 0 CPU, threads in LpcReply = hidden hard-error dialog); the
  real exit code only surfaced via a Task Scheduler launch. Fix WITHOUT admin: extract
  the two x64 DLLs from Microsoft's directx_Jun2010_redist.exe cabs next to
  CarlaUE4-Win64-Shipping.exe. Boot verified: Town10HD_Opt, RPC on :2000, ~7GB VRAM
  during shader compile. uv-managed py3.10 needed --python <explicit exe path> for venv
  creation (uv 0.11.30 minor-version-link bug on Windows).
- **D23 (2026-07-23) G4 frozen — sim feeder validated; what the validation may claim.**
  24 CARLA scenarios rendered (Town10HD, deterministic teleport paths), 223 occlusion
  segments [ANNOTATION D66, 2026-08-19: the committed `results/carla_feeder.json` records
  `n_occlusion_segments: 225`. The 223 above is left as written -- dated entries are not
  rewritten -- but 225 is the number that traces to the artifact and is what
  report/omot_report.md and status.txt now use.], exact per-walker visibility via
  isolation-calibrated instance ids.
  Validation lessons burned in: (a) cross-backend visibility correlation is structurally
  unsound where occluder PROPS differ from spec slabs (a mock walker lingering in a
  spec-box shadow reads "always hidden" while the sim correctly sees it — walker-6 case);
  (b) the gate now tests inter-walker occlusion only (identical geometry in both
  backends), outside static shadows, with a per-scenario hard floor 0.35 (sign/axis bugs
  read ~0) and cross-scenario median >= 0.5 (slab noise floor ~0.45 in densest crowds);
  (c) absolute sim-GT sanity bounds apply everywhere (p75 vis >= 0.85, deep-occl <= 0.45).
  Center-err median 0.09-0.16 boxnorm across all 24. Buggy render sets preserved as
  carla_render_v1_buggy / _v2_buggy pending user approval to delete (~35GB).
  Re-ID dataset extracted from v3: 161 identities, 59,563 crops (occlusion-tagged),
  identity-disjoint 131/30 split -> data/sim/reid (phase 6 training input, PERUN).
- **D24 (2026-07-23) Re-ID identity audit [user-prompted] + two-source training design.**
  Audit: 161 sim identity labels collapsed onto 10 unique blueprints (walker_id % n_bps
  reused the same models every scenario) — contrastive supervision would be poisoned
  (identical-appearance pairs as negatives). Mitigation: (1) sim identities relabeled to
  BLUEPRINT level (appearance-correct; 45-class ceiling = CARLA's without texture mods);
  (2) driver diversifies blueprints per scenario (seed*31 + wid*7 mod n_bps) + emits
  walkers_meta.json; v4 re-render; v3 kept as carla_render_10bp (tracking GT still valid);
  (3) MOT17 DEV-HALF tracklet identities added as a real-domain source (D18-compliant):
  335 identities / 46,859 crops with GT visibility tags. Sim2real ablation now 3-armed:
  ImageNet vs sim-only vs sim+dev-real.
- **D25 (2026-07-23) Trained-embedder dev results + G2 ceiling analysis (amendment
  proposal per D19 standing note — decision is the user's).**
  Grid 4 (proto embedder, 5 local epochs, sim+dev-real): best retention 0.345 @ app-gate
  0.40 (baseline 0.292 -> geometric 0.310 -> ImageNet 0.327 -> proto 0.345); center-err
  0.0055 (best measured; frozen threshold 0.015 passes with 2.7x margin).
  CEILING ANALYSIS: oracle-association retention ceiling on dev = 0.584 (taxonomy: 22%
  never tracked pre-gap + 20% no detection at re-emergence are DETECTOR-capped, untouchable
  by any association/re-ID improvement). Frozen retention target 0.55 = 94% of oracle.
  Frozen coverage target 0.90 is ARITHMETICALLY UNATTAINABLE: coverage counts all segments
  but is hard-capped by pre-match rate (0.768 dev); conditional coverage (pre-matched
  segments only) is already 0.914.
  PROPOSED AMENDMENT (user decides): (a) retention >= 0.45 for the association-only
  scope (77% of oracle; current best 0.345, full PERUN training pending), or keep 0.55
  contingent on adding a detector-upgrade workstream (MOT-finetuned detector on PERUN
  raises the ceiling itself; anticipated in D2); (b) redefine coverage as conditional on
  pre-matched segments, threshold 0.90 (definitional fix of an impossible criterion, not
  a weakening — unconditional 0.90 > pre-match ceiling 0.768).
- **D26 (2026-07-23) G2 amendment DECIDED [user].** (1) Coverage: conditional on
  pre-matched segments, >= 0.90 — logged explicitly as a definitional repair of an
  impossible criterion (D16 error acknowledged: unconditional coverage was capped by
  pre_match_rate); functions as a regression floor on val. (2) Retention SPLIT:
  G2a association-scope >= 0.45 (amended; measures the module) / G2b end-to-end >= 0.55
  RETAINED, contingent on the detector-upgrade workstream (option b: MOT/CARLA-finetuned
  detector on PERUN, raising the oracle ceiling itself). (3) STANDING REQUIREMENT: at
  val time, compute and report the same failure taxonomy (pre_match_rate, oracle ceiling)
  on val alongside metrics so amendment premises are verifiable — aggregate() now emits
  these automatically. (4) Queue unchanged: PERUN embedder training + 3-arm ablation ->
  detector workstream -> D18 val-run request.
- **D27 (2026-07-23) Response protocol [user directive, binding]:** VISUAL-FIRST (every
  viewable milestone ships annotated video / crop grids / plots + artifact paths + open
  command; metric deltas as tables <= 8 rows) and a mandatory CTA FOOTER (REVIEW / PICK /
  WAIT+ETA / STUCK / DOWNLOADING+ETA / DONE) as the final line of every response.
  Encoded in CLAUDE.md. Concurrent: LOCAL-MAX phase L1-L6 (G2a audit; viz pipeline;
  local embedder best-effort; 3-arm ablation dry-run; detector-finetune prototype;
  scaling-study runner) — dev-half only, commit per item.
- **D28-PROPOSAL (2026-07-23) G2a recalibration [audit-triggered, PENDING user].**
  L1 audit under identical denominators (dev, n_assoc ~97): baseline assoc-retention
  0.505, geometric 0.531, ImageNet-gated 0.556/0.552, proto-trained 0.598. G2a >= 0.45
  is non-discriminative (baseline clears by 5.5pt; calibration error: threshold derived
  on end-to-end scale, assoc denominator inflates all configs ~1.7x). PROPOSAL: G2a
  >= 0.58 (~1.5 sigma above baseline at n=97, above off-the-shelf-appearance nulls,
  met by current proto 0.598). Single change; no other gate touched. Audit artifacts:
  results/hidden_dev_audit_{base,geom,in45,in40}.json.
- **D28-FINAL (2026-07-23) G2a two-part recalibration DECIDED [user, option 3].**
  (1) G2a-floor: id_retention_assoc >= 0.58 at val — regression-floor semantics ONLY
  (like cov_prematched), not the module claim. (2) G2a-paired, THE discriminating
  criterion: trained embedder beats the ImageNet-R18 null on the IDENTICAL segment set
  (intersection of both configs' association scopes — fixed denominator, kills the
  per-config n drift from audit caveat a); per-segment paired outcomes -> McNemar exact
  one-sided, p < 0.05; result JSON carries outcome vectors + discordant counts + p
  alongside the taxonomy guard. (3) FROZEN val-invariant from this commit: segment
  inclusion rule (hidden_eval IoU>=0.5 pre/post), intersection-denominator definition,
  paired-test code path (src/omot/eval/paired.py + scripts/paired_test.py; --canonical
  merges into hidden_state.json). Note: the val paired test requires running the null
  config at val as a measurement instrument — this is part of the single canonical val
  event, not a second candidate (D18 intact). (4) Dev paired run (immediate, report-not-
  tune): proto vs in45 null — n=97, retention 0.598 vs 0.557, discordant 14/10,
  p=0.271; vs in40 — n=95, 0.611 vs 0.558, 16/11, p=0.221. NOT significant pre-L3, as
  expected; L3 convergence training is the margin-builder. (5) gates.yaml updated
  (floor 0.58 + g2a_paired_p <= 0.05); standalone 0.45 formulation dropped; G2b
  untouched. (6) Deletions approved & executed: stray 0-byte files None/int removed;
  the D23 buggy render sets (~35GB) were ALREADY ABSENT from disk (data/sim holds only
  carla_render v4 4.5GB + carla_render_10bp 4.5GB + scenarios 0.3GB) — nothing
  adjacent deleted per D-INCIDENT rule; carla_render_10bp kept per D24.
- **D29 (2026-07-23) L3 result: convergence training saturates — hypothesis REFUTED.**
  reid_conv (40 epochs, sim+dev-real, occ-rank1 0.888 -> 0.957 at epoch 33) cached as
  embedder tag `conv`; dev grid app 0.30-0.50 at b90/d1.0/g1.5/kf1.0: best assoc 0.604
  @ app0.45 (n=96, e2e 0.345, center 0.0064). vs proto: +0.6pt assoc, IDENTICAL
  retained count (58/58), discordant 6/6 (p=0.61) — the two trained embedders agree on
  84/96 segments. vs ImageNet null (paired, D28 criterion): 0.604 vs 0.562, discordant
  13/9, p=0.262 — still NOT significant on dev. The session-handoff hypothesis
  (convergence adds +2-4pt) is refuted: retrieval gains are flattered by near-duplicate
  galleries as suspected; tracker-level appearance signal saturates ~0.60 assoc with
  yolo11x detections. Residual failures are appearance-hard (blur/crowd-handoff per
  crops_assoc_scope.png). IMPLICATION for G2a-paired at val (n~76): local-training
  effect size (~4pt vs null, ~22 discordant) is underpowered; the margin must come from
  PERUN-scale training (305 train ids is the binding constraint, not epochs) and/or the
  detector workstream (raises n and scope). Dev config selection unchanged rules-wise;
  current top-3 dev configs for any future D18 val request: conv_app45 (0.604),
  d26best/proto_app40 (0.598), in45 (0.556).
- **D30 (2026-07-23) L4: 3-arm ablation dry-run complete.** sim-only arm trained
  (reid_simonly, 40 ep, occ-rank1 0.933 on sim-val — not cross-arm comparable at
  retrieval level), cached as embedder tag `simonly`; dev sweep best 0.577 @ app0.50
  (n=97, e2e 0.333). Tracker-level ordering: ImageNet 0.556 < sim-only 0.577 <
  sim+real 0.604 — sim2real transfer positive, real data additive on top. Paired
  (fixed denominators): sim-only vs ImageNet 6/4 discordant p=0.377; sim+real vs
  sim-only 10/8 p=0.407 — all n.s. at n~97. Dry-run verdict: pipeline validated
  end-to-end, effect DIRECTION consistent across arms, local power insufficient
  (~2x-3x more discordant pairs needed for p<0.05 at these effect sizes) — the real
  3-arm ablation is the PERUN run (mission phase 6). Artifacts:
  results/hidden_dev_simonly_app{30..50}.json, paired_g2a_dev_* (2 new),
  viz/ablation_3arm_dev.png.
- **D31 (2026-07-23) L5: detector-finetune prototype validates the G2b workstream.**
  yolo11s finetuned on MOT17 DEV-HALF GT only (10 ep, 960px, 2391 train / 266 monitor
  frames; mAP50 0.940, recall 0.863); detections cached under NEW tag yolo11s_ft
  (yolo11x cache untouched, D1 intact). Dev-half effect (config-matched geometric
  baseline = hidden_dev_audit_geom.json): pre_match 0.780 -> 0.940, oracle ceiling
  0.583 -> 0.845, e2e retention 0.310 -> 0.458, center-err 0.0040. Full stack (ft dets + conv embedder + app0.45): e2e 0.494
  (yolo11x stack 0.345), assoc 0.589 @ n=141, center-err 0.0033. MANDATORY CAVEAT:
  the detector TRAINED on dev-half frames, so all dev numbers in this entry are
  optimistic (train-on-train for the detector); they demonstrate mechanism + headroom,
  not expected val performance. The val protocol is clean by construction (detector
  never saw val frames) — G2b at val runs on this arm per D26. Assoc-scope retention
  moved little (0.531 -> 0.542 geometric), confirming G2a/G2b decomposition: the
  detector moves the CEILING, the embedder moves retention WITHIN scope. Artifacts:
  results/hidden_dev_ftdet_{geom,conv_app45}.json, viz/detector_arm_dev.png,
  checkpoint data/models/det_finetune/y11s_proto (gitignored).
- **D32 (2026-07-23) L6: identity-scaling runner built + smoke-validated; LOCAL-MAX
  queue complete.** scaling_study.py (nested identity subsets via seeded-shuffle
  prefix; crash-safe incremental resume; fixed yolo11x detections, dev only). Smoke
  (3 epochs, fractions 0.25/1.0): 76 ids -> assoc 0.526 / occ-rank1 0.852; 305 ids ->
  assoc 0.567 / 0.891 (identical denominator n=97). Identity count moves tracker-level
  assoc even at 3 epochs (+4.1pt for 4x ids) — supports D29's claim that identities,
  not epochs, are the binding constraint; full 4-fraction 40-epoch curve is cheap on
  PERUN and sizes the required identity pool. L1-L6 all DONE. Remaining queue (D26):
  PERUN embedder training + 3-arm ablation, detector workstream at scale, D18 val
  request (canonical val + null-config run + taxonomy report + G2a-paired), phase 7
  demo. Artifacts: results/scaling_study.json, viz/scaling_study.png.
- **D33 (2026-07-23) Strategy correction [user directive].** "Local done" was wrong
  under L3+L6: epochs are dead as a lever (L3), identities bind (L6), rendering is
  local. DATA WORK IS THE LOCAL FRONTIER. PERUN case restated: fast sweeps over
  ENLARGED identity pools + detector finetune at scale — NOT longer training on
  current data.
- **D34 (2026-07-23) Identity-ceiling audit (server-verified).** (a) CARLA 0.9.15
  install has 51 walker blueprints (walker.pedestrian.0001-0051; live
  blueprint-library dump); 6 child models (0009-0014) unusable under teleport control
  (D22) -> 45 usable adults; v4 render set already uses ALL 45 (walkers_meta union) —
  the sim blueprint-identity ceiling is EXHAUSTED. (b) Cheap sim diversity levers:
  union of modifiable attributes across all 51 walker blueprints = {role_name,
  ros_name, is_invincible, speed} — ZERO appearance attributes; CARLA's texture API
  targets named static map meshes, not skeletal walker actors; weather/lighting vary
  illumination, not identity -> intra-identity augmentation only, ~0 identity-
  equivalent gain (and L3 showed the training-side lever is saturated). Genuine sim
  identity scaling needs UE4-editor clothing/material variants — exceeds the 1-day
  cap, REJECTED as a local lever. (c) Real-data option sheet delivered (response
  2026-07-23): recommendation = MOT20-train tracklets (~2.2k ids, visibility-tagged,
  pipeline reuse, zero MOT17-val contamination) + Market-1501 (1.5k ids, trivially
  disjoint); MSMT17 optional after the L6 curve sizes the need. NO integration done —
  user decides.
- **D35 (2026-07-23) Val manifest FROZEN (val_manifest.md, committed val-invariant).**
  Pre-registers the single D18 val event: R1 canonical (conv app0.45, gate-bearing),
  R2 ImageNet null instrument, R3 proto secondary (report-only), R4 baseline
  reference, R5 paired test (--canonical merge), R6 detector-arm secondary with
  mandatory honest label (trained on dev half; val = first honest read). Checkpoint
  SHA256s + commit pinned in the manifest; pass/fail table maps every gate criterion
  to its JSON key; pre-registered expectations (G2a-paired underpowered pre-PERUN;
  floor margin ~0.4 sigma) prevent outcome re-narration. Val stays blocked on user
  approval; misses -> report + HALT (D18.3).
- **D36 (2026-07-23) L6 local curve (3 fractions x 2 seeds + anchor, 25 ep) — D32's
  smoke conclusion DOWNGRADED.** Assoc by (n_ids, seed): 76 -> 0.536/0.594;
  152 -> 0.573/0.542; 229 -> 0.588/0.577; 305 -> 0.579 (seed 0). Seed-to-seed spread
  (up to 5.8pt at fixed n_ids) is the same order as the whole 76->305 identity
  effect: tracker-level G2a CANNOT resolve the identity slope at n~97 dev segments
  over the current pool range — the smoke's clean +4.1pt (D32) was seed-0 luck.
  What DOES resolve: occ_rank1 is monotone in identities for BOTH seeds
  (76: 0.874-0.888 -> 152: 0.901-0.923 -> 229: 0.915-0.926 -> 305: 0.928), still
  rising at 305 (decelerating). REVISED claim: identities scale re-ID quality
  (retrieval-level, robust); the tracker-level payoff needs either order-of-magnitude
  pool growth (2c data adds) or seed/segment aggregation to measure. PERUN sweep
  design consequence: sweep pool sizes on a LOG scale incl. 2c-enlarged pools
  (~300 -> ~2k -> ~4k+), score retrieval + multi-seed tracker means (>= 3 seeds),
  and treat single-seed tracker deltas < ~6pt as noise. Artifacts:
  results/scaling_curve_local.json, viz/scaling_curve_local.png (P2-writeup figure).
- **D37 (2026-07-23) 2c verdict + integration [user]: MOT20-train tracklets first,
  Market-1501 second.** (1) Order binding: MOT20's visibility tags extend the
  occluded-query protocol — validate that pipeline before Market's plain crops.
  (2) LICENSE HYGIENE (binding): no dataset-derived content (crops/images/frame
  dumps/serialized embeddings) in repo or ANY remote — loaders, manifests, SHA256s,
  download docs only. Enforced: .gitignore block + tests/test_license_guard.py
  (data/ must have zero tracked files; no raw-image extensions tracked anywhere; no
  weight/embedding blobs outside tests/fixtures; PNGs = plots under viz/ only).
  RETROACTIVE FIX: viz/crops_assoc_scope.png (MOT17 GT crops) was tracked+pushed —
  untracked at this commit; NOTE it persists in git history; full purge = history
  rewrite + force-push, PENDING user approval. (3) Ablation redesigned in mission.md
  BEFORE any training on new data: arms A ImageNet / B sim-only / C real-only
  (MOT17-dev + MOT20 + Market) / D sim+real; P2 claim = D > C; >= 3 seeds per arm
  (D36). (4) Post-integration verification owed: pool-size report, occluded-query
  eval extended to MOT20, one SMALL single-seed retrieval sanity run (labeled; no
  tracker-level claims). (5) val_manifest.md untouched (training pools only).
  (6) Final artifact before PERUN submission approval: sweep design doc v2
  (log-scale pools ~300/~1k/~2k/~4k, >= 3 seeds, arms A-D, GPUh per arm).
- **D38 (2026-07-23) Split-policy correction [user], applied BEFORE extraction.**
  (1) MOT20 identity budget: per-sequence stratified, seeded, identity-disjoint
  80/20 train/eval (was 85/15); manifest-pinned via split_manifest_sha256 in
  index.json (sha256 over sorted identity:split lines — drift-detectable). Train
  share joins arm C/D pools; eval share extends the OCCLUDED-QUERY protocol only —
  NOT tracker val (MOT17 val-half remains the only tracker val). (2) Disjointness
  guard: tests/test_disjointness_guard.py — zero namespaced-identity overlap between
  any training pool and the occluded-query eval set across all four sources;
  market1501 must contribute ZERO eval ids (no vis GT); mot20 manifest re-hash check;
  runs in G3 like the license guard (skips gracefully on data-less clones).
  (3) Visibility bounds, explicit + logged (train_reid.py constants): queries vis in
  [0.10, 0.50), gallery vis >= 0.60. Derived from the MOT20 GT vis distribution
  (MOT20-01/02, 174.6k ped rows): vis=0 5-9%, (0,0.1) 12-14% — pixel-less slivers,
  unanswerable as queries -> Q_LO=0.10; Q_HI=0.50 matches the D14 occlusion boundary;
  [0.5,0.6) is 7-8% boundary-ambiguous mass -> G_LO=0.60 gives a clean visible-gallery
  margin. Extraction min-vis: 0.0 -> 0.10 (the coder's min-vis=0 intent — feeding
  low-vis crops to the occluded-query protocol — survives: the [0.1,0.5) band is fully
  retained; only sliver crops are dropped). NOTE: occ_rank1 numbers under the new
  bounds are NOT comparable to pre-D38 values (protocol refinement).
- **D39 (2026-07-23) Demo package [user directive], built + cold-read reviewed.**
  demo/ guided tour S0-S8 (README.md) for technical colleagues: problem teaser ->
  baseline failure -> geometric coasting -> ImageNet veto -> trained embedder (incl.
  paired-test honesty: p=0.26 n.s.) -> CARLA data engine -> detector arm (train-on-dev
  caveat burned into the clip itself) -> scaling story -> gate scoreboard. Renderer:
  scripts/render_demo.py (segment auto-pick by legibility x outcome pattern; 0.5x
  slow-mo through the occlusion window; legend strip; hero.mp4 = S1->S4 same segment
  MOT17-09:t10:f174-207 + CARLA tail). TWO RULE CHANGES, precedent-setting:
  (a) license-guard PNG allowlist extended to demo/ (CARLA-derived teaser + blueprint
  grid are committable; crops_* still banned everywhere); (b) .gitignore negation
  !demo/s5_carla.mp4 under the global *.mp4 block — CARLA-only renders are synthetic
  content, committable per D37; ALL MOT17-derived clips live in gitignored
  demo/clips/ with a regeneration one-liner in the README. Verified: every README
  path exists, numbers cross-checked to results/*.json by the cold-read reviewer;
  S8 scoreboard pinned to the conv app0.45 config. PC crash mid-review-fix
  2026-07-23: working tree survived intact; downloads resumed from HF cache.
- **D40 (2026-07-24) D37/D38 integration COMPLETE + PERUN sweep doc v2 delivered.**
  (1) MOT20 verified vs official spec (429/2782/2405/3315 frames, resolutions, GT
  ranges) after one verifier fix: MOT20 legitimately adds GT class 13 "crowd"
  (2,489 rows in MOT20-03, ALL conf-flag 0 = non-evaluation) — ALLOWED_CLASSES
  widened to 1-13. Provenance: hf:Lekim89/MOT20 @ 5fcaa0ef (same author as the D13
  MOT17 cross-check mirror), manifest sha256 530c67c3d0cc...3057
  (results/mot20_verification.json). Market: hf:aveocr @ b5e654a4, canonical file
  counts PASSED; 1,500 usable ids after junk exclusion (0000/-1).
  (2) POOL REPORT: train 3,520 ids / 509k crops (mot17_dev 269, sim 36, mot20 1,715,
  market1501 1,500) — 11.5x pre-integration; occluded-query eval 506 vis-tagged ids
  (66/9/431), 422 answerable under D38 bounds (was 58). Disjointness + license guards
  8/8 green incl. mot20 manifest re-hash. (3) SANITY RUN (labeled: single seed, 3 ep,
  NO tracker-level claims): combined pool trains without collapse — loss 8.46->7.81,
  occ-rank1 0.239->0.290->0.329 monotone. Absolute occ-rank1 NOT comparable to
  pre-D38 values (harder bounds + 6.7x eval identities; a 3,520-class CE head gets
  only 600 steps in 3 ep). (4) perun_sweep_v2.md delivered: arms A-D, log-scale pools
  {300, 1k, 2k, 3.52k} x 3 seeds, detector arm at scale, <= 25 H200h (ceiling 40),
  pre-registered decision rules incl. D>C verdict criteria and the D35 val-manifest
  interaction (PERUN winner at val requires user-approved manifest amendment).
  AWAITING user review of the doc before any PERUN submission.
- **D41 (2026-07-24) History purge [user-approved] + guard-loophole fix.**
  (1) PURGE: audited every image/video ever added across all refs (9 files);
  exactly ONE contained dataset pixels: viz/crops_assoc_scope.png (MOT17 GT crops,
  present in history 1a5a0ed..0d962da). Tool: git-filter-repo 2.47.0
  (--invert-paths --path viz/crops_assoc_scope.png --force). Repo size-pack
  17.55 -> 16.46 MiB; post-purge scan: zero occurrences in any ref. All commit
  hashes from 1a5a0ed onward rewritten (new HEAD lineage); pre-purge backup bundle
  at ..\occlusion_mot_prepurge_20260724.bundle (local only — contains the purged
  content; delete after confidence window). Force-push to the private remote:
  user-approved in the same directive, EXECUTED immediately after this commit lands
  (this entry is written pre-push; remote verification logged in the response).
  Local working-tree copy of the crop grid survives untracked
  (regen: render_dev_viz --pngs-only). (2) GUARD FIX (D39
  loophole, user-flagged): path-prefix allowlisting could not distinguish content
  classes — a MOT crop grid under demo/ would have passed. Replaced with an
  explicit per-file allowlist in tests/test_license_guard.py: every tracked visual
  must be classified "plot" (pure matplotlib) or "carla-render" (synthetic);
  8 files currently allowlisted; crop-grid names are unallowlistable by
  construction; any new tracked visual fails CI until a human classifies it.
  demo/ committed set re-verified under the fixed guard (teaser + blueprint grid +
  s5 clip = carla-render; no README path changes needed — the crop grid already
  lived untracked in viz/ with its regen command documented). Post-purge guard run:
  130 tests green, ruff clean. DESIGN NOTE (logged 2026-07-26, user-prompted):
  ALLOWED_TRACKED_VISUALS entries are EXEMPT from the 2MB size ceiling BY DESIGN —
  the ceiling is the default-deny for unclassified binaries; explicit human
  classification IS the gate for large visuals (hence the 13.4MB s5_carla.mp4 and
  6.8MB s5_carla_excerpt.gif coexist with the ceiling; verified: the >2MB check
  fails only files NOT in the allowlist).
- **D42 (2026-07-24) Wild-clip showcase pipeline [user directive].**
  scripts/showcase.py: folder of mp4s -> annotated mp4s + contact sheet, one command,
  live yolo11x + conv-embedder + OcclusionAwareTracker at the val-manifest R1
  operating point; overlays = track ids, occlusion-state coloring, dashed-orange
  hidden-agent prediction, legend + "placeholder-ckpt | qualitative only" watermark;
  ZERO metrics/GT anywhere (D36). FIXED-DETECTIONS explicitly N/A (no comparison —
  documented in the script). Sources: 10 Pexels-License clips (116MB; 4 night,
  2 rain, 2 high-angle, 2 dashcam; provenance table in showcase/sources.md —
  committed; no YouTube). Renders are source-pixel-derived -> gitignored
  (showcase/sources/, showcase/renders/); guard gains reserved class
  "licensed-stock-render" with NOTHING classified under it yet — committing any
  render requires license verification + explicit allowlist entry. Renders are
  DISPOSABLE placeholder-ckpt output; final re-render post-PERUN with the canonical
  checkpoint. Batch runtime: ~12 fps processed, 18-60s wall/clip (3,824 frames
  total). 27 new synthetic tests (154 green).
- **D43 (2026-07-24) perun_sweep_v2.md APPROVED [user] + 4 pre-registration
  amendments (appended to the doc, no design changes).** (1) Detector-arm R6
  selection rule pre-registered: dev oracle-ceiling, tie-break e2e at fixed gate —
  no post-hoc selection. (2) Gate probe: ONE gate per arm chosen by 3-seed mean
  assoc; never per-seed. (3) "dev-optimistic" label mandatory wherever a dev-trained
  detector is scored on dev; only R6-at-val is honest. (4) MSMT17 HF-mirror path
  STRUCK (license-hygiene violation class D39/D41); official request form only,
  post-val, user-gated. NEXT (same directive): local sweep dry-run as PERUN parity
  check — SLURM launcher + config-driven entrypoint (identical entrypoint local vs
  PERUN, no code fork), arms A-D @ pool 300 / 1 seed / 3 ep end-to-end incl.
  McNemar emission, detector smoke, runtime extrapolation vs the 25 H200h budget.
  DRY-RUN PASSED (2026-07-24): 4 arms + detector smoke end-to-end through the
  parity entrypoint (sweep_unit.py; sbatch emits the identical command — pinned by
  a test); no OOM/NaN; unit JSONs schema-complete incl. taxonomy guard
  (pre_match_rate/oracle_ceiling verbatim from run_hidden); 3 McNemar JSONs
  parseable. All dry-run METRICS are meaningless by design (3 ep, pool 300,
  1 seed — validation only, no claims; e.g. 3-ep arm D scores BELOW the ImageNet
  null). Two portability bugs caught by the dry-run: cp1250 subprocess-pipe decode
  crash on ultralytics output (FIXED: utf-8/errors=replace; residual non-fatal
  console-logging encode warnings remain on Windows only — irrelevant on the Linux
  target) and the hidden_ tag prefix mismatch between run_hidden outputs and
  paired_test consumption (FIXED).
  EXTRAPOLATION (corrected after reviewer catch — the first draft wrongly reused
  the D40 600-step basis; the dry-run config actually ran 3 ep x 20 batches = 60
  steps): honest decomposition from the D40 sanity run (200-batch epochs, warm
  cache, 4060): train ~0.45 s/step @ batch 64; occluded-query eval ~140 s over the
  98k-crop val set. Full-scale unit (60 ep x 400 batches = 24k steps + 60
  per-epoch evals) ~ 10,800s + 8,400s ~ 5.3h on 4060. The pipeline is CPU-decode-
  bound (single-threaded loader), so the H200-node factor is conservatively ~3x,
  not 8x -> ~1.8 H200h/unit, 18 units ~32 H200h: the doc's <= 25h estimate does
  NOT survive measured throughput AS-IS (the 40h ceiling barely holds). Eval
  cadence is a free implementation parameter (unspecified in the approved design;
  runs/arms/seeds/epochs unchanged): evaluating every 5 epochs cuts eval cost 5x
  -> ~1.2 H200h/unit. BUDGET CORRECTED AT D43 CLOSE-OUT [user]: my ~25h total had
  silently dropped blocks — honest total = embedders 21.6 + caches ~2.6 + detector
  6-12 = ~30-36 H200h; the 25h estimate does NOT hold, the 40h ceiling holds.
  Standing rule: no block ever drops out of a total silently again. STAGING
  [user pre-registration, verified]: the pre-resized-64x128 staging requirement is
  ALREADY SATISFIED BY CONSTRUCTION — all four extractors write 64x128 at
  extraction (Market natively so); verified on disk (4/4 sources 64x128, 1.38 GB).
  Pre-resizing is therefore a no-op; residual decode cost is single-threaded
  tiny-jpg decode — optional extra levers: parallel DataLoader / uint8 npy shards. PERUN config deltas owed BEFORE submission: (a) --eval-every
  in train_reid + sweep configs (budget-load-bearing); (b) detector `mix` dimension
  is label-only (finetune_detector lacks the CARLA-mix path); (c) emit
  submit.sbatch from the Linux side (path separators); (d) fill
  partition/account/time placeholders.
- **D44 (2026-07-24) README update [reconstructed from user AMENDMENT — the base
  D44 directive itself was never received; only "Amendment to D44 directive"
  arrived, appended to a resend of the D43 close-out text].** Acted on the
  amendment's explicit requirements only: README.md now states the corrected sweep
  budget (~30-36 H200h), includes the two-command PERUN submission path as
  readiness evidence, and carries the status line "submission-ready, awaiting HPC
  access"; plus a minimal freshen (dev-half state summary with the dev-optimistic
  caveat, layout pointers to val_manifest/perun_sweep_v2/demo). Scope beyond the
  amendment NOT invented — if the base D44 directive contained more, it needs
  re-sending. NOTE: the resent D43 items (budget correction, staging, deltas a-d)
  were already complete at commit d1ceb57; not redone.
  BASE DIRECTIVE RECEIVED + FOLDED IN (2026-07-26): full README rewrite for the
  ML-professor / internship audience — problem, architecture (own ByteTrack,
  frozen-det protocol, embedder, occlusion taxonomy), methodology table (gates
  G0-G4 status, val freeze, D36 multi-seed discipline, license guards), honest
  results table (dev-half; progression claimed only where it exceeds the 6pt seed
  band; embedder-variant deltas explicitly NOT claimed, p=0.26 stated;
  dev-optimistic label on the detector arm), data-engine table (3,520/506/509k),
  PERUN section retained (budget/two-command/status line), quickstart, repo map,
  cross-link to demo/README.md without duplication. MEDIA (guard-compliant):
  new demo/s5_carla_excerpt.gif (6s, 6.8MB, pure-CARLA excerpt of the committed
  s5 clip) classified carla-render in ALLOWED_TRACKED_VISUALS BEFORE commit;
  embedded plots viz/scaling_curve_local.png + viz/summary_retention_grid.png
  (class plot, already allowlisted). MOT/Pexels renders: regeneration reference
  only, one line "available locally / on request" — nothing committed.
- **D45 (2026-08-09) D18 VAL EVENT EXECUTED [user approval received].** val_manifest.md
  (frozen f2e0102) run EXACTLY: R1-R6, zero edits, zero deviations, no re-runs.
  Pre-flight: both ckpt sha256 MATCH; MOT17 manifest sha 49c218a3 == D13; 133 segments;
  pytest 240 green; CUDA probe ok. Execution notes (faithful-interpretation calls, logged,
  no semantic deviation): (a) manifest's frozen-code hash bc46f72 no longer exists post-D41
  purge — rewritten equivalent a8e4da0 (D28-final) located; only val-critical diff since is
  the L5 cache_tag routing in detect/cache.py that the manifest's own R6 requires, paired
  path byte-identical. (b) The D45 directive's premise "val-half detections not cached" was
  FALSE — the D1 yolo11x cache covers full sequences (verified frame ranges vs seqinfo all
  7 seqs), as the manifest itself states for R6; no new detector pass run (FIXED DETECTIONS
  preserved — running one would have violated "D1 cache, untouched"). (c) R3/R4 given as
  config shorthand in the manifest — expanded to exact flags via the frozen dev precedents
  (tune_hidden.py run_combo devbase for R4; grid-4 dev-best app0.40 proto for R3).
  RESULTS (val half, n=133): G1 PASS (HOTA 51.07/IDF1 60.09/IDsw 298 vs base
  49.98/58.28/359). G2: center_err 0.0090 PASS, cov_prematched 0.907 PASS, n_segments PASS;
  G2a-floor 0.440 vs 0.58 FAIL (~2.4 sigma below, NOT within noise); G2a-paired p=0.4073
  n=83 NOT MET (pre-registered expected outcome, pending PERUN); G2b e2e 0.278 contingent/
  non-blocking. R6 detector-arm (dev-half-GT caveat label mandatory): e2e direction
  replicated (0.278->0.338), assoc 0.577, but oracle-ceiling lift did NOT transfer
  (0.632->0.586). Full report: results/val/val_report.md. Per D18.3 + manifest header:
  misses reported, work on val-gated claims HALTS pending user decision.
- **D46 (2026-08-09) G2a-floor val miss ACCEPTED [user].** Margin deferred to the
  PERUN-scale embedder per the manifest's pre-registration; NO re-tuning, NO re-scoping
  — thresholds unchanged, manifest unchanged (annotation-only footnote added for the
  D41 hash remap bc46f72->a8e4da0, marked post-hoc). D18.3 post-val decision blocker
  RESOLVED; remaining blocker: PERUN access. README patched with the honest val
  section (G1 PASS, center/coverage PASS, G2a-floor MISS with the dev->val
  generalization gap named as the finding, paired p=0.41 pre-registered-expected,
  R6 with mandatory caveat label); status line updated. Framing per user directive:
  "pre-registered val executed as frozen; misses reported, not tuned away."
- **D-INCIDENT (2026-07-22) Visibility check during repo rename [user-filed].** Repo was
  PUBLIC during the rename step despite the instruction to confirm visibility and flag.
  Session record: visibility was queried (gh repo view -> PUBLIC) and flagged at the top of
  the rename reply; however work PROCEEDED past the flag without waiting for user
  acknowledgment, and the post-set-url verification checked reachability (ls-remote), not
  the asked property. BINDING corrective rule: any explicit report-back item must be
  answered verbatim AND acknowledged before proceeding; never substitute an adjacent check.
  Repo made PRIVATE by owner action 2026-07-22.
- **D18 (2026-07-22) Val-half protocol, BINDING [user directive].** (1) All config selection
  on dev-half metrics only; G1-regression proxy on dev-half; val never used to compare or
  select. (2) Exactly ONE config advances; the single canonical val-half run requires
  explicit user approval — report top-3 dev configs (retention, center-err, dev G1-proxy
  deltas) + pick + rationale, then HALT. (3) If the val run misses targets, re-tuning +
  re-running requires explicit approval (contamination decision, user-owned). Enforced in
  workflow: run_hidden.py --canonical exists but is only invoked after approval.
- **D17 (2026-07-22) Repo renamed occulsion_mot -> occlusion_mot** (typo fix; rename was
  already applied on GitHub when checked — old URL redirects). Local remote updated to
  https://github.com/matus012/occlusion_mot.git. Visibility re-confirmed PUBLIC and flagged
  to user per instruction (matches the D12 approval; user decides any change themselves).
- **D16 (2026-07-22) G2 position/retention thresholds frozen from measured baseline floor**
  (results/baseline_hidden.json, 133 val-half segments): baseline id_retention 0.263,
  coasting center-err median 0.0155 diag at 75.2% coverage, time-err median 25.5 frames.
  Frozen: center_err_med <= 0.015 (beats coasting AND stay-put 0.020), coverage >= 0.90
  (new criterion — forbids cherry-picking), id_retention >= 0.55 (2.1x baseline).
  time_err stays provisional per user amendment (freeze_order last); gate-level frozen flag
  stays false until the hidden-state module exists. The 26.3% baseline retention is the
  headline motivation number: ByteTrack loses 3 of 4 identities through occlusion.
- **D14 (2026-07-22) Occlusion-segment definition revised after contact with real GT.**
  D7's rule (all gap frames vis < 0.25, anything in [0.25, 0.5) contaminates) yielded 1
  segment on MOT17-02 because MOT GT visibility decays gradually through the 0.25-0.5 band.
  Revised: anchors vis >= 0.5; gap frames stay < 0.5; at least one dip < 0.25 (unannotated
  frames count as occluded); min_len 5; pedestrian class + consider flag only. Calibration:
  MOT17-02 -> 68 full / 41 val-half segments, median gap 37 frames (n.b. > ByteTrack's
  default 30-frame track buffer — exactly the failure mode the hidden-state module targets).
  Same numeric thresholds kept (0.25 / 0.5 / 5); only the contamination semantics changed.
- **D50 (2026-08-19) PERUN sweep 77150 COMPLETE — G2a FAILS on both limbs at PERUN scale.**
  23/23 units, zero task failures, zero cap kills. Actual compute **7.74 H200-h** against a
  32.25-34.25 h projection and a 40 h ceiling: the UNIT_EST_H estimates were derived from
  RTX 4060 timings and overstate H200 cost ~4.3x. The estimates are not moved (they are
  pre-registered and they bound the budget conservatively), but the discrepancy is logged
  here so a future grid can be sized honestly.
  **G2a-floor: FAIL.** Arm D (pool 3520, gate 0.45) assoc_mean **0.5779** vs the frozen
  floor **0.58** — short by 0.0021, i.e. a hair, but the threshold does not move (CLAUDE.md).
  **G2a-paired (THE discriminating criterion, D28-FINAL): FAIL, and not marginally.**
  D vs A McNemar p = 0.1431 / 0.6612 / 0.5000 across seeds 0/1/2 (n=95); D vs C
  p = 0.1445 / 0.6128 / 0.7734; C vs A p = 0.3388 / 0.6682 / 0.3318. Nothing approaches
  p < 0.05. The trained embedder is NOT distinguishable from the ImageNet-R18 null at
  tracker level, at full PERUN scale, with 3 seeds.
  **Why: the retrieval/tracker disconnect of D29/D36, now measured on a log-scale identity
  curve.** occ-rank1 rises monotonically with identities (300 -> 0.408, 1000 -> 0.514,
  2000 -> 0.577, 3520 -> 0.610) and is still climbing at the pool ceiling. Tracker
  id_retention_assoc moves far less (0.523 / 0.540 / 0.563 / 0.578) and the per-pool SEED
  SPREAD reaches 0.0776 — larger than the entire 0.055 effect across a 12x identity range.
  D36's "single-seed tracker deltas < ~6pt = noise" holds at scale; 3 seeds is not enough
  to resolve an effect this small. Identity count was the binding constraint on RETRIEVAL,
  as D32/D36 said, but retrieval is not the binding constraint on ASSOCIATION.
  **Anomaly logged: occ-rank1 is NOT comparable across arms.** sweep_unit.py sets
  `eval_sources = arm_sources(cfg, arm)`, so each arm is scored on a gallery drawn from its
  OWN sources. Arm B (sim only, 36 train identities) posts the highest retrieval of the
  whole sweep, occ_rank1 0.875, while producing the LOWEST tracker association, 0.549 —
  below the ImageNet null's 0.557. That is a small-gallery inflation artifact, not a real
  win, and it means perun_sweep_v2.md's "occ-rank1 primary at sweep scale" cannot carry a
  cross-arm claim. Within-arm (the D pool curve) the comparison is sound: same sources,
  same held-out gallery, only pool size varies. The pre-registered discriminating criterion
  is unaffected — it is tracker-level on a fixed intersection denominator.
  **Detector workstream (G2b, dev-optimistic label stands):** yolo11s mot17dev mAP50 0.9361
  / mAP50-95 0.6385 (0.48 h); mot17dev_carla ran 2.42 h against the 2.50 h cap — an 8 min
  margin, which retroactively confirms amendment 7: yolo11m at ~3.15x GFLOPs would have
  been killed at the cap in both mixes. Only the R6-at-val number is honest (D43 amd 3).
  **Consequence:** appearance re-ID at PERUN scale does not deliver G2a. The remaining
  levers are (a) more seeds to resolve a sub-noise effect, which is expensive for an effect
  this small, (b) the detector/oracle-ceiling workstream (G2b), or (c) accepting the
  negative result as the finding. User decision required; nothing is re-scoped unilaterally.
- **D51 (2026-08-19) Null ACCEPTED as the finding; G2b detector workstream authorized
  [user decision: (c) + (b)].** The PERUN-scale embedder claim is CLOSED. D46 deferred the
  G2a margin to "the PERUN-scale embedder"; D50 ran that test properly powered (23 units,
  3 seeds, pre-registered) and it fails both limbs. Per D18.3 discipline the outcome is
  reported, not tuned away, and no threshold moves. Write-up executed in three places:
  (a) **README** — the sweep section is rewritten from "designed, submission-ready" to
  "executed; the null is the finding", carrying the full gate scoreboard, ALL nine McNemar
  p-values (no cherry-picking), the identity-curve figure, the mechanism, and the
  cross-arm caveat stated prominently rather than buried.
  (b) **perun_sweep_v2.md amendments 8 + 9** — annotations in place, no history rewritten.
  Amendment 8 demotes occ-rank1 from "primary at sweep scale" to a within-arm diagnostic
  and bounds exactly what may still be said with it; the original line is left intact with
  a STRUCK pointer, matching the amendment-4 precedent. Amendment 9 records the 4.3x
  UNIT_EST_H over-estimate as a measured recalibration WITHOUT retro-editing the
  pre-registered numbers, and qualifies amendment 7: the detector class is bimodal by MIX
  (0.485 h vs 2.423 h for the same model), so yolo11m would have fit on mot17dev (~1.5 h)
  though not on mot17dev_carla (~7.6 h). Amendment 7 was applied exactly as written — its
  lever is model-level — so the action stands; the lesson (cap per unit, not per model
  class) is carried into the next design rather than used to re-litigate the last one.
  (c) **P2 claim reported as NOT ESTABLISHED.** The pre-registered D > C rule requires mean
  assoc(D) > (C) AND mean occ-rank1(D) > (C) with non-overlapping seed ranges. The first
  holds (0.5779 > 0.5674); the second fails (0.6095 < 0.6180) and is cross-arm-invalid
  under amendment 8. "sim+real > real-only" is therefore not established — stated plainly
  in README rather than being quietly dropped.
  **G2b workstream:** pre-registration drafted as perun_detector_v1.md against the 32.26 h
  remaining under the unchanged 40 h ceiling, with UNIT_EST_H rebuilt from measured H200
  throughput. VAL IS THE OPEN QUESTION and is flagged, not assumed — see D52.
- **D52 (2026-08-19) A G2b val event is NOT pre-authorized — flagged, not assumed.**
  Checked what D45's pre-registration actually permits, per user instruction. Findings:
  `val_manifest.md` binds "the **single** D18 val event", executed once on 2026-08-09; it
  is written for one pass and contains no re-use provision. D18.2 allows exactly ONE
  config to advance and requires explicit user approval for the canonical val run. D18.3
  makes re-running a user-owned contamination decision requiring explicit approval. D46
  accepted the G2a-floor miss and deferred to the PERUN embedder — that deferral is now
  discharged by D50 — but authorizes nothing further. And R6 has ALREADY spent a
  detector-arm read at val (e2e 0.278 -> 0.338, oracle-ceiling lift did NOT transfer,
  0.632 -> 0.586). **Conclusion: a G2b val pass would be a SECOND val event, needs its own
  justification and its own frozen manifest, and must be reported as a second-look result
  with the first look disclosed — val has been observed once, so a second read is not a
  virgin read and its nominal error rate is not what it appears to be.**
  Design response, so the workstream does not depend on that decision: perun_detector_v1.md
  trains the detector ONLY on MOT17-disjoint sources (MOT20 + CARLA) and evaluates on
  MOT17 dev-half, which is honest by construction and consumes no val. This also repairs
  the `dev-optimistic` label (D43 amd 3) that forced the previous detector arm to reach for
  val in the first place. Stages 0-1 are entirely dev-side; a val event is proposed only if
  the dev dose-response clears, and is the user's call either way.
- **D53 (2026-08-19) perun_detector_v1.md FROZEN; Stage 0 authorized, Stage 1 blocked
  [user decisions on all four open questions].**
  (1) **VAL CLOSED.** Stages 0-1 are dev-only; no val event is authorized under this
  document. A positive Stage-1 slope earns only the right to DRAFT a second frozen val
  manifest with second-look disclosure, for separate approval — drafting is not approval.
  (2) **Stage 0 runs and reports ALONE**; Stage 1 is blocked on its verdict AND an explicit
  user go. Explicitly forbidden: queueing Stage-1 units in anticipation, or chaining the
  stages with a SLURM dependency — the gate is a human decision point, not a scheduler edge.
  (3) **Grid as drafted, no widening**; the 10.6 h headroom is margin, not an invitation.
  (4) **MOT20 incremental transfer approved**, launched in background parallel to Stage 0.
  Section 7 rewritten from open questions to frozen decisions; doc title and status header
  updated from DRAFT to FROZEN.
- **D54 (2026-08-19) Stage-0 specification gap resolved BEFORE the run: `gtvis` drives the
  kill, `gtall` is reported alongside.** perun_detector_v1.md section 2 said "GT boxes as
  the detection stream" but justified the kill with "the supremum of any DETECTOR's
  output". MOT17 GT annotates fully-occluded targets (visibility 0) — **10,485 boxes, 9.3%
  of all consider-flagged pedestrian annotations** (112,297 vs 101,812 with vis > 0). A
  stream including them is not a detector; it sees through occluders and would hand the
  hidden-state module the very answer it exists to infer. Adopted, and fixed before any
  number was seen: **`gtvis` (vis > 0) is PRIMARY and gate-bearing**; `gtall` is an
  upper-upper bound, reported but never gate-bearing. Deliberately the stricter choice —
  `gtall` would be too permissive and could clear 0.55 in a world where no real detector
  could, sending ~21 h after an unreachable ceiling, which is the exact failure the kill
  gate exists to prevent. Logged as perun_detector_v1.md amendment 1 rather than chosen
  silently, since the document was frozen at D53. Implementation
  `scripts/cache_gt_detections.py`, byte-compatible with the ordinary detection cache so
  no downstream consumer can tell a GT cache from a detector cache; 6 synthetic tests
  including a containment guard proving gtall superset gtvis.
- **D55 (2026-08-19) STAGE 0 PASSED — the detector IS the binding constraint. Kill gate
  does not fire; Stage 1 is now unblocked-in-principle but still HELD per D53 decision 2.**
  Three arms, frozen canonical tracker (b90/d1.0/g1.5/overlap 0.25/kf/app0.45, conv
  embedder), MOT17 dev half, n=168 segments. ONLY the detection source varies (the
  FIXED-DETECTIONS inversion of perun_detector_v1.md section 1).

  | arm | pre_match | oracle_ceiling | e2e id_retention | assoc | center_err |
  |---|---|---|---|---|---|
  | yolo11x (current) | 0.762 | 0.571 | 0.345 | 0.604 | 0.00639 |
  | **gtvis (PRIMARY)** | 0.994 | **0.970** | **0.786** | 0.810 | 0.00037 |
  | gtall (not achievable) | 1.000 | 1.000 | 0.952 | 0.952 | 0.00032 |

  **Kill criterion (fires below 0.55): measured 0.786 on the gate-bearing gtvis arm.
  PASSES by +0.236.** D25/D26's mechanism is confirmed at the ceiling: detection quality
  accounts for almost the entire oracle_ceiling gap (0.571 -> 0.970), and e2e retention
  more than doubles (0.345 -> 0.786) with the tracker completely untouched. G2b at 0.55 is
  reachable in principle, with ~0.24 of margin above the gate.
  **This corrects the pessimistic prior I recorded in perun_detector_v1.md section 0.**
  That prior rested on D45's R6: a yolo11s finetuned on dev-half GT failed to lift the val
  ceiling (0.632 -> 0.586). Both readings are true and compatible — R6 says *that
  particular finetune was not better than yolo11x*, Stage 0 says *a better detector has
  enormous headroom*. The prior was evidence about one weak detector, not about the
  mechanism. Stage 1 remains the test of how much of the 0.399 ceiling headroom a
  realistically trainable detector actually captures; gtvis is an upper bound, NOT a
  prediction, and must never be reported as an achievable number.
  **Residual finding, logged for later:** even at perfect visible detection, assoc
  retention is 0.810, not 1.0 — ~19% of associable segments are still lost inside the
  tracker. That is an association-side gap the detector workstream cannot close, and it
  bounds what G2b can reach even with a perfect detector. Note also reemergence_time_err
  stays at 21.5 frames (provisional gate <= 5, non-gate-bearing) and does NOT improve with
  perfect detections — it is not detector-limited.
  Implementation D54; figure viz/stage0_kill_gate.png; raw
  results/hidden_dev_stage0_{yolo11x,gtvis,gtall}.json. Cost: ~0 H200-h (run locally on the
  RTX 4060; the budgeted 0.20 h was not spent, so 32.26 h remains under the 40 h ceiling).
  **Stage 1 is NOT submitted and MUST NOT be: D53 decision 2 requires the verdict to be
  reported to the user and an explicit go before any Stage-1 unit is queued.**
- **D56 (2026-08-19) Stage-1 implementation: MOT17-disjoint mixes, unit runner, launcher.**
  Built to perun_detector_v1.md section 5. (a) finetune_detector.py gains `mot20` and
  `mot20_carla`; MOT20 is not an eval half so `_split_all_frames` uses every frame, while
  the MOT17 family keeps its D18 half-split guard untouched. Disjointness is asserted
  TWICE — on resolved source paths (`_assert_no_mot17`, which resolves symlinks so a
  MOT17 sequence renamed `MOT20-99` is still caught) and again on the materialised
  dataset. (b) detector_unit.py runs a unit end to end (finetune -> cache detections over
  MOT17 -> conv embeddings -> FROZEN canonical tracker) and emits both dose-response
  coordinates per unit. (c) detector_launcher.py emits the array and, in local mode,
  writes the pre-registered OLS slope + 95% CI.
  **Caps transcribed, not derived.** My first launcher derived wall caps from the cost
  model and produced 01:00:00 where the frozen doc's section 4 table says 01:15:00 —
  tighter than the pre-registration, so it would have killed units the doc permits to
  finish. Caps now come verbatim from the doc into config `wall_caps`, with an assertion
  that each sits above its own estimate. Worst case 27.00 h vs 32.26 h remaining.
  Tests: 7 mix/disjointness guards (incl. the renamed-MOT17 smuggling case and a
  regression that MOT17 keeps its half-split), 11 launcher guards (incl. the D49
  timeout-grammar assertion re-asserted for the new emitter, a budget guard proved to
  fire, and a flat-curve OLS case proving a null cannot look significant).
- **D57b (2026-08-19) PERUN site policy discovered: submission is gated at 4 nodes per
  user.** `sbatch` refused the second Stage-1 batch with *"celkovy limit je 4 nody (mate 7
  short + 0 long = 7, pyta 1)"*. It is a SUBMIT-TIME check against jobs already held, not a
  run-time concurrency cap — which is why every `%8` array sailed through: an array is a
  single submission regardless of its width. Consequence for operations: a partial
  re-submission (the D57 recovery path, and the resume rule generally) can be refused
  purely because earlier units are still running. Handled with a retrying submitter rather
  than by widening anything; the `%8` cap and every wall limit are untouched.
- **D60 (2026-08-19) Stage-1 x-axis defect caught on the first four units — it would have
  manufactured a null.** perun_detector_v1.md s3 fixes the dose-response x-axis as
  mAP50-95 **measured on MOT17 dev-half**; my implementation read it from ultralytics'
  `results.csv`, which validates on the TRAINING mix's own split (MOT20). Two different
  datasets on the two axes. Symptom: across the first four units the results.csv mAP
  spanned **0.005** (0.6011-0.6060) while oracle_ceiling spanned 0.083 — regressing on a
  near-constant x gives a huge CI, which the pre-registered kill criterion would have read
  as "CI includes zero -> mechanism not established". The bug's failure mode was a FALSE
  NULL, which is exactly the direction a careless reading would have accepted as a result.
  Fixed by `scripts/eval_detector_map.py`, computing mAP on MOT17 from the same cached
  detections the tracker consumes. Instrument-proved before use: `gtvis` (built from GT)
  scores exactly mAP50-95 = 1.0000, `yolo11x` scores 0.3969. 10 unit tests. The old number
  is retained as `map50_95_mot20_split`, not discarded. No gate, threshold or grid change.
  **Second observation, independent of the bug:** the MOT20-trained detectors transfer
  POORLY to MOT17 — oracle_ceiling 0.250-0.333 against the yolo11x baseline's 0.571, and
  e2e 0.167-0.202 against 0.345. Honest training on a disjoint domain costs real accuracy.
  That is itself a finding about the section-1 design and must be reported alongside
  whatever the slope says.
- **D61 (2026-08-19) STAGE 1 COMPLETE — mechanism CONFIRMED, intervention FAILED.**
  12/12 units, zero task failures, **10.72 H200-h actual** against a 21.43 h nominal
  estimate and a 27.00 h worst case at the caps. Cumulative spend 18.46 h of the 40 h
  ceiling; 21.54 h remain. The ceiling never moved.

  **Pre-registered primary (perun_detector_v1.md s3, all 14 dose-response levels):**
  `oracle_ceiling ~ mAP50-95(MOT17)` slope **0.9021**, 95% CI **[0.7837, 1.0205]**,
  R² 0.949. The CI excludes zero, so **the detector-ceiling mechanism is SUPPORTED** —
  and the slope is near 1:1, i.e. a point of detector mAP buys about a point of
  recoverable-occlusion ceiling. Secondary `id_retention ~ mAP` slope 0.7738,
  CI [0.7313, 0.8163], R² 0.991. This is the strongest quantitative confirmation the
  project has produced of D25/D26's claim that end-to-end retention is detector-capped.

  **G2b verdict: NOT MET, and not close.** Best trained unit `yolo11s:mot20_carla:2` at
  e2e 0.2024 against the frozen 0.55 — and **0.1428 BELOW the yolo11x baseline's 0.3452**.
  Every one of the 12 trained detectors is worse on MOT17 than the off-the-shelf detector
  it was meant to improve on (mAP50-95 0.189-0.228 vs the baseline's 0.397; ceiling
  0.232-0.345 vs 0.571).

  **Why: the honesty tax.** Section 1 chose MOT17-disjoint training (MOT20 + CARLA) so the
  dev read would be honest by construction and consume no val. That worked as designed —
  the numbers are clean — but MOT20 is a far denser, different-domain benchmark, and the
  domain gap cost more accuracy than finetuning gained. The detectors also over-fire
  badly: 72k-226k detections over MOT17 dev-half against the baseline's 80k for the same
  53,678 GT boxes. **The design bought honesty at the price of the very quality it was
  trying to demonstrate.** That trade is the finding, and it was not visible before the run.

  **Sensitivity analysis, reported because it qualifies the headline.** The pre-registered
  fit spans x = 0.189 -> 1.000, but the 12 trained units occupy only 0.0391 of that range;
  the lever arm comes from the two reference points. Restricted to trained units alone:
  `oracle ~ mAP` slope 2.2374, CI [0.7556, 3.7192], R² 0.467 — still excludes zero, so the
  ceiling relationship survives; but `e2e ~ mAP` slope 0.4375, CI **[-0.3808, 1.2558]**,
  R² 0.099 — **includes zero**. So within the narrow band of detectors actually trained
  here, end-to-end retention does NOT track detector quality significantly. The
  pre-registered verdict stands as written (it was defined over all 14 levels); this
  caveat is recorded so the headline is not read as more than it is.

  **What this does and does not license.** It confirms WHERE the bottleneck is; it does
  NOT demonstrate that training a better detector is achievable by this route. The
  actionable next step is a better MOT17-domain detector, which conflicts with the
  disjointness that made this read honest — that tension is the real open problem, and it
  is not resolved by anything in this document.
  Figure viz/stage1_dose_response.png; summary results/sweep/perun_detector/summary.json.
- **D63 (2026-08-19) val2 REJECTED [user]; Stage-1 detector demo re-render SKIPPED [user].**
  The Stage-1 slope was positive, which under D53 decision 1 permitted *drafting* a second
  val manifest. Drafted with a recommendation against executing it — a val event exists to
  read a candidate honestly, and Stage 1 produced no candidate worth reading (every trained
  detector is below the yolo11x baseline). User agreed: **val stays read exactly once
  (D45)**. `PROPOSED_val2_manifest.md` is retained, marked REJECTED, as the record of what
  was proposed and declined; it authorises nothing. Separately, the planned second demo
  render "with the best Stage-1 detector" is cancelled for the same reason — a hero clip
  built on a detector that is worse than the baseline would misrepresent the result. The
  demo suite therefore ships on the current best stack (conv embedder + yolo11x
  detections), which is also the configuration every headline number in the README refers
  to. Session state consolidated into `SESSION_QUEUE.md` (D62) so a zero-context session
  can resume.
- **D64 (2026-08-19) Demo scope corrected: hero clip rebuilt as the side-by-side it was
  always meant to be.** I had over-applied D63 and treated the whole demo item as
  cancelled; user clarified that only the *Stage-1 trained-detector variant* was dropped.
  The demo ships on the **current best stack** — yolo11x cached detections + geometric
  hidden-state + trained conv embedder @ app-gate 0.45, the exact configuration behind the
  dev and val tables.
  The existing `hero.mp4` was not what had been asked for: a four-stage progression
  (S1->S2->S3->S4) with per-stage title cards and a CARLA tail, ~63 s. That is a tour, not
  a hook. `render_hero` rewritten to produce **one side-by-side clip, ~31 s**: ByteTrack
  baseline left, full stack right, same frames and same cached detections, persistent ID
  labels, across **two segments from two different sequences** (MOT17-09 t10, MOT17-04 t89)
  so it cannot be dismissed as one lucky pick — 2 s opener, 3 s outro carrying the headline
  numbers. `render_pair_clip` gained a `canvas_size` parameter mirroring `render_solo_clip`
  so multi-segment concat cannot fail on a frame-size change.
  Also fixed: `demo/README.md` never mentioned the hero clip at all. The tour now opens
  with a "Watch this one first" section, and the top-level README points at it too, both
  stating that the Stage-1 detectors are deliberately absent because every one of them
  scored below the baseline detector (D61) — showing one would misrepresent the result.
  License rules unchanged: `hero.mp4` contains MOT17 pixels, stays gitignored, regenerable
  with `--only hero`.
- **D66 (2026-08-19) CARLA segment count reconciled to the artifact: 223 -> 225.**
  `results/carla_feeder.json` records `n_occlusion_segments: 225`; two documents still
  carried 223. Corrected in `status.txt` (current-state, so it is simply wrong there) and
  **annotated, not rewritten**, in the D23 entry above — dated decision entries are
  historical record and the D-log rule forbids editing them silently, so the annotation
  states the artifact value and leaves the original text standing. Also corrected my own
  error in `SESSION_QUEUE.md`: I had reported the stale value as living in `README.md`,
  which never carried it. No gate, threshold or result changes; this is a provenance fix so
  every published count traces to the committed artifact.

- **D67 (2026-09-04) PROJECT CLOSED — close-out session.** State set to CLOSED in
  `status.txt`; final compute **18.46 / 40 H200-h** (embedder sweep 7.74 + detector
  Stage 1 10.72; Stage 0 ran locally at ~0 H200-h), 21.54 h unspent, the 40 h ceiling
  never moved. Delivered: the G0-frozen tracker, the hidden-state module (parity gates
  PASS on held-out val, retention 0.292 -> 0.345), the CARLA data engine (24 scenarios,
  225 segments, exact per-walker visibility GT), the offline hash-verified HPC pipeline,
  and `report/omot_report.md`. Proven: end-to-end retention is **detector-capped** — GT
  visible boxes take retention 0.345 -> 0.786 and the ceiling 0.571 -> 0.970, dose-response
  slope 0.902 CI [0.784, 1.021] R² 0.949. Null and reported as such: the trained embedder
  never beat the ImageNet null at tracker level (best paired p = 0.143 vs frozen 0.05);
  all 12 MOT17-disjoint-trained detectors scored below off-the-shelf yolo11x, so **G2b is
  NOT MET**; *sim+real > real-only* is **not established** by its own pre-registered rule;
  and G2's 0.55 retention target is **not met** (0.345 dev / 0.278 val). **No model trained
  in this project improved end-to-end tracking, and nothing here claims otherwise.** A
  `## Headline` section was added at the top of `README.md` carrying exactly these numbers,
  matched to the report.
  **The three open forks are recorded as FUTURE WORK, NOT PURSUED:** (1) a better
  *MOT17-domain* detector, which conflicts with the disjointness that made the Stage-1 read
  honest — the tension is unresolved; (2) the ~19% of recoverable segments lost *inside the
  tracker* even with perfect detections, which no detector work can close; (3) sim-to-real
  transfer (P2) beyond the data engine itself. None is blocked; each is simply out of
  scope at closure.
  Housekeeping: `DELETION_PROPOSAL.md` **Group A executed** on owner approval — A1
  (superseded transfer bundle `omot_hpc_f1df992.tar.gz` + sidecars), A2 (`dist/omot_hpc`
  staging tree) and A4 (`tools/CARLA_0.9.15.zip`) hard-deleted for **28.81 GiB** freed.
  All three were gitignored build artifacts: zero repository, history or result impact,
  and each is regenerable (`make_hpc_bundle.py build --reuse`; re-download for CARLA).
  A3 (`omot_hpc_8421645.tar.gz`, the provenance anchor for the deployed cluster tree) and
  A5 (the working CARLA install) were **KEPT** per the proposal's own recommendation.
  Groups B, C and D were not touched.

- **D68 (2026-09-26) Licensing docs added post-closure.** Verified the repo's own claim
  that no weights trained on MOT17/MOT20/CrowdHuman are released: `git ls-files` carries
  no `*.pt/.pth/.onnx/.engine/.ckpt/.safetensors/.bin/.zip/.tar*` (blocked by `.gitignore`
  and `tests/test_license_guard.py`), `gh release list -R matus012/occlusion_mot` returns
  no releases, and README/docs link no downloadable weights. `scripts/finetune_detector.py`
  (MOT17 dev-half), `scripts/detector_unit.py` (MOT20) and `scripts/train_reid.py` all write
  checkpoints under gitignored paths (`*/weights/best.pt`, `data/models/*.pt`) — the claim
  holds as written. Added `THIRD_PARTY.md` (every dependency + licence from `.venv`
  `importlib.metadata`, ultralytics AGPL-3.0 called out as why the repo is AGPL-3.0-only;
  datasets, CARLA, PERUN sections) and a `## Licensing & data` section in `README.md`
  pointing to it. Fixed the `occulsion` -> `occlusion` typo in `status.txt`'s "done:"
  summary line (the current, correctly-spelled repo URL). **Not fixed, deliberately:** D12
  and D17 above both use the spelling `occulsion_mot` because that was the repo's *actual*
  name at the time each entry describes (D12: creation; D17: the rename itself, which reads
  as a no-op if "fixed"). Rewriting those would falsify the decision log rather than correct
  a typo. State line updated: `status.txt` now reads `shipped (closed 2026-09-04; licensing
  docs added 2026-09-26)`. Closed the stale 2026-08-10 PERUN-submission row in
  `reports/blockers.md` (project closed 2026-09-04, no PERUN work is outstanding).
