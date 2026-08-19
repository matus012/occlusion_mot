# Stage-1 integration runbook — trigger aggregation without Claude

Written 2026-08-19 as the deadline hedge. Everything Stage 1 needs is already on PERUN and
already submitted; this file is only about **finishing** it if the array is still running
when the session budget ends.

Nothing here requires writing code. It is four commands and one decision.

---

## 0 — Where things stand

| | |
|---|---|
| Cluster repo | `/mnt/project/perun26011488/omot/omot_hpc/repo` |
| Login | `ssh login02.perun.tuke.sk` (user `mafike202`, key-based) |
| Array | 12 units, submitted in batches (see *Recovery* below) |
| Config | `configs/sweep/perun_detector.yaml` (FROZEN — do not edit) |
| Results land in | `results/sweep/perun_detector/*.json` → symlinked to Lustre |
| Budget | nominal 21.43 h, worst case at caps 27.00 h, vs 32.26 h remaining under the 40 h ceiling |

## 1 — Is it done?

```bash
ssh login02.perun.tuke.sk 'cd /mnt/project/perun26011488/omot/omot_hpc/repo && squeue -u $USER -o "%.14i %.8T %.10M" && echo "done: $(ls results/sweep/perun_detector/*.json 2>/dev/null | wc -l)/12"'
```

**Done = 12 result JSONs and an empty queue.** Fewer than 12 with an empty queue means some
units died — go to *Recovery*.

## 2 — Aggregate (this is the whole analysis)

```bash
ssh login02.perun.tuke.sk 'cd /mnt/project/perun26011488/omot/omot_hpc/repo && source scripts/hpc_env.sh && .venv/bin/python scripts/detector_launcher.py --config configs/sweep/perun_detector.yaml --mode local'
```

It runs **no new units** when every result exists. On first run it also computes each
detector's **mAP50-95 on MOT17 dev-half** from the cached detections (D60) and caches that
to `results/detmap_<tag>.json` — this is the dose-response x-axis, and it is NOT the number
in ultralytics' `results.csv` (that one validates on the MOT20 training split; see
amendment 2). Expect it to take a few minutes the first time.

It writes `results/sweep/perun_detector/summary.json` and prints the lines that matter:

```
PRIMARY oracle_ceiling ~ mAP50-95: slope=... CI=[..., ...] R2=...
mechanism supported (CI excludes zero): True|False
best e2e id_retention ... -> G2b >=0.55: True|False
```

## 3 — Read the verdict (pre-registered — do not renegotiate it)

`perun_detector_v1.md` section 3 fixed this before the run:

- **Slope positive AND the 95% CI excludes zero** → the detector-ceiling mechanism is
  supported. Then, and only then, `G2b >= 0.55` at the best single detector (reported with
  its seed spread) is the G2b verdict.
- **CI includes zero** → the mechanism is **not established**. G2b is reported as not
  achieved by this route, no further detector budget is requested, and **thresholds do not
  move**. A flat curve is a result, not a tuning prompt.
- Single-seed differences below ~6pt are noise (D36/D50) and are never claimed.

**If the slope is positive, val is still NOT unlocked.** D53 decision 1: a positive slope
earns only the right to *draft* a second frozen val manifest with second-look disclosure,
for separate approval. Drafting is not approval. Do not run val.

### Sanity checks before you believe the slope

The two reference points are computed, not hardcoded, and both are already verified on the
cluster:

| detection source | mAP50-95 on MOT17 | meaning |
|---|---|---|
| `gtvis` | **1.0000** | built from GT, so anything but 1.0 means the evaluator is broken |
| `yolo11x` | **0.3969** | the baseline the trained detectors are compared against |

If `gtvis` is not exactly 1.0000, stop and fix the evaluator before reading any verdict.

Also expect the trained detectors to sit **below** the yolo11x baseline on MOT17 —
observed 0.25–0.33 oracle-ceiling against the baseline's 0.571. Training on MOT20 (a
different, far denser domain) costs real accuracy on MOT17. That is a genuine cost of the
honest disjoint-training design and should be reported alongside the slope, not hidden.

## 4 — Pull the results back

```bash
scp 'login02.perun.tuke.sk:/mnt/project/perun26011488/omot/omot_hpc/repo/results/sweep/perun_detector/*.json' results/sweep/perun_detector/
```

Then commit them, and write the verdict into `README.md` (the *Detector workstream*
section has a slot for it) and `context.md` as a dated D-entry — to the same standard as
the embedder null: report the number that came out, not the one that was hoped for.

---

## Recovery — if units are missing

Re-submitting is safe and cheap: every finished unit exits immediately on seeing its own
result JSON (`if [ -f "$RESULT" ]`).

```bash
ssh login02.perun.tuke.sk 'cd /mnt/project/perun26011488/omot/omot_hpc/repo && sbatch results/sweep/perun_detector/submit.sbatch'
```

**Two failure modes seen on 2026-08-19, both already fixed in the shipped code — but know
the symptoms:**

1. **Site node limit (D57b).** `sbatch` refuses with
   *"celkovy limit je 4 nody (mate N short + 0 long = N, pyta 1)"*. This is a SUBMIT-TIME
   check against jobs you already hold, not a run-time cap — which is why a wide `%8`
   array is fine but a *partial* re-submission gets refused while earlier units still run.
   **Fix: wait for capacity and submit again.** Do not widen anything.
2. **Dataset-prep race (D57).** Symptom was `SameFileError` / `FileExistsError` during
   `finetune_detector.py`, killing tasks within seconds. Fixed by a tolerant
   `_link_or_copy` plus `_PrepLock`. If it ever recurs, prep the datasets **once,
   serially**, before submitting:
   ```bash
   .venv/bin/python scripts/finetune_detector.py --prep-only --mix mot20
   .venv/bin/python scripts/finetune_detector.py --prep-only --mix mot20_carla
   ```

**Never** raise a wall cap, the `%8` concurrency, or the 40 h ceiling to make units fit.
A unit killed at its cap is reported as a missing dose-response point (frozen doc s4).

## If you want to abandon Stage 1 entirely

That is a legitimate outcome and costs nothing already banked. Stage 0 (D55) stands on its
own and is the headline detector finding: perfect visible detections take the ceiling
0.571 → 0.970 and end-to-end retention 0.345 → 0.786. The tracker-side residual (0.810
assoc even at perfect detection) also stands. Both are already written into README and
`context.md`. Stage 1 only sharpens *how much* of that headroom a realistic detector
captures.

## Provenance note

The cluster repo is **bundle 8421645 + patch 4c97f1d + the Stage-1 files** (see `READY.md`
on the cluster for the sha256 table). It is not a pristine extraction of the shipped
bundle. Recorded deliberately; the bundle manifest itself is untouched and still verifies.
