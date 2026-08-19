# READY — PERUN sweep (updated 2026-08-19 09:05, post-smoke, lever fired)

Smoke job **passed** (SLURM 77122, gpu04). The pre-registered detector lever
**fired** on its throughput read, so the grid is now **23 units, not 25**.

**Array 77150 SUBMITTED and running** (2026-08-19 09:26). A first attempt, 77126,
died instantly on a `timeout` format bug -- see *Cluster repo provenance* below.

---

## 1 — Smoke job (DONE)

Ran as job **77122** on `gpu04`, empty stderr. Results:

```
OK  NVIDIA H200  torch=2.13.0+cu126  cuda=12.6  capability=(9, 0)
SMOKE OK -- safe to submit results/sweep/perun_full/submit.sbatch
```

Both detector datasets prepped and cached (`data/cache/det_finetune_mot17dev`,
`data/cache/det_finetune_mot17dev_carla`), so array tasks never race on them.

**Measured throughput: yolo11s = 88 s/epoch** at `imgsz 960` (epoch 2; epoch 1 discarded
as warmup-contaminated). 100 epochs -> ~2.44 h, inside the 02:30 cap by ~2 s/epoch.

## 2 — Expected log path

```
results/sweep/perun_full/logs/smoke_perun_full_<JOBID>.out
results/sweep/perun_full/logs/smoke_perun_full_<JOBID>.err
```

which resolves through the symlink to Lustre:

```
/mnt/scratch/mafike202/omot/sweep/perun_full/logs/smoke_perun_full_<JOBID>.out
```

Watch it live:

```bash
tail -f results/sweep/perun_full/logs/smoke_perun_full_*.out
```

Queue state:

```bash
squeue -u $USER -o '%.10i %.20j %.9P %.8T %.10M %.6D %R'
```

## 3 — What the smoke job does, and what to read from it

Three stages, in order:

1. **GPU probe** — asserts `torch.cuda.is_available()` and that a real matmul lands on
   `cuda`. Expect a line like:
   `OK  NVIDIA H200  torch=2.13.0+cu126  cuda=12.6  capability=(9, 0)`
   If this fails the job stops there and **the array must not be submitted**.
2. **Detector dataset prep**, both mixes (`mot17dev`, `mot17dev_carla`), serialised here
   so the array's two units sharing a mix never race on the same directory.
3. **2-epoch `yolo11s` finetune** on `mot17dev_carla` — the real throughput read.

### The D47/D48 lever — FIRED 2026-08-19

Projected for yolo11m by GFLOPs ratio at `imgsz 960`: **154.2 / 48.9 = 3.15x**, so
88 s x 3.15 = **~277 s/epoch -> ~7.7 h per 100-epoch run** — roughly 3x the 02:30 hard
cap. Both yolo11m units would have been killed leaving no result JSON, spending ~5 h of
allocation for nothing.

So the pre-registered lever was applied (perun_sweep_v2.md **amendment 7**, D49):

```yaml
detector:
  models: [yolo11s]        # was [yolo11s, yolo11m]
```

The cap was **not** raised and the 40 h ceiling did **not** move.

**Standing risk, stated before the fact:** the two surviving yolo11s detector units sit
only ~2 s/epoch under the cap and are themselves marginal. If one or both die at 02:30,
the R6 selection rule (amendment 1) applies over whichever detector units completed and
the shortfall is reported in the selection — no cap change, no model substitution without
a new dated amendment.

## 2 — The array (23 units)

```bash
cd /mnt/project/perun26011488/omot/omot_hpc/repo
sbatch results/sweep/perun_full/submit.sbatch
```

**23 units** (18 embedder + 3 arm-A ImageNet-null + 2 detector), `--array=0-22%8`,
per-task `--time=02:30:00`, 8 concurrent (one full node).
Budget **32.25–34.25 H200-h against the 40 h ceiling** — worst case fits.

Monitoring one-liner:

```bash
watch -n 60 "squeue -u $USER -o '%.10i %.20j %.8T %.10M %R' | tail -20; ls results/sweep/perun_full/*.json 2>/dev/null | wc -l"
```

Done = **23** result JSONs in `results/sweep/perun_full/`. Then:

```bash
.venv/bin/python scripts/sweep_launcher.py --config configs/sweep/perun_full.yaml --mode local
```

runs no new units and writes `summary.json` (per-arm gate selection + every
pre-registered McNemar p-value).

If the array dies partway, just re-submit it — every finished unit exits immediately.

## 3 — Regenerating the scripts (only if you change the config)

```bash
.venv/bin/python scripts/sweep_launcher.py --config configs/sweep/perun_full.yaml --mode slurm
```

Any `STILL a placeholder` warning means the SLURM values got reverted — stop and fix.

---

## What was already verified

| Check | Result |
|---|---|
| `sha256sum -c` on the transferred 10.94 GB bundle | OK |
| `make_hpc_bundle.py verify` | **VERIFY OK** — 9 entries, 12.24 GB, 529,911 members |
| Pinned checkpoint re-verify after the move to Lustre | `reid_conv.pt` MATCH, `reid_proto.pt` MATCH |
| `hpc_bootstrap.sh` | **BOOTSTRAP OK** — glibc 2.34, Python 3.11.13, 72 wheels offline, headless opencv enforced, staged inputs OK |
| Runtime stack | torch 2.13.0+cu126, torchvision 0.28.0+cu126, numpy 1.26.4, opencv 4.11.0, ultralytics 8.4.104 |
| Placeholder scan on both sbatch scripts | clean |
| `sbatch --test-only` (both scripts) | both accepted |
| Smoke job 77122 | **PASSED** — H200 probe OK, both datasets prepped, 88 s/epoch |
| Queue | empty — nothing submitted |

Grid after amendment 7: **23 units, 32.25-34.25 H200-h vs the 40 h ceiling**.

SLURM values filled in `configs/sweep/perun_full.yaml`:
`partition: gpu_short`, `account: perun26011488`.

`gpu_short` (2-day limit) and `gpu_long` (4-day) span the same 26 nodes; the longest
unit wall is 02:30, so the short queue is correct and schedules sooner.

## Cluster repo provenance (IMPORTANT)

The cluster tree is **NOT** a pristine extraction of the shipped bundle. It is:

```
bundle 8421645  +  patch 4c97f1d  (3 files, hot-patched 2026-08-19 09:20)
```

`repo.tar` inside `omot_hpc_8421645.tar.gz` still hashes as `VERIFY OK` — the manifest
is untouched and remains valid for the bundle as shipped. These three files on the
cluster now differ from it, by deliberate operator action (user-approved, option (a)):

| File | sha256 (patch 4c97f1d) |
|---|---|
| `scripts/sweep_common.py` | `a2b4a6dc732eead34a2dc3340e6be191a8256a6dfc5c28c945bc2d139bf8df15` |
| `scripts/sweep_launcher.py` | `f68e7ca52e342037a35839c77a0a47c119bcdf095c2fe15a647a5dbb38e50838` |
| `tests/test_sweep_launcher.py` | `13cebad76cc94c5b6736a4b4c422dceaff4668fecd147e3f6547468079f1fc44` |

Verified byte-identical local vs post-scp (`sha256sum -c`, all three OK). The
pre-patch originals are preserved at `.pre_4c97f1d/` in the repo root:

| File | sha256 (bundle 8421645) |
|---|---|
| `sweep_common.py` | `b517851cf0fb5886adf74e5b598ec8d3b5d3342657db3b6b73777fe230cde6af` |
| `sweep_launcher.py` | `8110395b15cb5f2fcefdfb01b8032b2cde9cf964c8920376b39146c40eaf537e` |
| `test_sweep_launcher.py` | `276bc5c364cf7a0ed10e6fa69db1f4688cd7af0f850595135085dd588da726af` |

**Why:** array 77126 lost all 23 tasks in under a second to
`timeout: invalid time interval '02:10:00'`. coreutils `timeout` takes `NUMBER[smhd]`
and rejects SLURM's `HH:MM:SS`; the emitted `LIMITS` array reused the `#SBATCH --time`
string, so per-class wall enforcement could never have run. Zero result JSONs, zero
GPU-hours. Fix emits `2400s / 7800s / 9000s` — the same walls to the second — while
`--time` keeps `HH:MM:SS`.

It is a third file, not two, because the new regression test had to ship for it to be
runnable on the cluster tree. It was run there before regenerating: **passes**.

Three cluster test failures are environmental and pre-existing, not from the patch:
`test_shipped_perun_full_config_has_slurm_placeholders` (we filled the placeholders --
that is the HPC-day edit), `test_every_requirement_names_a_real_path` (wants the dev-box
`data/wheelhouse`; on the cluster it is `../wheelhouse`), and
`test_every_source_module_is_tracked_by_git` (the tree is a tar extraction, not a git repo).

## Storage layout

Bundle, repo and venv on **PROJECT**; all run outputs, checkpoints and logs on
**SCRATCH** (Lustre), reached through symlinks so every repo-relative path in the
sbatch scripts stays valid.

| Repo path | Backing store |
|---|---|
| `repo/`, `.venv/`, `../wheelhouse/` | `/mnt/project/perun26011488/omot/` (NFS) |
| `data/reid` → | `/mnt/scratch/mafike202/omot/reidroot/data/reid` (Lustre) |
| `data/models` → | `/mnt/scratch/mafike202/omot/models` |
| `results/sweep/perun_full` → | `/mnt/scratch/mafike202/omot/sweep/perun_full` |
| `.cache` → | `/mnt/scratch/mafike202/omot/cache` |

`data/MOT17` and `data/sim/carla_render` stay on PROJECT — whole frames, not 509k
small crops, so NFS is fine for them.

Note: `/mnt/project/mafike202` is **not** yours (owned by `haabke763`, mode `drwx------`,
writes denied). Your project storage is the group directory `/mnt/project/perun26011488`.

## Deviations from HPC_RUNBOOK.md (deliberate, both recorded)

1. **Payload tars extracted with GNU `tar`, not `make_hpc_bundle.py unpack`.**
   `unpack` uses `tarfile`'s `data` filter, which calls `realpath` on each destination
   and raises `OutsideDestinationError` for anything routed through a symlink to another
   filesystem — so it cannot place `data/reid` on Lustre. Integrity is unaffected: it is
   established by `verify`, which re-hashes all 9 tars against `MANIFEST.json` and ran
   first. `unpack` only chooses destinations. GNU tar is also far faster on 509k members.
2. **`results/sweep/perun_full` is symlinked, not `results/sweep`.** `repo.tar` ships a
   git-tracked `results/sweep/dryrun_local/`, so `results/sweep` already exists as a real
   directory and must stay one.
