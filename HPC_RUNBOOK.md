# PERUN runbook — login day

Everything below assumes the bundle is already built on the dev box
(`scripts/make_hpc_bundle.py build`). **Exactly two values get filled in by hand**:
`slurm.partition` and `slurm.account`, in step 4. Nothing else is edited on the cluster.

### 1 — transfer

```bash
rsync -avP --partial dist/omot_hpc_<sha>.tar.gz dist/omot_hpc_<sha>.tar.gz.sha256 <user>@perun.tuke.sk:~/
```

On a flaky link, build with `--no-archive` and send the staged directory instead —
same contents, resumable per member, and step 2 drops the `sha256sum`/`tar` line:
`rsync -avP dist/omot_hpc/ <user>@perun.tuke.sk:~/omot_hpc/`

### 2 — unpack + verify

```bash
cd ~ && sha256sum -c omot_hpc_<sha>.tar.gz.sha256 && tar xzf omot_hpc_<sha>.tar.gz
cd omot_hpc && python3 make_hpc_bundle.py verify && python3 make_hpc_bundle.py unpack --dest .
```

`verify` must print `VERIFY OK`. Unpack creates `omot_hpc/repo/` and `omot_hpc/wheelhouse/`.
It writes ~530k files — if `$HOME` has an inode quota, unpack to scratch instead:
`python3 make_hpc_bundle.py unpack --dest $SCRATCH/omot`.

### 3 — environment (offline, from the bundled wheelhouse)

```bash
cd ~/omot_hpc/repo && module load Python/3.11 2>/dev/null; bash scripts/hpc_bootstrap.sh
```

Must end with `BOOTSTRAP OK`. It hard-fails on glibc < 2.28 or a non-3.11 interpreter.

### 4 — fill the two unknowns

```bash
sed -i 's|<FILL-PERUN-PARTITION>|YOUR_PARTITION|; s|<FILL-PERUN-ACCOUNT>|YOUR_ACCOUNT|' configs/sweep/perun_full.yaml
grep -E 'partition|account' configs/sweep/perun_full.yaml
```

### 5 — generate the job scripts

```bash
.venv/bin/python scripts/sweep_launcher.py --config configs/sweep/perun_full.yaml --mode slurm
```

Emits `results/sweep/perun_full/{submit_smoke.sbatch,submit.sbatch}` and prints the budget
line. Any `STILL a placeholder` warning means step 4 did not take — stop and fix it.

### 6 — submit (smoke first, array chained behind it)

```bash
cd ~/omot_hpc/repo
SMOKE=$(sbatch --parsable results/sweep/perun_full/submit_smoke.sbatch)
sbatch --dependency=afterok:$SMOKE results/sweep/perun_full/submit.sbatch
```

Submit **from the repo root** — the scripts check for this and refuse otherwise. The smoke
job probes the GPU, preps both detector datasets (serialised, so array tasks never race)
and runs a 2-epoch detector finetune. The array does not start unless it succeeds.

### 7 — monitor

```bash
squeue -u $USER -o '%.10i %.20j %.9P %.8T %.10M %.6D %R'
tail -f results/sweep/perun_full/logs/*.out
grep -l . results/sweep/perun_full/*.json | wc -l    # units finished (target: 25)
```

**Done** = 25 result JSONs in `results/sweep/perun_full/` (21 embedder + 4 detector), then:

```bash
.venv/bin/python scripts/sweep_launcher.py --config configs/sweep/perun_full.yaml --mode local
```

which runs no new units (all results present) and writes `summary.json` — per-arm gate
selection and every pre-registered McNemar p-value.

### 8 — pull results back

```bash
rsync -avP <user>@perun.tuke.sk:~/omot_hpc/repo/results/sweep/perun_full/ results/sweep/perun_full/
```

---

**If the array dies partway**, just re-submit it: every task exits immediately when its
result JSON already exists. No flags, no bookkeeping.

**Budget:** 25 units, 35.25–41.25 H200-h against a 40 h ceiling. Per-task wall limits are
enforced with `timeout` inside each job (embedder 02:10, ImageNet-null 00:40, detector
04:05). The high end sits above the ceiling — driven by the detector arm's 1.5–3 h/run
spread. If the smoke job's 2 epochs extrapolate past ~2 h/run, drop to one detector model
before submitting the array.
