# blockers — 100_occlusion_mot

> `refactored_method.md` §4: **there is no `AskUserQuestion`.** A blocker is a row here, and then
> **work continues on everything not blocked.** The loop never stops to ask.
>
> A row is only for the five things that actually require the operator — force-push, deleting
> repos/branches, making a repo public, spending money, wiping an environment — or for a genuine
> external dependency (dataset access, hardware, a threshold change). Ambiguity is **not** a
> blocker: implement the reading you can defend and state the assumption.

Every row carries a **recommendation**. A row that presents options without one is asking a
question with extra steps.

Created 2026-08-09 by the infra audit: this project's three agents were routed to
`reports/blockers.md` by `ws/CLAUDE.md` and by their own definitions, and the file did not exist.

| date | task | what is blocked | what was tried | what is needed | recommendation | state |
|---|---|---|---|---|---|---|
| 2026-08-10 | D26 / PERUN sweep | **The 25-unit PERUN sweep**, and with it the G2a-floor margin that D46 deferred to PERUN-scale. Nothing else — the whole path below it is done. | Everything short of submission. D47 completed PERUN-readiness and the full dry run passed end to end from the *unpacked bundle*: 10.94 GB tar → sha256 verified → unpack 8m55s → smoke unit `B:full:0` exit 0 in 5m13s. Wheelhouse resolves all 70 lock entries offline for linux-cp311. D48 closed the budget at 25 units / 35.25–39.25 H200-h against a 40 h ceiling — low **and** high fit. | **HPC access** (operator; possible spend). HPC day is then: fill 2 values → transfer → `sbatch`. `HPC_RUNBOOK.md`, under a page. | **(a) Submit jointly with 101's 5 H200-h pilot calibration in one request.** The pilot de-risks 101's 6–10× band assumption before its 350 h ask, and a single request reads better than two. | open |

## Resolved

Keep them. A resolved row is the record of who decided what.

| date | task | what was blocked | how it resolved |
|---|---|---|---|
| 2026-08-09 | D18 / val event | The pre-registered one-shot val event, and the D35 sequencing call (val before or after PERUN). | **Both discharged the same day.** D45: val executed under `val_manifest.md` frozen at `f2e0102`, user-approved, pre-flight green (ckpt sha256 ×2, MOT17 manifest `49c218a3` == D13, 240 tests, CUDA ok), R1–R6 run exactly with no deviations and no re-runs. D46: the G2a-floor miss (0.440 vs 0.58, ~2.4σ) **accepted by the user and reported rather than tuned away**, margin deferred to PERUN-scale per the pre-registration. Sequencing answered by the event itself — val was taken first. |

---

**Correction, 2026-08-10.** This file read *"(none open)"* while `status.txt` ended on
`blockers: PERUN access (user)` — flagged by the pre-HPC audit (`ws/reports/prehpc_audit_2026-08-09.md`
§7) and fixed here. That audit's own row 1 for this project (*"the pre-registered val event —
G1/G2 have no results at all"*) is **stale**: it was inferred from `check_gates.py` reporting
G1/G2 PENDING and from `results/tracker_ours.json` / `results/hidden_state.json` not existing on
disk, but D45 had already executed the val event that same day and G1 now **PASSES** (HOTA 51.07 /
IDF1 60.09 / MOTA 45.28). G2 **FAILS** on the G2a-floor, knowingly and on the record. The missing
result JSONs are a gate-checker path question for whoever next touches 100 — **not** evidence that
the val never ran; `results/val/val_report.md` is the artifact.
