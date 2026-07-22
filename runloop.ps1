# runloop.ps1 — autonomous outer loop for the occlusion-mot mission.
# Each iteration: check gates -> if not done, invoke `claude -p` with fresh context
# (mission.md/status.txt on disk + latest gate report inlined) -> log -> repeat.
# Stops when: all gates pass (exit 0), HARD_FAIL marker present (exit 2), or MaxIterations (exit 1).
param(
    [int]$MaxIterations = 20,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path logs | Out-Null
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

for ($i = 1; $i -le $MaxIterations; $i++) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $gateReport = "logs\gate_report_$stamp.txt"

    & $py scripts\check_gates.py --report $gateReport
    $gateExit = $LASTEXITCODE
    Write-Host "[runloop] iteration $i gate check exit=$gateExit"
    if ($gateExit -eq 0) { Write-Host "[runloop] ALL GATES PASS — mission complete."; exit 0 }
    if ($gateExit -eq 2) { Write-Host "[runloop] HARD_FAIL marker present — stopping."; exit 2 }

    if ($DryRun) { Write-Host "[runloop] DryRun: would invoke claude -p now."; exit 0 }

    $report = ""
    if (Test-Path $gateReport) { $report = Get-Content $gateReport -Raw }
    $prompt = @"
Autonomous runloop iteration $i for the occlusion-mot mission.
Read CLAUDE.md, mission.md, context.md, status.txt first. You are the single writer session.
Latest gate report:
$report
Do the highest-leverage work toward the current phase's gates. Delegate to coder/reviewer/eval-runner
subagents per CLAUDE.md; reviewer must approve diffs. Update status.txt, log decisions in context.md,
commit at checkpoints. Create a HARD_FAIL file in the repo root ONLY on unrecoverable failure.
Never wait for user input.
"@

    $log = "logs\iter_${i}_$stamp.log"
    Write-Host "[runloop] launching claude iteration $i -> $log"
    claude -p $prompt --permission-mode bypassPermissions 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[runloop] claude exited $LASTEXITCODE; continuing."
    }
}
Write-Host "[runloop] MaxIterations ($MaxIterations) reached without gate pass."
exit 1
