#!/usr/bin/env pwsh
# Autonomous Phase 2 pipeline: download -> verify -> cache detections -> run both trackers -> check G0.
# Designed to run as a detached background process (Start-Process ... -WindowStyle Hidden).
# Writes logs/pipeline.log. Writes logs/pipeline_done.sentinel on success.

$ROOT = Split-Path $PSScriptRoot -Parent
$PYTHON = "$ROOT\.venv\Scripts\python.exe"
$LOG = "$ROOT\logs\pipeline.log"

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path "$ROOT\logs" | Out-Null
Log "=== Phase 2 pipeline started ==="

# Step 1: Download MOT17 (resumable — skips already-downloaded files)
Log "STEP 1: Download MOT17 from HF mirror"
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
$repo = "ling1016/MOT17"
& $PYTHON "$ROOT\scripts\download_mot17_hf.py" --repo $repo 2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) {
    Log "ERROR: download failed (exit $LASTEXITCODE) — trying fallback Morrison1025/MOT17"
    & $PYTHON "$ROOT\scripts\download_mot17_hf.py" --repo "Morrison1025/MOT17" 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) { Log "FATAL: both mirrors failed"; exit 1 }
    $repo = "Morrison1025/MOT17"
}
Log "STEP 1 done"

# Step 2: Verify structure + record hashes
Log "STEP 2: Verify MOT17 structure"
& $PYTHON "$ROOT\scripts\verify_mot17.py" --source "hf:$repo" 2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) { Log "ERROR: verification failed"; exit 1 }
Log "STEP 2 done"

# Step 3: Cache detections (yolo11x, FRCNN sequences only, cuda auto-select)
Log "STEP 3: Cache detections"
& $PYTHON "$ROOT\scripts\cache_detections.py" 2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) { Log "ERROR: cache_detections failed"; exit 1 }
Log "STEP 3 done"

# Step 4: Run our ByteTracker
Log "STEP 4: Run our ByteTracker"
& $PYTHON "$ROOT\scripts\run_baseline.py" --tracker ours 2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) { Log "ERROR: run_baseline ours failed"; exit 1 }
Log "STEP 4 done"

# Step 5: Run reference ByteTracker (supervision)
Log "STEP 5: Run reference ByteTracker"
& $PYTHON "$ROOT\scripts\run_baseline.py" --tracker reference 2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) { Log "ERROR: run_baseline reference failed"; exit 1 }
Log "STEP 5 done"

Log "=== All pipeline steps complete. Results in results/baseline_ours.json + baseline_reference.json ==="
# Sentinel for next runloop iteration
Set-Content -Path "$ROOT\logs\pipeline_done.sentinel" -Value (Get-Date -Format "o") -Encoding UTF8
Log "Sentinel written: logs/pipeline_done.sentinel"
