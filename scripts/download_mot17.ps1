# Resumable MOT17 download + extraction. Safe to re-run; runloop iterations retry this
# until data/MOT17/train exists. motchallenge.net is intermittently unreachable.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path data | Out-Null

if (Test-Path "data\MOT17\train") {
    Write-Host "MOT17 already extracted."
    exit 0
}

$zip = "data\MOT17.zip"
curl.exe -L -C - --retry 8 --retry-delay 30 --connect-timeout 30 -o $zip https://motchallenge.net/data/MOT17.zip
if ($LASTEXITCODE -ne 0) {
    Write-Host "download failed (exit $LASTEXITCODE) — motchallenge.net may be down; re-run later."
    exit 1
}

Write-Host "extracting..."
Expand-Archive -Path $zip -DestinationPath data -Force
if (-not (Test-Path "data\MOT17\train")) {
    Write-Host "unexpected archive layout after extraction"; exit 1
}
Write-Host "MOT17 ready."
exit 0
