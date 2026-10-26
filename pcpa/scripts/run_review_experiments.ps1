# Review-response experiments: the four checks a reviewer asked for.
#
#   .\.venv\Scripts\powershell -File scripts\run_review_experiments.ps1   (or run in place)
#
# Each run appends a log under ``runs/review`` and, on success, a record under
# ``data/manifest``; the manifest is what the result tables are read from, so the
# numbers this produces can be quoted without touching a hand-written table.
#
#   1. capacity match    a concatenated baseline with 264,073 parameters, i.e. the
#                        same budget as the 262,866-parameter gate, so a gain cannot
#                        be read as "the gate is simply bigger"
#   2. boundary gap      four sampled frames dropped either side of every drive's
#                        train/test boundary, to see how much of the result rests on
#                        adjacent frames
#   3. severe only       trained on s=4 alone: same sample budget as the mild regime,
#                        no diversity, to separate "coverage" from "more augmentations"
#   4. detection seeds   a third run per detection regime, so the 0.0026 difference
#                        can be quoted with a spread rather than a single pair
#
# Runtime, measured on this machine: a fusion run is ~2 min, a detection run ~8 min.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$logDir = Join-Path $root 'runs\review'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Study([string]$name, [string]$script, [string[]]$arguments) {
    $log = Join-Path $logDir "$name.log"
    Write-Host "== $name"
    & $python (Join-Path $root "scripts\$script") @arguments *> $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   FAILED, see $log"
        throw "$name failed with exit code $LASTEXITCODE"
    }
    Write-Host "   ok, log at runs/review/$name.log"
}

# 1. capacity-matched concatenated baseline (base 24 -> 264,073 parameters)
foreach ($seed in 0, 1) {
    Invoke-Study "capacity_mild_s$seed" 'run_fusion_study.py' @(
        '--base-channels', '24', '--train-levels', '0,1,2', '--seed', "$seed")
    Invoke-Study "capacity_full_s$seed" 'run_fusion_study.py' @(
        '--base-channels', '24', '--train-levels', '0,1,2,3,4', '--seed', "$seed")
}

# 2. boundary gap: the reported split, with 4 frames removed either side
Invoke-Study 'gap4_mild_s0' 'run_fusion_study.py' @(
    '--boundary-gap', '4', '--train-levels', '0,1,2', '--seed', '0')
Invoke-Study 'gap4_full_s0' 'run_fusion_study.py' @(
    '--boundary-gap', '4', '--train-levels', '0,1,2,3,4', '--seed', '0')

# 3. severe level only: no diversity, same sample budget
foreach ($seed in 0, 1) {
    Invoke-Study "severe_only_s$seed" 'run_fusion_study.py' @(
        '--train-levels', '4', '--seed', "$seed")
}

# 4. detection: a third seed for each regime
foreach ($levels in '0,1,2', '0,1,2,3,4') {
    $tag = ($levels -replace ',', '')
    Invoke-Study "detection_${tag}_s2" 'run_detection_study.py' @(
        '--train-levels', $levels, '--seed', '2')
}

Write-Host "all review experiments finished"
