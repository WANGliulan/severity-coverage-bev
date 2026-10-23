<#
.SYNOPSIS
    Download a set of Boreas sequences in one go.

.DESCRIPTION
    Thin wrapper around download_boreas.ps1 that loops over several sequences,
    so a long download can be started once and left running. It is resumable:
    re-running skips nothing (curl -C - continues partial files) and finishes
    the ones that are missing.

    Frames are subsampled with -Stride and capped with -MaxFrames. That is not
    a compromise but a necessity: one full sequence measures about 110 GB
    (46 GB LiDAR, 57 GB camera, 5 GB radar, 1.2 GB video) and the bucket holds
    121 sequences, close to 13 TB --- which at the measured link speed would
    take months of continuous transfer.

    The defaults (every 20th frame, 60 frames per sensor) came out of measuring
    the link: it delivers 0.07-0.63 MB/s depending on the hour, so the original
    "all frames" plan would have taken days. 60 frames spanning about two
    minutes of driving per sequence is roughly 0.5 GB and is more than the
    ablation and the weather labels need. Raise -MaxFrames later to extend a
    sequence --- re-running only fetches what is new.

.EXAMPLE
    .\scripts\download_boreas_batch.ps1
    .\scripts\download_boreas_batch.ps1 -Sequences "boreas-2021-01-15-12-17" -Stride 10
    .\scripts\download_boreas_batch.ps1 -MaxFrames 200 -Parallel 8
#>
[CmdletBinding()]
param(
    [string]$Sequences = (
        "boreas-2020-11-26-13-58," +      # late autumn
        "boreas-2021-01-15-12-17," +      # deep winter, snow likely
        "boreas-2021-02-02-14-07," +      # winter
        "boreas-2021-03-23-12-43," +      # spring thaw, wet roads
        "boreas-2021-06-03-16-00," +      # summer, clear
        "boreas-2021-11-16-14-10"         # late autumn, rain likely
    ),
    [string]$Modalities = "calib,applanix,lidar,radar,camera",
    # 300 is the value that matters for multi-modal work. The downloader picks
    # the LiDAR sample first and then takes the camera and radar frames nearest
    # those timestamps, so one stride gives frames that actually coexist. At 300
    # the 34 LiDAR samples of a recording span nearly all 1035 s of it and cover
    # about 99% of the 7.9 km route, for roughly 280 MB per sequence. A stride of
    # 20 would spend 490 MB to cover 11% of the route --- and, without the
    # alignment, would pair each sweep with a camera frame from a minute away.
    [int]$Stride = 300,
    [int]$MaxFrames = 60,     # per sensor; 0 means "every frame", which is days of transfer
    [string]$Dest = "D:\数据集\boreas",
    [int]$Parallel = 6,
    [switch]$SkipObjectsBundle
)

$ErrorActionPreference = "Continue"
$downloader = Join-Path $PSScriptRoot "download_boreas.ps1"
if (-not (Test-Path $downloader)) { throw "missing $downloader" }

$list = $Sequences -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
Write-Host "destination : $Dest"
Write-Host "modalities  : $Modalities"
Write-Host "stride      : $Stride  (every $Stride-th frame; 10 Hz source)"
Write-Host "parallel    : $Parallel  (files fetched at once)"
Write-Host "sequences   : $($list.Count)"
foreach ($sequence in $list) { Write-Host "  $sequence" }
Write-Host ""

$index = 0
foreach ($sequence in $list) {
    $index++
    Write-Host ("=" * 70)
    Write-Host "[$index/$($list.Count)] $sequence"
    Write-Host ("=" * 70)
    & $downloader -Sequence $sequence -Modalities $Modalities -Stride $Stride `
        -MaxFrames $MaxFrames -Dest $Dest -Parallel $Parallel
}

if (-not $SkipObjectsBundle) {
    Write-Host ("=" * 70)
    Write-Host "[bundle] boreas-objects-v1  (detection and tracking labels)"
    Write-Host ("=" * 70)
    & $downloader -Sequence "boreas-objects-v1" `
        -Modalities "calib,applanix,labels_detection,labels_tracking" -Dest $Dest `
        -Parallel $Parallel
}

Write-Host ""
$total = (Get-ChildItem $Dest -Recurse -File -ErrorAction SilentlyContinue |
          Measure-Object -Property Length -Sum).Sum
Write-Host ("done. {0:N1} GB under {1}" -f ($total / 1GB), $Dest)
Write-Host "re-run this script at any time to fetch whatever is still missing."
