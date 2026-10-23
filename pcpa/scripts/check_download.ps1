<#
.SYNOPSIS
    Report Boreas download progress and say plainly whether it has finished.

.DESCRIPTION
    The batch downloader announces, for every sequence, how many files it is
    about to fetch. Those numbers are the denominator, and they are already
    sitting in the download log, so this script reads them from there and
    counts what is on disk. No network access, no re-listing of the bucket.

    It also looks for truncated frames, and that check is not optional: the
    downloader writes with `curl -C -`, which appends to whatever is already
    present. A file interrupted halfway keeps its name and its place in the
    file count, so counting files alone can declare an incomplete download
    complete. File size is what catches it.

.EXAMPLE
    .\scripts\check_download.ps1
    .\scripts\check_download.ps1 -Sequence boreas-2021-06-03-16-00
#>
[CmdletBinding()]
param(
    [string]$Log = "D:\boreas_download.log",
    [string]$Dest,
    [string]$Sequence,
    [int]$Stride = 20
)

$ErrorActionPreference = "Stop"

# Resolve the destination from the project configuration instead of hard-coding
# it. The real path contains non-ASCII characters, and a literal here would
# depend on how PowerShell guesses this file's encoding (the bug that once
# produced a mojibake directory under D:).
if (-not $Dest) {
    $config = Join-Path $PSScriptRoot "..\config\default.yaml"
    if (-not (Test-Path $config)) { throw "missing $config" }
    $match = Select-String -Path $config -Pattern '^\s*data_raw:\s*(\S+)' |
        Select-Object -First 1
    if (-not $match) { throw "data_raw is not set in $config" }
    $raw = $match.Matches[0].Groups[1].Value.Trim().Replace('/', '\')
    $Dest = Join-Path $raw "boreas"
}

# Below these sizes a frame cannot be a whole recording. Sizes in this dataset
# are roughly LiDAR 4.9 MB, camera 2.6 MB, radar 0.4 MB per frame, so the floors
# sit far below the truth and only fire on a genuinely cut-off file.
$floor = @{ lidar = 2MB; camera = 0.5MB; radar = 0.05MB }
$frameModalities = @($floor.Keys)

# ---- what the downloader said it would fetch --------------------------
$plan = @{}          # sequence -> @{ modality = count }
$order = New-Object System.Collections.ArrayList
$reported = @{}      # sequence -> the summary line printed when it finished
$current = $null

if (Test-Path $Log) {
    foreach ($line in (Get-Content $Log -Encoding UTF8)) {
        if ($line -match '^sequence\s*:\s*(\S+)\s*$') {
            $current = $Matches[1]
            if (-not $plan.ContainsKey($current)) { [void]$order.Add($current) }
            $plan[$current] = @{}
        }
        elseif ($current -and
                $line -match '^\s*([A-Za-z_]+)\s+\d+ keys available, taking (\d+)\s*$') {
            $plan[$current][$Matches[1]] = [int]$Matches[2]
        }
        elseif ($current -and
                $line -match '^downloaded\s+(\d+)\s+file\(s\),\s*([\d.,]+)\s*MB,\s*(\d+)\s+failure') {
            $reported[$current] = [pscustomobject]@{
                files = [int]$Matches[1]; mb = $Matches[2]; failures = [int]$Matches[3]
            }
        }
    }
} else {
    Write-Warning "no log at $Log --- falling back to whatever is on disk"
}


# ---- what is actually there ------------------------------------------
# Is the downloader still alive? A partially written frame is entirely expected
# while curl is running, so this decides whether a small file is a fault or
# merely in flight.
$curl = @(Get-Process curl -ErrorAction SilentlyContinue).Count
$startedAt = if (Test-Path $Log) { (Get-Item $Log).CreationTime } else { $null }

$onDisk = @(Get-ChildItem $Dest -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { $_.Name })
$names = if ($Sequence) { @($Sequence) }
         else { @($order) + $onDisk | Select-Object -Unique | Sort-Object }

$rows = @()
$wantTotal = 0; $haveTotal = 0; $bytesTotal = 0.0; $truncTotal = 0
$planHave = 0; $planBytes = 0.0    # only sequences that have a plan, so that
                                   # stray files from earlier runs cannot
                                   # distort the average file size
foreach ($name in $names) {
    $dir = Join-Path $Dest $name
    $want = 0
    if ($plan.ContainsKey($name)) {
        foreach ($value in $plan[$name].Values) { $want += $value }
    }

    $files = @(Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue)
    $have = $files.Count
    $bytes = 0.0
    if ($have) { $bytes = [double]($files | Measure-Object -Property Length -Sum).Sum }

    $truncated = @()
    foreach ($mod in $frameModalities) {
        $modDir = Join-Path $dir $mod
        if (-not (Test-Path $modDir)) { continue }
        $truncated += @(Get-ChildItem $modDir -File -ErrorAction SilentlyContinue |
                        Where-Object { $_.Length -lt $floor[$mod] })
    }

    # A file count that is met is not yet a finished sequence: frames being
    # written right now are present, counted, and empty.
    $status = if ($want -eq 0) { "queued" }
              elseif ($have -ge $want -and $truncated.Count -eq 0) { "COMPLETE" }
              elseif ($truncated.Count -and -not $curl) { "TRUNCATED" }
              else { "downloading" }

    $rows += [pscustomobject]@{
        sequence  = $name
        have      = $have
        want      = $want
        percent   = if ($want) { 100.0 * $have / $want } else { [double]::NaN }
        gb        = $bytes / 1GB
        truncated = $truncated.Count
        status    = $status
    }
    $wantTotal += $want; $haveTotal += $have
    $bytesTotal += $bytes; $truncTotal += $truncated.Count
    if ($want -gt 0) { $planHave += $have; $planBytes += $bytes }
}

# ---- report -----------------------------------------------------------
$logAge = if (Test-Path $Log) { (Get-Date) - (Get-Item $Log).LastWriteTime } else { $null }

Write-Host ""
Write-Host "=== Boreas download status ==="
Write-Host "destination : $Dest"
if ($logAge) {
    Write-Host ("log         : {0}   (written {1:N1} min ago)" -f $Log, $logAge.TotalMinutes)
} else {
    Write-Host ("log         : {0}   (missing)" -f $Log)
}
if ($curl) {
    Write-Host "downloader  : RUNNING ($curl curl process(es))"
} else {
    Write-Host "downloader  : not running"
}
Write-Host ""
Write-Host ("{0,-30} {1,7} {2,7} {3,7} {4,10}  {5}" -f
            "sequence", "have", "want", "done", "size", "status")
Write-Host ("-" * 84)
foreach ($row in $rows) {
    # Clamp the percentage: an earlier, larger plan can leave more files on disk
    # than this plan asks for, and 113% in a progress column reads as a bug.
    $percent = if ([double]::IsNaN($row.percent)) { [double]::NaN }
               else { [Math]::Min(100.0, $row.percent) }
    $pct = if ([double]::IsNaN($percent)) { "  -  " } else { "{0,5:N1}%" -f $percent }
    Write-Host ("{0,-30} {1,7} {2,7} {3,7} {4,7:N2} GB  {5}" -f
                $row.sequence, $row.have, $row.want, $pct, $row.gb, $row.status)
}
Write-Host ("-" * 84)
if ($wantTotal) {
    Write-Host ("{0,-30} {1,7} {2,7} {3,7} {4,7:N2} GB" -f
                "planned total", $planHave, $wantTotal,
                ("{0,5:N1}%" -f [Math]::Min(100.0, 100.0 * $planHave / $wantTotal)),
                ($planBytes / 1GB))
}
if ($haveTotal -gt $planHave) {
    Write-Host ("{0,-30} {1,7} {2,7} {3,7} {4,7:N2} GB  {5}" -f
                "outside the plan", ($haveTotal - $planHave), "", "", 
                (($bytesTotal - $planBytes) / 1GB), "kept from earlier runs")
}

# A rate and an ETA, because "5.2%" says nothing about whether this will finish
# tonight or next week. The rate averages the whole campaign, including any
# pause, so it is a floor rather than a prediction.
if ($startedAt -and $planBytes -gt 0) {
    $elapsed = ((Get-Date) - $startedAt).TotalSeconds
    if ($elapsed -gt 60) {
        $rate = $planBytes / $elapsed
        Write-Host ("{0,-30} rate {1:N2} MB/s average over {2:N1} h" -f
                    "", ($rate / 1MB), ($elapsed / 3600))
        if ($wantTotal -gt $planHave -and $planHave -gt 0 -and $rate -gt 0) {
            $perFile = $planBytes / $planHave
            $leftBytes = ($wantTotal - $planHave) * $perFile
            Write-Host ("{0,-30} est. {1:N0} MB left, about {2:N1} h at this rate" -f
                        "", ($leftBytes / 1MB), ($leftBytes / $rate / 3600))
        }
    }
}

if ($curl -and $truncTotal) {
    Write-Host ("{0,-30} {1} frame(s) in flight --- curl is appending to them" -f "", $truncTotal)
}

if ($reported.Count) {
    Write-Host ""
    Write-Host "sequences the downloader reported as finished:"
    foreach ($name in $reported.Keys) {
        $info = $reported[$name]
        Write-Host ("  {0,-30} {1} file(s), {2} MB, {3} failure(s)" -f
                    $name, $info.files, $info.mb, $info.failures)
    }
}

if ($truncTotal -and -not $curl) {
    Write-Host ""
    Write-Warning "$truncTotal frame file(s) are far smaller than a real frame."
    Write-Host "  Re-running the batch downloader resumes them in place (curl -C -)."
}

# ---- verdict ----------------------------------------------------------
Write-Host ""
$pending = @($rows | Where-Object { $_.status -ne "COMPLETE" })
$bundle = @($onDisk | Where-Object { $_ -like "*objects*" })
if (-not $pending.Count -and $bundle.Count -and -not $Sequence) {
    Write-Host "VERDICT: every sequence and the annotation bundle are complete."
} elseif (-not $pending.Count -and -not $Sequence) {
    Write-Host "VERDICT: every planned sequence is complete."
    Write-Host "         boreas-objects-v1 (the labels) has not been fetched yet."
} elseif ($curl) {
    Write-Host "VERDICT: still downloading --- there is nothing to do but wait."
} else {
    Write-Host "VERDICT: INCOMPLETE and nothing is running."
    Write-Host "         Resume exactly where it stopped with:"
    Write-Host "           .\scripts\download_boreas_batch.ps1"
}
Write-Host ""
