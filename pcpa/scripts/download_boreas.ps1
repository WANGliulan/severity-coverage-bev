<#
.SYNOPSIS
    Download a subset of the Boreas autonomous-driving dataset.

.DESCRIPTION
    The Boreas bucket is served as AWS open data and is readable ANONYMOUSLY
    over plain HTTPS, so no AWS account, no AWS CLI and no approval are
    needed. This script lists the keys of one sequence through the S3 REST
    API and downloads only what you ask for --- important, because a full
    sequence is tens of GB.

    Verified against:
      https://boreas.s3.amazonaws.com/?list-type=2&prefix=boreas-2020-11-26-13-58/calib/

    Sensors in this dataset: 128-beam Velodyne Alpha Prime lidar, 5 MP
    Blackfly camera, 360-degree Navtech radar, wheel encoder (DMI) and
    post-processed Applanix POS LV GNSS/INS.

.PARAMETER Sequence
    Sequence name, e.g. boreas-2020-11-26-13-58. Use -List to see options.

.PARAMETER Modalities
    Comma-separated list drawn from: calib, applanix, lidar, radar, camera.
    With -File, pass it as one token, e.g. -Modalities "calib,applanix,lidar".

.PARAMETER MaxFrames
    How many sensor frames to fetch per frame-based modality (lidar, radar,
    camera). calib and applanix are always fetched in full (they are small
    and every frame depends on them).

.PARAMETER CsvFrames
    For applanix, optionally limit the per-frame CSV files too.

.PARAMETER Parallel
    How many files to fetch at once (default 1, i.e. one at a time). curl runs
    the transfers concurrently itself, in invocations of at most 25 files, which
    keeps each command line well under the Windows limit.

    This is usually the single largest speed-up available: the objects are
    small (LiDAR 4.9 MB, camera 2.6 MB, radar 0.4 MB) and there are thousands
    of them, so a serial loop spends most of a high-latency link waiting on TLS
    handshakes and request round trips rather than transferring bytes.

    Each file still uses `-C -`, so re-running to resume remains safe.

.EXAMPLE
    .\scripts\download_boreas.ps1 -List
    .\scripts\download_boreas.ps1 -Sequence boreas-2020-11-26-13-58 -Modalities calib,applanix,lidar,radar -MaxFrames 20
    .\scripts\download_boreas.ps1 -Sequence boreas-2021-06-03-16-00 -Modalities calib,applanix,lidar,radar,camera -Stride 20 -Parallel 6
#>
[CmdletBinding()]
param(
    [string]$Bucket = "boreas",
    [string]$Sequence = "boreas-2020-11-26-13-58",
    [string]$Modalities = "calib,applanix",
    [int]$MaxFrames = 3,
    [int]$Stride = 1,
    [int]$Parallel = 1,
    [switch]$List,
    [switch]$AllApplanix,
    [string]$Dest
)

$ErrorActionPreference = "Stop"

# $PSScriptRoot is not set while parameter defaults are being evaluated, so
# the default destination is resolved here instead of in the param block.
if (-not $Dest) { $Dest = Join-Path $PSScriptRoot "..\data\raw\boreas" }

# Accept "calib,applanix,lidar" as a single token --- which is how -File
# passes the argument --- as well as a genuine array.
$modalityList = @($Modalities -split ',' | ForEach-Object { $_.Trim().ToLower() } |
    Where-Object { $_ })

$endpoint = "https://$Bucket.s3.amazonaws.com/"

function Get-S3Keys {
    param([string]$Prefix, [int]$Max = 100000)
    $keys = @()
    $token = $null
    do {
        $url = "$endpoint`?list-type=2&prefix=$Prefix&max-keys=1000"
        if ($token) { $url += "&continuation-token=$([uri]::EscapeDataString($token))" }
        $xml = & curl.exe -s --max-time 60 $url
        if ($LASTEXITCODE -ne 0) { throw "listing failed for prefix '$Prefix'" }
        foreach ($m in [regex]::Matches($xml, '<Key>(.*?)</Key>')) {
            $keys += $m.Groups[1].Value
            if ($keys.Count -ge $Max) { return $keys }
        }
        $t = [regex]::Match($xml, '<NextContinuationToken>(.*?)</NextContinuationToken>')
        $token = if ($t.Success) { $t.Groups[1].Value } else { $null }
    } while ($token)
    return $keys
}

# Keys are listed once per prefix and remembered, because the frame-modality
# selection below needs the LiDAR, camera and radar listings before it can
# decide anything, and listing 20k camera keys twice per sequence is slow.
$keyCache = @{}
function Get-Keys {
    param([string]$Prefix)
    if (-not $keyCache.ContainsKey($Prefix)) {
        $keyCache[$Prefix] = @(Get-S3Keys -Prefix $Prefix)
    }
    return , $keyCache[$Prefix]
}

function Get-KeyTimestamp {
    # Microsecond timestamp carried in a Boreas frame name, or -1 if absent.
    param([string]$Key)
    $value = 0L
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($Key)
    if ([long]::TryParse($stem, [ref]$value)) { return $value }
    return -1L
}

function Get-NearestKey {
    """The key whose timestamp is closest to Target, or $null."""
    param([string[]]$Keys, [long[]]$Stamps, [long]$Target)
    if ($Stamps.Length -eq 0) { return $null }
    # Timestamps are equal-length digit strings, so key order is time order and
    # a binary search is valid.
    $index = [Array]::BinarySearch($Stamps, $Target)
    if ($index -lt 0) { $index = -$index - 1 }
    $best = $null
    $bestDelta = [long]::MaxValue
    foreach ($j in @([Math]::Max(0, $index - 1), [Math]::Min($Stamps.Length - 1, $index))) {
        $delta = [Math]::Abs($Stamps[$j] - $Target)
        if ($delta -lt $bestDelta) { $bestDelta = $delta; $best = $Keys[$j] }
    }
    return $best
}

function Get-BoreasFiles {
    [CmdletBinding()]
    [OutputType([array])]
    param()

    $files = @()

    # A download that samples every sensor by its own key index does NOT produce
    # simultaneous frames. LiDAR runs at ~9.6 Hz, the camera at ~20 Hz and the
    # radar at 4 Hz, so the k-th sample of each drifts apart by about a second
    # per step: measured on boreas-2020-11-26-13-58 with -Stride 20, one of the
    # 60 LiDAR samples had a camera frame within 60 ms, and the last pair was a
    # minute apart. Frames that far apart cannot be fused --- a camera image from
    # a minute ago actively misleads a bird's-eye view --- so they are not worth
    # the bandwidth either.
    #
    # When LiDAR is requested together with another frame modality, the LiDAR
    # sample is therefore chosen first and the camera and radar frames nearest
    # those timestamps are taken instead.
    $matched = @{}
    $align = ($modalityList.Count -gt 1) -and ($modalityList -contains "lidar")
    if ($align -and $MaxFrames -le 0) {
        Write-Warning "-MaxFrames 0 with alignment would match every LiDAR frame; alignment skipped"
        $align = $false
    }
    $lidarKeys = @()
    if ($align) {
        $lidarKeys = @((Get-Keys "$Sequence/lidar/") | Where-Object { $_ -match '\.bin$' })
        $align = $lidarKeys.Count -gt 0
    }
    if ($align) {
        $step = [Math]::Max(1, $Stride)
        $limit = $MaxFrames
        $sample = New-Object System.Collections.ArrayList
        for ($i = 0; $i -lt $lidarKeys.Count -and $sample.Count -lt $limit; $i++) {
            if ($i % $step -eq 0) { [void]$sample.Add($lidarKeys[$i]) }
        }
        foreach ($mod in @("camera", "radar")) {
            if ($modalityList -notcontains $mod) { continue }
            $candidates = @((Get-Keys "$Sequence/$mod/") |
                Where-Object { $_ -match "^$Sequence/$mod/\d+\.png$" })
            if ($candidates.Count -eq 0) { continue }
            $stamps = [long[]]@($candidates | ForEach-Object { Get-KeyTimestamp $_ })
            $keep = @{}
            foreach ($key in $sample) {
                $hit = Get-NearestKey -Keys $candidates -Stamps $stamps -Target (Get-KeyTimestamp $key)
                if ($hit) { $keep[$hit] = $true }
            }
            $matched[$mod] = $keep
            Write-Host ("  {0,-18} aligned to the LiDAR sample: {1} frame(s)" -f $mod, $keep.Count)
        }
    }

    foreach ($mod in $modalityList) {
        $prefix = "$Sequence/$mod/"
        $keys = @(Get-Keys $prefix)
        if ($keys.Count -eq 0) { Write-Warning "no keys under $prefix"; continue }
        $pool = $keys          # the keys this modality may actually draw from

        if ($mod -in @("calib", "labels_detection", "labels_tracking") -or
            ($mod -eq "applanix" -and $AllApplanix)) {
            # small and needed in full: calibration, labels, all of applanix
            $files += $keys
        }
        elseif ($mod -eq "applanix") {
            # The pose CSVs are essential (odometry, load transfer, session
            # alignment) and small; dmi.csv carries the wheel encoder.
            # gps_post_process.csv and imu_raw.csv are tens of MB each and are
            # fetched only on request.
            $essential = $keys | Where-Object { $_ -match '(_poses\.csv|/dmi\.csv|_times\.txt|seqtimes\.txt)$' }
            $files += ($essential | Select-Object -First 16)
            if ($AllApplanix) {
                $files += ($keys | Where-Object { $_ -match '(gps_post_process|imu_raw)' })
            }
        }
        else {
            # Frame modalities. -MaxFrames 0 means "all", and -Stride keeps
            # every n-th frame, which is what makes a download feasible at all:
            # one full sequence measures about 46 GB of LiDAR (9984 frames at
            # 4.65 MB), 57 GB of camera (20624 at 2.78 MB) and 5 GB of radar
            # (12426 at 0.39 MB), plus 1.2 GB of video, and the bucket holds
            # 121 sequences --- close to 13 TB in total.
            # The radar prefix hides three products per sweep. radar/<t>.png is
            # the raw polar sweep; radar/cart/<t>.png and radar/mask/<t>.png are
            # a cartesian re-projection and a moving-object annotation of the
            # same sweep. Those keys sort after the raw ones, so a stride long
            # enough to reach past the raw block would mix re-projections and
            # annotation masks into the sweep stream, and the polar rasteriser
            # downstream would accept them without complaining. Draw raw only.
            if ($mod -eq "radar") {
                $pool = @($keys | Where-Object { $_ -match "^$Sequence/radar/\d+\.png$" })
            }
            $limit = if ($MaxFrames -gt 0) { $MaxFrames } else { [int]::MaxValue }
            if ($matched.ContainsKey($mod)) {
                # Already exactly the frames that pair with the LiDAR sample, so
                # sampling them again by index would break the pairing.
                $selected = @($pool | Where-Object { $matched[$mod].ContainsKey($_) })
            }
            else {
                $selected = New-Object System.Collections.ArrayList
                for ($i = 0; $i -lt $pool.Count -and $selected.Count -lt $limit; $i++) {
                    if ($Stride -le 1 -or ($i % $Stride) -eq 0) { [void]$selected.Add($pool[$i]) }
                }
            }
            $files += $selected
        }
        Write-Host ("  {0,-18} {1,6} keys available, taking {2}" -f $mod, $pool.Count,
                    (($files | Where-Object { $_ -like "$prefix*" }).Count))
    }

    # Extras that live next to the sequences rather than inside them.
    $extras = switch ($Sequence) {
        "boreas-objects-v1" { @("boreas-objects-v1/readme.txt", "boreas-objects-v1/test_times.txt") }
        default { @() }
    }
    foreach ($extra in $extras) {
        if (@(Get-S3Keys -Prefix $extra).Count -gt 0) { $files += $extra }
    }
    return @($files | Select-Object -Unique)
}

if ($List) {
    Write-Host "sequences available in s3://$Bucket/"
    # delimiter-based listing returns CommonPrefixes, i.e. the top-level
    # 'folders', which is exactly one per sequence
    $xml = & curl.exe -s --max-time 60 "$endpoint`?list-type=2&delimiter=%2F"
    $found = [regex]::Matches($xml, '<Prefix>(.*?)</Prefix>') |
        ForEach-Object { $_.Groups[1].Value.TrimEnd('/') }
    if ($found) { $found } else { Write-Warning "no sequences returned" }
    return
}

$target = Join-Path $Dest $Sequence
Write-Host "sequence : $Sequence"
Write-Host "modalities: $($modalityList -join ', ')"
Write-Host "dest     : $target"
Write-Host ""

$files = Get-BoreasFiles
Write-Host ""
Write-Host "downloading $($files.Count) file(s) (resumable: re-run to continue)"

# Resume, then verify. `curl -C -` continues a partial file in place and exits 0
# when the file is already complete, so re-running this script is always safe.
# Whether a file really arrived is decided afterwards by looking at the
# directory, which is a stronger check than trusting an exit code.
$plan = @()
foreach ($key in $files) {
    $plan += @{ key = $key
                out = Join-Path $target ($key.Substring($Sequence.Length + 1).Replace('/', '\')) }
}

if ($Parallel -le 1) {
    $attempted = 0
    foreach ($item in $plan) {
        New-Item -ItemType Directory -Force -Path (Split-Path $item.out) | Out-Null
        & curl.exe -s -L -C - --retry 5 --retry-delay 3 --retry-all-errors `
            --max-time 900 -o $item.out "$endpoint$($item.key)"
        $attempted++
        if ($attempted % 25 -eq 0) {
            Write-Host "  $attempted/$($plan.Count) file(s) attempted"
        }
    }
}

else {
    # A serial loop pays a fresh TLS handshake and a request round trip per file,
    # and there are thousands of small files here, so a high-latency link sits
    # idle most of the time. curl can run transfers concurrently itself, which
    # needs none of the PowerShell job infrastructure (Start-Job fails outright
    # in some sandboxes).
    #
    # The batch size is not about parallelism but about the Windows command line
    # limit: every file contributes an -o path and a URL, so each invocation is
    # kept far below 32767 characters.
    $batchSize = 25
    Write-Host "  $Parallel concurrent transfer(s), up to $batchSize file(s) per curl"
    Write-Host ""
    $attempted = 0
    for ($start = 0; $start -lt $plan.Count; $start += $batchSize) {
        $stop = [Math]::Min($start + $batchSize - 1, $plan.Count - 1)
        $curlArgs = @('-s', '-L', '-C', '-', '--retry', '5', '--retry-delay', '3',
                      '--retry-all-errors', '--max-time', '900',
                      '--parallel', '--parallel-max', "$Parallel")
        foreach ($item in $plan[$start..$stop]) {
            New-Item -ItemType Directory -Force -Path (Split-Path $item.out) | Out-Null
            $curlArgs += @('-o', $item.out, "$endpoint$($item.key)")
        }
        & curl.exe @curlArgs
        $attempted += ($stop - $start + 1)
        $present = @(Get-ChildItem $target -Recurse -File -ErrorAction SilentlyContinue)
        $size = 0.0
        if ($present.Count) {
            $size = [double]($present | Measure-Object -Property Length -Sum).Sum
        }
        Write-Host ("  {0,5}/{1} attempted, {2,5} on disk, {3,9:N0} MB" -f
                    $attempted, $plan.Count, $present.Count, ($size / 1MB))
    }
}

# A file counts as downloaded only if it is on disk afterwards.
$ok = 0; $fail = 0; $bytes = 0
foreach ($item in $plan) {
    if (Test-Path $item.out) {
        $ok++; $bytes += (Get-Item $item.out).Length
    }
    else { $fail++; Write-Warning "missing after download: $($item.key)" }
}

Write-Host ""
Write-Host ("downloaded {0} file(s), {1:N1} MB, {2} failure(s)" -f $ok, ($bytes / 1MB), $fail)
Write-Host "next: point config/default.yaml's datasets.boreas.root at $Dest"
