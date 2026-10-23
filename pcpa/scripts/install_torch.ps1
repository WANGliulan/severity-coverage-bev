<#
.SYNOPSIS
    Install torch + torchvision (CUDA 12.8) into the project venv.

.DESCRIPTION
    pip's own downloader is not resumable and can stall on a slow link,
    which costs the whole 2.7 GB. This script fetches the cu128 wheel with
    curl -C - (resume enabled), then asks pip to install that local file
    plus torchvision from the cu128 index.

    Re-running the script is safe: the download continues where it stopped.

.NOTES
    The RTX 5070 is Blackwell (sm_120), so ONLY cu128-or-newer builds
    contain a matching kernel. Bump -TorchVersion / -PyTag as needed.

.EXAMPLE
    .\scripts\install_torch.ps1
    .\scripts\install_torch.ps1 -TorchVersion 2.10.0
#>
[CmdletBinding()]
param(
    [string]$Python,
    [string]$TorchVersion = "2.11.0",
    [string]$PyTag = "cp313"
)

$ErrorActionPreference = "Stop"

# $PSScriptRoot is not set while parameter defaults are being evaluated.
if (-not $Python) { $Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe" }

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runs = Join-Path $root "runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null

if (-not (Test-Path $Python)) {
    throw "venv interpreter not found: $Python  (create it first with: py -3.13 -m venv .venv)"
}

$remoteName = "torch-$TorchVersion%2Bcu128-$PyTag-$PyTag-win_amd64.whl"
$url = "https://download.pytorch.org/whl/cu128/$remoteName"
$wheel = Join-Path $runs "torch-$TorchVersion+cu128-$PyTag-$PyTag-win_amd64.whl"

Write-Host "[1/4] downloading $url"
Write-Host "      -> $wheel"
Write-Host "      (resumable: re-run this script to continue, or Ctrl+C and re-run)"
& curl.exe -L -C - --retry 30 --retry-delay 5 --retry-all-errors -o $wheel $url
if ($LASTEXITCODE -ne 0) { throw "curl exited with code $LASTEXITCODE" }

Write-Host "[2/4] installing torch from the local wheel (small deps come from PyPI)"
& $Python -m pip install $wheel
if ($LASTEXITCODE -ne 0) { throw "pip failed on the torch wheel" }

Write-Host "[3/4] installing torchvision from the cu128 index"
& $Python -m pip install torchvision --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "pip failed on torchvision" }

Write-Host "[4/4] verifying the environment"
& $Python (Join-Path $root "scripts\check_env.py")
