<#
.SYNOPSIS
    NeuroCore 一键：检查 -> 部署 -> 演示 / One command: check -> setup -> demo
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\deploy.ps1
    powershell -ExecutionPolicy Bypass -File .\deploy.ps1 -Cuda cpu
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'cpu', 'cu126', 'cu130', 'cu132', 'default')]
    [string]$Cuda = 'auto',
    [switch]$Recreate,
    [switch]$FreshTorch,
    [switch]$SkipDemo
)
$ErrorActionPreference = 'Continue'
$s = Join-Path $PSScriptRoot 'scripts'

& (Join-Path $s 'check_env.ps1') -ReportPath (Join-Path $PSScriptRoot 'env_report.json')
if ($LASTEXITCODE -ne 0) { Write-Host "Environment check failed - fix the FAIL items first." -ForegroundColor Red; exit 1 }

$setupArgs = @{ Cuda = $Cuda }
if ($Recreate) { $setupArgs.Recreate = $true }
if ($FreshTorch) { $setupArgs.FreshTorch = $true }
& (Join-Path $s 'setup_env.ps1') @setupArgs
if ($LASTEXITCODE -ne 0) { exit 1 }

if (-not $SkipDemo) { & (Join-Path $s 'run.ps1') -Task demo }
exit $LASTEXITCODE
