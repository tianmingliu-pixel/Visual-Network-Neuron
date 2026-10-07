<#
.SYNOPSIS
    检查本机是否满足 NeuroCore 运行条件 / Check whether this machine meets NeuroCore requirements.
.DESCRIPTION
    Checks: OS / PowerShell / Python (3.10-3.14) / pip / venv / Git / NVIDIA driver + CUDA /
    RAM / free disk / Windows long paths / execution policy / existing .venv + PyTorch.
    Prints a PASS/WARN/FAIL table and the recommended PyTorch wheel index.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\check_env.ps1
    powershell -ExecutionPolicy Bypass -File .\scripts\check_env.ps1 -ReportPath env_report.json
#>
[CmdletBinding()]
param(
    [string]$Python,          # 指定 Python，例如 "py -3.12" / force a specific interpreter
    [string]$ReportPath       # 输出 JSON 报告 / write a JSON report
)

. (Join-Path $PSScriptRoot 'common.ps1')
$results = New-Object System.Collections.Generic.List[object]
function Add-Result([string]$Item, [string]$Status, [string]$Detail, [string]$Fix = '') {
    $results.Add([pscustomobject]@{ Item = $Item; Status = $Status; Detail = $Detail; Fix = $Fix })
}

Write-Host "`n=== NeuroCore environment check ===`n" -ForegroundColor Cyan

# --- OS / 操作系统 -------------------------------------------------------------
try {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    $osDesc = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
} catch { $arch = $env:PROCESSOR_ARCHITECTURE; $osDesc = [Environment]::OSVersion.VersionString }
if (-not [Environment]::Is64BitOperatingSystem) {
    Add-Result 'OS' 'FAIL' "$osDesc ($arch)" 'PyTorch requires a 64-bit OS'
} else { Add-Result 'OS' 'PASS' "$osDesc ($arch)" }

# --- PowerShell -------------------------------------------------------------
$psv = $PSVersionTable.PSVersion
if ($psv -ge [version]'5.1') { Add-Result 'PowerShell' 'PASS' "$psv ($($PSVersionTable.PSEdition))" }
else { Add-Result 'PowerShell' 'FAIL' "$psv" 'Install PowerShell 5.1+ or 7+' }

# --- Python -----------------------------------------------------------------
$py = Find-Python -Preferred $Python
if (-not $py) {
    Add-Result 'Python' 'FAIL' 'not found' 'Install Python 3.12 from https://www.python.org (tick "Add to PATH") or: winget install Python.Python.3.12'
} elseif (-not $py.Supported) {
    Add-Result 'Python' 'FAIL' "$($py.Version) via '$($py.Cmd -join ' ')'" "Need $($script:MinPython)-$($script:MaxPython). Install: winget install Python.Python.3.12"
} else {
    Add-Result 'Python' 'PASS' "$($py.Version) via '$($py.Cmd -join ' ')'"
    $pip = Invoke-Cmd $py.Cmd @('-m', 'pip', '--version')
    if ($LASTEXITCODE -eq 0) { Add-Result 'pip' 'PASS' (($pip -join ' ') -replace ' from .*', '') }
    else { Add-Result 'pip' 'FAIL' 'pip missing' "$($py.Cmd -join ' ') -m ensurepip --upgrade" }
    $null = Invoke-Cmd $py.Cmd @('-c', 'import venv')
    if ($LASTEXITCODE -eq 0) { Add-Result 'venv' 'PASS' 'available' }
    else { Add-Result 'venv' 'FAIL' 'venv module missing' 'Linux: sudo apt install python3-venv' }
}

# --- Git (optional) ---------------------------------------------------------
if (Test-Command 'git') { Add-Result 'Git' 'PASS' ((& git --version) -join '') }
else { Add-Result 'Git' 'WARN' 'not found (optional)' 'winget install Git.Git' }

# --- Node.js (optional, only for UI development) -------------------------------
if (Test-Command 'node') { Add-Result 'Node.js' 'PASS' ("node " + ((& node --version) -join '') + " (only needed for ui-dev / build-ui)") }
else { Add-Result 'Node.js' 'INFO' 'not found - optional (the prebuilt UI works without it)' 'For UI development: winget install OpenJS.NodeJS.LTS' }

# --- GPU / CUDA ---------------------------------------------------------------
$nv = Get-NvidiaInfo
if ($nv) {
    foreach ($g in $nv.Gpus) { Add-Result 'GPU' 'PASS' "$($g.Name) | driver $($g.Driver) | $($g.Memory)" }
    if ($nv.CudaVersion -and $nv.CudaVersion -ge [version]'12.6') {
        Add-Result 'CUDA (driver max)' 'PASS' "$($nv.CudaVersion)"
    } else {
        Add-Result 'CUDA (driver max)' 'WARN' "$($nv.CudaVersion)" 'Driver too old for current CUDA wheels; update the NVIDIA driver or use CPU'
    }
} elseif ($script:OnMac) {
    Add-Result 'GPU' 'PASS' 'macOS - Apple MPS backend will be used if available'
} else {
    Add-Result 'GPU' 'WARN' 'no NVIDIA GPU / nvidia-smi not found - CPU mode' 'Optional: install the latest NVIDIA driver for CUDA acceleration'
}
$cands = Get-TorchIndexCandidates -CudaVersion $(if ($nv) { $nv.CudaVersion } else { $null })
Add-Result 'PyTorch wheel' 'INFO' ("recommended index order: " + ($cands -join ' -> '))

# --- RAM / Disk -------------------------------------------------------------
try {
    if ($script:OnWindows) {
        $ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
    } elseif ($script:OnMac) {
        $ramGB = [math]::Round([double](& sysctl -n hw.memsize) / 1GB, 1)
    } else {
        $kb = (Get-Content /proc/meminfo | Select-String 'MemTotal' | ForEach-Object { ($_ -split '\s+')[1] })
        $ramGB = [math]::Round([double]$kb / 1MB, 1)
    }
    if ($ramGB -ge 8) { Add-Result 'RAM' 'PASS' "$ramGB GB" }
    else { Add-Result 'RAM' 'WARN' "$ramGB GB" '8 GB+ recommended' }
} catch { Add-Result 'RAM' 'WARN' 'could not read' }

try {
    $drive = (Get-Item $script:ProjectRoot).PSDrive
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    if ($freeGB -ge 10) { Add-Result 'Disk free' 'PASS' "$freeGB GB on $($drive.Name):" }
    elseif ($freeGB -ge 5) { Add-Result 'Disk free' 'WARN' "$freeGB GB" 'CUDA PyTorch needs ~5-8 GB' }
    else { Add-Result 'Disk free' 'FAIL' "$freeGB GB" 'Free at least 5 GB' }
} catch { Add-Result 'Disk free' 'WARN' 'could not read' }

# --- Windows specifics --------------------------------------------------------
if ($script:OnWindows) {
    $lp = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
    if ($lp -eq 1) { Add-Result 'Long paths' 'PASS' 'enabled' }
    else {
        Add-Result 'Long paths' 'WARN' 'disabled (pip may fail on deep paths)' `
            "Admin PowerShell: New-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force"
    }
    $ep = Get-ExecutionPolicy
    if ($ep -in 'Restricted', 'AllSigned') {
        Add-Result 'ExecutionPolicy' 'WARN' "$ep" 'Run scripts with: powershell -ExecutionPolicy Bypass -File <script>'
    } else { Add-Result 'ExecutionPolicy' 'PASS' "$ep" }
}

# --- Existing system PyTorch / 本机已装的 PyTorch --------------------------------
$sysTorch = Find-TorchPython -Preferred $Python
if ($sysTorch) {
    $t = $sysTorch.Torch
    $dev = if ($t.cuda_available) { "CUDA $($t.cuda_build) | " + (($t.gpus | ForEach-Object { $_.name }) -join ', ') }
           elseif ($t.mps) { 'MPS' } else { 'CPU only' }
    Add-Result 'PyTorch (system)' 'PASS' "torch $($t.torch) | $dev | $($t.executable)"
    if ($nv -and -not $t.cuda_available) {
        Add-Result 'PyTorch GPU' 'WARN' 'NVIDIA GPU present but system torch is CPU-only' 'For GPU: .\scripts\setup_env.ps1 -FreshTorch -Recreate'
    } else {
        Add-Result 'Reuse' 'INFO' 'setup_env.ps1 will reuse this PyTorch (no download)'
    }
} else { Add-Result 'PyTorch (system)' 'INFO' 'none found - setup_env.ps1 will install one' }

# --- Existing venv / 已有虚拟环境 ------------------------------------------------
$venvPy = Get-VenvPython
if (Test-Path $venvPy) {
    $ts = Get-TorchStatus $venvPy
    if ($ts -and $ts.torch) {
        $dev = if ($ts.cuda_available) { "CUDA $($ts.cuda_build) | " + (($ts.gpus | ForEach-Object { $_.name }) -join ', ') }
               elseif ($ts.mps) { 'MPS' } else { 'CPU only' }
        $st = if ($ts.matmul_ok) { 'PASS' } else { 'FAIL' }
        Add-Result 'PyTorch (.venv)' $st "torch $($ts.torch) | $dev"
        if ($nv -and -not $ts.cuda_available) {
            Add-Result 'PyTorch GPU' 'WARN' 'NVIDIA GPU present but torch is CPU-only' 'Re-run: .\scripts\setup_env.ps1 -Recreate'
        }
    } else { Add-Result 'PyTorch (.venv)' 'WARN' '.venv exists but torch not importable' '.\scripts\setup_env.ps1' }
} else { Add-Result 'PyTorch (.venv)' 'INFO' 'not installed yet' '.\scripts\setup_env.ps1' }

# --- Report / 报告 ------------------------------------------------------------
foreach ($r in $results) {
    Write-Status $r.Status ("{0,-18} {1}" -f $r.Item, $r.Detail)
    if ($r.Fix -and $r.Status -ne 'PASS') { Write-Host ("        -> " + $r.Fix) -ForegroundColor DarkGray }
}
$fails = @($results | Where-Object Status -eq 'FAIL').Count
$warns = @($results | Where-Object Status -eq 'WARN').Count
Write-Host ""
if ($fails -eq 0) { Write-Status 'PASS' "Ready to deploy ($warns warning(s)). Next: .\scripts\setup_env.ps1" }
else { Write-Status 'FAIL' "$fails blocking issue(s). Fix the items above, then re-run this check." }

if ($ReportPath) {
    [pscustomobject]@{ timestamp = (Get-Date).ToString('s'); torch_index_candidates = $cands; checks = $results } |
        ConvertTo-Json -Depth 5 | Set-Content -Path $ReportPath -Encoding UTF8
    Write-Host "Report written to $ReportPath"
}
exit $(if ($fails -eq 0) { 0 } else { 1 })
