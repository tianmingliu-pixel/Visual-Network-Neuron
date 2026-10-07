<#
.SYNOPSIS
    一键部署：创建 .venv、按 GPU 自动安装 PyTorch、安装依赖、运行自检
    One-shot deploy: create .venv, install the right PyTorch build for this GPU, install deps, self-test.
.PARAMETER Cuda
    auto (default) | cpu | cu126 | cu130 | cu132 | default (plain PyPI)
.PARAMETER FreshTorch
    Do not reuse an already-installed system PyTorch; download a separate build into .venv.
    默认会复用本机已装好的 PyTorch（.venv 使用 --system-site-packages），加此开关则单独下载。
.PARAMETER TorchVersion
    Pin a version, e.g. 2.14.0. Empty = latest available on the chosen index.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
    powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Cuda cpu -Recreate
    powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -FreshTorch -Recreate
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'cpu', 'cu126', 'cu130', 'cu132', 'default')]
    [string]$Cuda = 'auto',
    [string]$TorchVersion = '',
    [string]$Python,
    [switch]$Recreate,
    [switch]$FreshTorch,
    [switch]$SkipTests
)

. (Join-Path $PSScriptRoot 'common.ps1')
Set-Location $script:ProjectRoot
$logFile = Join-Path $script:ProjectRoot 'deploy.log'
Start-Transcript -Path $logFile -Append | Out-Null

function Stop-Deploy([string]$msg) {
    Write-Status 'FAIL' $msg
    exit 1   # finally{} below stops the transcript
}

try {
    Write-Host "`n=== NeuroCore setup ===`n" -ForegroundColor Cyan

    # 1. Python -----------------------------------------------------------------
    Write-Status 'STEP' 'Locating Python...'
    $venvDir = Join-Path $script:ProjectRoot '.venv'
    $nv = Get-NvidiaInfo
    $reuse = $false
    $py = $null
    $canReuse = (-not $FreshTorch) -and ($Cuda -eq 'auto') -and (-not $TorchVersion) -and
                ($Recreate -or -not (Test-Path (Get-VenvPython $venvDir)))
    if ($canReuse) {
        $py = Find-TorchPython -Preferred $Python
        if ($py) {
            $reuse = $true
            $t = $py.Torch
            $dev = 'CPU only'; if ($t.cuda_available) { $dev = "CUDA $($t.cuda_build)" }
            Write-Status 'OK' "Found existing PyTorch $($t.torch) ($dev) in $($t.executable) - will reuse it"
            if ($nv -and -not $t.cuda_available) {
                Write-Status 'WARN' 'An NVIDIA GPU is present but this PyTorch is CPU-only. For GPU use: .\scripts\setup_env.ps1 -FreshTorch -Recreate'
            }
        }
    }
    if (-not $py) { $py = Find-Python -Preferred $Python }
    if (-not $py -or -not $py.Supported) {
        Stop-Deploy "Need Python $($script:MinPython)-$($script:MaxPython). Run .\scripts\check_env.ps1 for details."
    }
    Write-Status 'OK' "Python $($py.Version) ($($py.Cmd -join ' '))"

    # 2. venv -------------------------------------------------------------------
    if ($Recreate -and (Test-Path $venvDir)) {
        Write-Status 'STEP' 'Removing old .venv...'
        Remove-Item -Recurse -Force $venvDir
    }
    $vpy = Get-VenvPython $venvDir
    if (-not (Test-Path $vpy)) {
        $venvArgs = @('-m', 'venv', $venvDir)
        if ($reuse) { $venvArgs += '--system-site-packages' }   # 继承系统 PyTorch / inherit system torch
        Write-Status 'STEP' "Creating virtual environment .venv $(if ($reuse) { '(sharing system site-packages)' })..."
        $null = Invoke-Cmd $py.Cmd $venvArgs
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $vpy)) { Stop-Deploy 'venv creation failed' }
    }
    Write-Status 'OK' ".venv -> $vpy"
    & $vpy -m pip install --upgrade pip setuptools wheel --disable-pip-version-check -q
    if ($LASTEXITCODE -ne 0) { Write-Status 'WARN' 'pip self-upgrade failed (network/proxy?) - continuing' }

    # 3. PyTorch ------------------------------------------------------------------
    $cudaVer = $null; if ($nv) { $cudaVer = $nv.CudaVersion }
    $cands = Get-TorchIndexCandidates -CudaVersion $cudaVer -Override $Cuda
    $pkg = 'torch'; if ($TorchVersion) { $pkg = "torch==$TorchVersion" }
    $installed = $null

    $current = Get-TorchStatus $vpy
    $wantGpu = ($cands[0] -like 'cu*')
    if ($reuse -and $current -and $current.torch) {
        Write-Status 'OK' "Reusing system PyTorch $($current.torch) - no download needed"
        $installed = $current
    } elseif ($current -and $current.torch -and (-not $wantGpu -or $current.cuda_available) -and -not $TorchVersion) {
        Write-Status 'OK' "PyTorch $($current.torch) already installed - skipping (use -Recreate to reinstall)"
        $installed = $current
    } else {
        foreach ($idx in $cands) {
            Write-Status 'STEP' "Installing $pkg from index '$idx' ..."
            $pipArgs = @('-m', 'pip', 'install', '--upgrade', $pkg, '--disable-pip-version-check')
            if ($idx -ne 'default') { $pipArgs += @('--index-url', "$($script:TorchIndexBase)/$idx") }
            & $vpy @pipArgs
            if ($LASTEXITCODE -ne 0) { Write-Status 'WARN' "index '$idx' failed, trying next"; continue }
            $st = Get-TorchStatus $vpy
            if (-not $st -or -not $st.torch) { Write-Status 'WARN' 'torch not importable, trying next'; continue }
            if ($idx -like 'cu*' -and -not $st.cuda_available) {
                Write-Status 'WARN' "torch $($st.torch) installed but CUDA unavailable with '$idx', trying next"
                & $vpy -m pip uninstall -y torch | Out-Null
                continue
            }
            $installed = $st; break
        }
    }
    if (-not $installed) { Stop-Deploy 'Could not install a working PyTorch build.' }
    $devDesc = 'CPU'
    if ($installed.cuda_available) { $devDesc = "CUDA $($installed.cuda_build) on " + (($installed.gpus | ForEach-Object { $_.name }) -join ', ') }
    elseif ($installed.mps) { $devDesc = 'Apple MPS' }
    Write-Status 'OK' "PyTorch $($installed.torch) | $devDesc"

    # 4. Project deps -------------------------------------------------------------
    Write-Status 'STEP' 'Installing requirements + neurocore (editable)...'
    & $vpy -m pip install -r requirements.txt --disable-pip-version-check -q
    if ($LASTEXITCODE -ne 0) { Stop-Deploy 'requirements install failed' }
    & $vpy -m pip install -e . --no-deps --disable-pip-version-check -q
    if ($LASTEXITCODE -ne 0) { Stop-Deploy 'editable install of neurocore failed' }
    Write-Status 'OK' 'dependencies installed'

    # 5. Self-test ----------------------------------------------------------------
    if (-not $SkipTests) {
        Write-Status 'STEP' 'Running unit tests (pytest)...'
        & $vpy -m pytest -q
        if ($LASTEXITCODE -ne 0) { Stop-Deploy 'tests failed - see output above / deploy.log' }
        Write-Status 'OK' 'all tests passed'
    }

    Write-Host ""
    Write-Status 'PASS' 'Deployment complete. Try:  .\scripts\run.ps1 -Task demo'
} finally {
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
}
exit 0
