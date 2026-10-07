# =============================================================================
# NeuroCore - shared PowerShell helpers (dot-sourced by the other scripts)
# 公共函数：查找 Python、读取 NVIDIA 驱动、选择 PyTorch 安装源
# Compatible with Windows PowerShell 5.1 and PowerShell 7+ (Windows/Linux/macOS)
# =============================================================================

$script:MinPython = [version]'3.10'
$script:MaxPython = [version]'3.14'   # newest Python version tested with current PyTorch wheels
$script:TorchIndexBase = 'https://download.pytorch.org/whl'

$script:OnWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or ($IsWindows -eq $true)
$script:OnMac = ($IsMacOS -eq $true)
$script:ProjectRoot = Split-Path -Parent $PSScriptRoot

function Write-Status {
    param([string]$Status, [string]$Message)
    $color = switch ($Status) {
        'PASS' { 'Green' } 'OK' { 'Green' } 'WARN' { 'Yellow' }
        'FAIL' { 'Red' } 'STEP' { 'Cyan' } default { 'Gray' }
    }
    Write-Host ('[{0,-4}] ' -f $Status) -ForegroundColor $color -NoNewline
    Write-Host $Message
}

function Test-Command { param([string]$Name) return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

# Run a command given as an array (exe + leading args) and return stdout lines.
function Invoke-Cmd {
    param([string[]]$Cmd, [string[]]$Arguments)
    $exe = $Cmd[0]
    $pre = @(); if ($Cmd.Count -gt 1) { $pre = $Cmd[1..($Cmd.Count - 1)] }
    $all = @($pre) + @($Arguments)
    $out = & $exe @all 2>&1
    return , @($out | ForEach-Object { "$_" })
}

# 注意：Windows PowerShell 5.1 传参给外部程序时会吃掉双引号，所以 -c 代码里不能用双引号
# Windows PowerShell 5.1 strips embedded double quotes from native-command arguments: keep -c code free of them.
function Get-PythonVersion {
    param([string[]]$Cmd)
    try {
        $o = Invoke-Cmd $Cmd @('-c', 'import sys;print(*sys.version_info[:3],sep=chr(46))')
        if ($LASTEXITCODE -eq 0 -and ($o -join '') -match '^\s*(\d+\.\d+\.\d+)') { return [version]$Matches[1] }
    } catch { }
    return $null
}

# 所有可能的 Python 命令（PATH 上的 python 优先，与 `pip` 指向一致）
# All candidate interpreters; PATH `python` first (that is what a bare `pip` uses).
function Get-PythonCandidates {
    param([string]$Preferred)
    $cands = New-Object System.Collections.Generic.List[object]
    if ($Preferred) {
        # 完整路径（可能含空格）当作一个整体；否则如 "py -3.12" 按空格拆开 / a full path may contain spaces
        if (Test-Path -LiteralPath $Preferred -PathType Leaf) { $cands.Add(@($Preferred)) } else { $cands.Add(@($Preferred -split ' ')) }
    }
    foreach ($n in 'python', 'python3') { if (Test-Command $n) { $cands.Add(@($n)) } }
    if (Test-Command 'py') {
        foreach ($v in '3.13', '3.12', '3.11', '3.10', '3.14') { $cands.Add(@('py', "-$v")) }
    }
    return , $cands
}

# 返回 @{ Cmd = @('py','-3.12'); Version = [version]; Supported = bool } ；找不到则 $null
# Finds the best Python in [MinPython, MaxPython].
function Find-Python {
    param([string]$Preferred)
    $fallback = $null
    foreach ($c in (Get-PythonCandidates $Preferred)) {
        $v = Get-PythonVersion $c
        if (-not $v) { continue }
        $mm = [version]("{0}.{1}" -f $v.Major, $v.Minor)
        $hit = @{ Cmd = $c; Version = $v; Supported = ($mm -ge $script:MinPython -and $mm -le $script:MaxPython) }
        if ($hit.Supported) { return $hit }
        if (-not $fallback) { $fallback = $hit }
    }
    return $fallback
}

# 查找“已经装好 PyTorch 且能用”的系统 Python —— 可直接复用，免去数 GB 下载
# Find a system Python that already has a working PyTorch (reuse it, skip the multi-GB download).
function Find-TorchPython {
    param([string]$Preferred)
    $seen = @{}
    foreach ($c in (Get-PythonCandidates $Preferred)) {
        $st = Get-TorchStatus $c
        if (-not $st -or -not $st.torch -or -not $st.matmul_ok) { continue }
        if ($seen.ContainsKey($st.executable)) { continue }
        $seen[$st.executable] = $true
        $v = [version]$st.python
        $mm = [version]("{0}.{1}" -f $v.Major, $v.Minor)
        if ($mm -lt $script:MinPython) { continue }
        return @{ Cmd = @($st.executable); Version = $v; Supported = $true; Torch = $st }
    }
    return $null
}

# 读取 NVIDIA 显卡与驱动支持的最高 CUDA 版本 / GPU + max CUDA version supported by the driver
function Get-NvidiaInfo {
    if (-not (Test-Command 'nvidia-smi')) { return $null }
    try {
        $q = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
        $hdr = (& nvidia-smi 2>$null) -join "`n"
        $cuda = $null
        if ($hdr -match 'CUDA Version:\s*([\d\.]+)') { $cuda = [version]$Matches[1] }
        $gpus = @($q | Where-Object { $_ } | ForEach-Object {
            $p = $_ -split ',\s*'
            [pscustomobject]@{ Name = $p[0]; Driver = $p[1]; Memory = $p[2] }
        })
        if ($gpus.Count -eq 0) { return $null }
        return [pscustomobject]@{ Gpus = $gpus; CudaVersion = $cuda }
    } catch { return $null }
}

# 根据驱动 CUDA 版本给出候选 wheel 源（按优先级）/ ordered candidate wheel indexes
# Notes (2026): PyTorch 2.12+ ships cu126 (legacy), cu130 (stable) and cu132;
# 2.14 is the last release with CUDA 12.x wheels. Newer drivers run older CUDA builds.
function Get-TorchIndexCandidates {
    param($CudaVersion, [string]$Override = 'auto')
    if ($Override -and $Override -ne 'auto') { return @($Override) }
    if ($script:OnMac) { return @('default') }            # macOS: PyPI wheel (MPS)
    if (-not $CudaVersion) { return @('cpu') }
    $list = @()
    if ($CudaVersion -ge [version]'13.2') { $list += 'cu132' }
    if ($CudaVersion -ge [version]'13.0') { $list += 'cu130' }
    if ($CudaVersion -ge [version]'12.6') { $list += 'cu126' }
    $list += 'cpu'
    return $list
}

function Get-VenvPython {
    param([string]$VenvDir = (Join-Path $script:ProjectRoot '.venv'))
    if ($script:OnWindows) { return (Join-Path $VenvDir 'Scripts\python.exe') }
    return (Join-Path $VenvDir 'bin/python')
}

function Get-TorchStatus {
    param([string[]]$Cmd)          # exe path or command array, e.g. @('py','-3.12')
    $verify = Join-Path $PSScriptRoot 'verify_torch.py'
    try {
        $out = Invoke-Cmd $Cmd @($verify)
        if ($LASTEXITCODE -ne 0) { return $null }
        $line = $out | Where-Object { $_ -like '{*' } | Select-Object -Last 1
        return ($line | ConvertFrom-Json)
    } catch { return $null }
}
