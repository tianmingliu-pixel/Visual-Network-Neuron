<#
.SYNOPSIS
  把你电脑上的 NeuroCore 后端开放到公网（免费 Cloudflare 隧道），给 Vercel 上的网页使用
  Expose the local NeuroCore backend through a free Cloudflare quick tunnel for the Vercel UI.
.EXAMPLE
  .\scripts\serve_public.ps1                       # 启动后端 + 隧道，打开网站并自动连上
  .\scripts\serve_public.ps1 -Publish              # 同时把新地址推送到 GitHub，Vercel 自动更新，别人打开网站也能直接连上
  .\scripts\serve_public.ps1 -Site https://xxx.vercel.app
.NOTES
  · 训练用你自己电脑的 CPU / GPU，数据留在你电脑上；关掉这个窗口（或 Ctrl+C）网站就连不上了。
  · 必须设置口令：没有口令的人只能观看训练过程，不能训练、上传，也看不到你的数据和记忆库。
  · 免费隧道每次启动地址都会变；用 -Publish 自动更新网站里的默认地址。
#>
param(
    [string]$Token = "",
    [int]$Port = 8765,
    [string]$Site = "",
    [switch]$Publish,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot 'common.ps1')
Set-Location $script:ProjectRoot
$dataDir = Join-Path $script:ProjectRoot 'data'
New-Item -ItemType Directory -Force $dataDir | Out-Null
$siteFile = Join-Path $dataDir '.public_site'
$tokenFile = Join-Path $dataDir '.public_token'

function Step($t) { Write-Host "`n==> $t" -ForegroundColor Cyan }
function Fail($t) { Write-Host "✗ $t" -ForegroundColor Red; exit 1 }

# ---- Python ----------------------------------------------------------------------
Step "检查 Python / PyTorch"
$py = Get-VenvPython
if (-not (Test-Path $py)) {
    $tp = Find-TorchPython
    if (-not $tp) { Fail "没有找到装了 PyTorch 的 Python，请先运行 .\scripts\setup_env.ps1" }
    $py = $tp.Cmd[0]
}
& $py -c "import starlette, uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) { & $py -m pip install -r requirements.txt -q }
Write-Host "Python: $py"

# ---- 口令 / token ------------------------------------------------------------------
Step "访问口令"
if (-not $Token -and (Test-Path $tokenFile)) { $Token = (Get-Content $tokenFile -Raw).Trim() }
if (-not $Token) {
    $Token = Read-Host "设置一个口令（在网页右上角「后端」里输入它才能训练/上传；保存在 data\.public_token，不会上传到 GitHub）"
    if (-not $Token) { Fail "必须设置口令，否则任何拿到网址的人都能控制你的电脑上的训练。" }
    Set-Content -Path $tokenFile -Value $Token -Encoding UTF8
}
Write-Host "口令已设置（修改：删除 data\.public_token 后重新运行）"

# ---- cloudflared --------------------------------------------------------------------
Step "检查 Cloudflare 隧道程序 cloudflared"
function Find-Cloudflared {
    $c = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($p in @("$env:ProgramFiles\cloudflared\cloudflared.exe", "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe",
                     "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cloudflared.exe")) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    $hit = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter cloudflared.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { return $hit.FullName }
    return $null
}
$cf = Find-Cloudflared
if (-not $cf) {
    Write-Host "正在安装 cloudflared（winget）…"
    winget install --id Cloudflare.cloudflared -e --accept-source-agreements --accept-package-agreements
    $cf = Find-Cloudflared
}
if (-not $cf) { Fail "cloudflared 安装失败。可手动下载：https://github.com/cloudflare/cloudflared/releases （cloudflared-windows-amd64.exe 改名为 cloudflared.exe 放进本项目的 scripts 文件夹）" }
Write-Host "cloudflared: $cf"

# ---- 启动后端 / backend ----------------------------------------------------------------
Step "启动后端 http://127.0.0.1:$Port"
$env:NEUROCORE_TOKEN = $Token
$env:NEUROCORE_PRIVATE_READ = "1"
$beLog = Join-Path $dataDir 'backend.log'; $beErr = Join-Path $dataDir 'backend.err.log'
$be = Start-Process -FilePath $py -ArgumentList @('-m', 'server', '--host', '127.0.0.1', '--port', "$Port") `
    -RedirectStandardOutput $beLog -RedirectStandardError $beErr -PassThru -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 500
    if ($be.HasExited) { break }
    try { $v = Invoke-RestMethod "http://127.0.0.1:$Port/api/version" -TimeoutSec 2; $ok = $true; break } catch { }
}
if (-not $ok) { Get-Content $beErr -Tail 30 -ErrorAction SilentlyContinue; Fail "后端没有启动成功（日志：data\backend.err.log）。端口 $Port 是否被占用？先关掉其它 run.ps1 -Task ui 窗口。" }
Write-Host "后端已启动（v$($v.version)）"

# ---- 启动隧道 / tunnel ----------------------------------------------------------------
Step "建立公网隧道（Cloudflare quick tunnel）"
$tnLog = Join-Path $dataDir 'tunnel.log'; $tnOut = Join-Path $dataDir 'tunnel.out.log'
Remove-Item $tnLog, $tnOut -ErrorAction SilentlyContinue
$tn = Start-Process -FilePath $cf -ArgumentList @('tunnel', '--no-autoupdate', '--url', "http://127.0.0.1:$Port") `
    -RedirectStandardError $tnLog -RedirectStandardOutput $tnOut -PassThru -WindowStyle Hidden
$url = $null
for ($i = 0; $i -lt 90 -and -not $url; $i++) {
    Start-Sleep -Seconds 1
    if ($tn.HasExited) { break }
    $txt = (Get-Content $tnLog -Raw -ErrorAction SilentlyContinue) + (Get-Content $tnOut -Raw -ErrorAction SilentlyContinue)
    if ($txt -match 'https://[a-z0-9-]+\.trycloudflare\.com') { $url = $Matches[0] }
}
if (-not $url) {
    Get-Content $tnLog -Tail 20 -ErrorAction SilentlyContinue
    Stop-Process -Id $be.Id -ErrorAction SilentlyContinue
    Fail "没有拿到隧道地址（日志：data\tunnel.log）。检查网络是否能访问 Cloudflare。"
}
Write-Host "等待公网地址生效…"
for ($i = 0; $i -lt 30; $i++) {
    try { Invoke-RestMethod "$url/api/version" -TimeoutSec 5 | Out-Null; break } catch { Start-Sleep -Seconds 2 }
}
try { Set-Clipboard -Value $url } catch { }

# ---- 网站地址 / site ----------------------------------------------------------------------
if (-not $Site -and (Test-Path $siteFile)) { $Site = (Get-Content $siteFile -Raw).Trim() }
if (-not $Site) { $Site = Read-Host "你的 Vercel 网站地址（例如 https://visual-network-neuron.vercel.app，留空跳过）" }
if ($Site) { $Site = $Site.TrimEnd('/'); Set-Content -Path $siteFile -Value $Site -Encoding UTF8 }

if ($Publish) {
    Step "把新地址推送到 GitHub（Vercel 会自动重新发布）"
    $f = Join-Path $script:ProjectRoot 'deploy\backend_url.txt'
    Set-Content -Path $f -Value $url -Encoding ASCII
    & git add -- 'deploy/backend_url.txt'
    & git commit -q -m "后端地址 → $url" -- 'deploy/backend_url.txt'
    & git push -q
    if ($LASTEXITCODE -eq 0) { Write-Host "已推送；约 1 分钟后别人打开网站也会自动连上。" } else { Write-Host "推送失败，网站默认地址没有更新（你自己用下面的链接照样能连）。" -ForegroundColor Yellow }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  后端公网地址：$url   （已复制到剪贴板）" -ForegroundColor Green
if ($Site) { Write-Host "  网站：         $Site   （你自己打开时会自动用本机后端；别人打开连这个隧道）" -ForegroundColor Green }
Write-Host "  网站右上角「后端」里输入口令即可训练 / 上传" -ForegroundColor Green
Write-Host "  关闭这个窗口或按 Ctrl+C = 停止后端和隧道" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
if ($Site -and -not $NoBrowser) { Start-Process $Site }

try {
    while ($true) {
        Start-Sleep -Seconds 3
        if ($be.HasExited) { Write-Host "后端已退出（日志：data\backend.err.log）" -ForegroundColor Red; break }
        if ($tn.HasExited) { Write-Host "隧道已断开（日志：data\tunnel.log），请重新运行本脚本" -ForegroundColor Red; break }
    }
} finally {
    Write-Host "正在停止后端和隧道…"
    foreach ($p in @($tn, $be)) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
}
