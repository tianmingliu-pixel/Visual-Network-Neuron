<#
.SYNOPSIS
  把 NeuroCore 推送到你的 GitHub 仓库（默认 Visual-Network-Neuron）/ push this project to GitHub
.EXAMPLE
  .\scripts\publish_github.ps1 -User 你的GitHub用户名
  .\scripts\publish_github.ps1 -User 你的GitHub用户名 -Message "加入机器记忆库"
.NOTES
  第一次推送时 Git 会弹出 GitHub 登录窗口（浏览器授权即可）。
  data\（上传的数据、记忆库）、exports\、.venv\ 不会被上传（见 .gitignore）。
#>
param(
    [string]$User = "",
    [string]$Repo = "Visual-Network-Neuron",
    [string]$Branch = "main",
    [string]$Message = "",
    [string]$RemoteUrl = ""      # 可选：直接给完整地址（例如 SSH 地址 git@github.com:用户/仓库.git）
)
$ErrorActionPreference = "Continue"   # 注意：Windows PowerShell 5.1 下 Stop 会把 git 的进度输出当成错误
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($t) { Write-Host "`n==> $t" -ForegroundColor Cyan }
function Fail($t) { Write-Host "✗ $t" -ForegroundColor Red; exit 1 }
$gitExe = if ($IsLinux -or $IsMacOS) { "git" } else { "git.exe" }
function Invoke-Git { & $gitExe @args; if ($LASTEXITCODE -ne 0) { throw "git $($args -join ' ') 失败 (exit $LASTEXITCODE)" } }

Step "检查 Git"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "没有找到 git。请先安装：winget install --id Git.Git -e   （装完重新打开 PowerShell）"
}
git --version

if (-not $User -and -not $RemoteUrl) {
    $guess = ""
    if (Get-Command gh -ErrorAction SilentlyContinue) { $guess = (gh api user --jq .login 2>$null) }
    $User = Read-Host "你的 GitHub 用户名$(if ($guess) { "（回车使用 $guess）" })"
    if (-not $User) { $User = $guess }
}
if (-not $User -and -not $RemoteUrl) { Fail "需要 GitHub 用户名：.\scripts\publish_github.ps1 -User 你的用户名" }
$url = if ($RemoteUrl) { $RemoteUrl } else { "https://github.com/$User/$Repo.git" }

Step "前端是否已构建（Vercel 直接使用 web\dist）"
if (-not (Test-Path "web\dist\index.html")) { Fail "缺少 web\dist\index.html，请先运行 .\scripts\run.ps1 -Task build-ui" }
Write-Host "web\dist 已就绪"

Step "初始化仓库"
if (-not (Test-Path ".git")) { Invoke-Git init -q; Write-Host "已创建本地仓库" }
if (-not (git config user.name)) {
    $n = Read-Host "Git 提交用的名字（例如 $User）"; if (-not $n) { $n = $User }
    Invoke-Git config user.name $n
}
if (-not (git config user.email)) {
    $e = Read-Host "Git 提交用的邮箱（GitHub 账户邮箱，或 $User@users.noreply.github.com）"
    if (-not $e) { $e = "$User@users.noreply.github.com" }
    Invoke-Git config user.email $e
}
Invoke-Git config core.autocrlf true
Invoke-Git config core.quotepath false

$remotes = @(git remote)
if ($remotes -contains "origin") { Invoke-Git remote set-url origin $url } else { Invoke-Git remote add origin $url }
Write-Host "远程仓库：$url"

Step "自动部署后端的 GitHub Actions 工作流"
$wfSrc = Join-Path $root "deploy\deploy-backend-hf.yml"
$wfDst = Join-Path $root ".github\workflows\deploy-backend-hf.yml"
if (Test-Path $wfSrc) {
    New-Item -ItemType Directory -Force (Split-Path $wfDst) | Out-Null
    Copy-Item $wfSrc $wfDst -Force
    Write-Host "已放入 .github\workflows\deploy-backend-hf.yml（推送后自动同步到 Hugging Face Space）"
}

Step "提交本地改动"
Invoke-Git add -A
$staged = git diff --cached --name-only
if ($staged) {
    if (-not $Message) { $Message = "NeuroCore 更新 $(Get-Date -Format 'yyyy-MM-dd HH:mm')" }
    Invoke-Git commit -q -m $Message
    Write-Host "已提交 $(@($staged).Count) 个文件"
} else { Write-Host "没有新的改动" }
$cur = (git rev-parse --abbrev-ref HEAD)
if ($cur -ne $Branch) { Invoke-Git branch -M $Branch }

Step "与 GitHub 上已有的内容合并"
git fetch origin $Branch *> $null
if ($LASTEXITCODE -eq 0) {
    git merge-base --is-ancestor "origin/$Branch" HEAD *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "GitHub 仓库里已有提交（例如建仓库时生成的 README），合并进来（冲突时以本地为准）"
        Invoke-Git merge "origin/$Branch" --allow-unrelated-histories -X ours -m "合并 GitHub 上已有的内容"
    }
} else { Write-Host "远程还是空仓库" }

Step "推送"
Invoke-Git push -u origin $Branch
Write-Host "`n✓ 已推送到 $($url -replace '\.git$','')" -ForegroundColor Green
Write-Host "  下一步（只需做一次）：见 DEPLOY.md —— Vercel 导入这个仓库；Hugging Face 建 Docker Space 跑后端。"
