<#
.SYNOPSIS
    运行 NeuroCore 任务 / Run NeuroCore tasks inside .venv
.EXAMPLE
    .\scripts\run.ps1 -Task demo                     # 全部演示 / all demos
    .\scripts\run.ps1 -Task demo -Model dit -Steps 200
    .\scripts\run.ps1 -Task test
    .\scripts\run.ps1 -Task list                     # 已注册模型 / registered models
    .\scripts\run.ps1 -Task info                     # PyTorch/GPU 状态 / torch + GPU status
    .\scripts\run.ps1 -Task ui                       # 训练可视化界面 http://127.0.0.1:8765（无需 Node）
    .\scripts\run.ps1 -Task ui-dev                   # 前端开发模式（Vite 热更新，需要 Node.js）
    .\scripts\run.ps1 -Task build-ui                 # 重新构建前端到 web\dist（需要 Node.js）
#>
[CmdletBinding()]
param(
    [ValidateSet('demo', 'test', 'list', 'info', 'ui', 'ui-dev', 'build-ui')]
    [string]$Task = 'demo',
    [ValidateSet('all', 'transformer', 'vit', 'unet', 'dit')]
    [string]$Model = 'all',
    [int]$Steps = 60,
    [string]$Device = 'auto',
    [int]$Port = 8765
)

. (Join-Path $PSScriptRoot 'common.ps1')
Set-Location $script:ProjectRoot
$vpy = Get-VenvPython
if (-not (Test-Path $vpy)) {
    Write-Status 'FAIL' '.venv not found. Run .\scripts\setup_env.ps1 first.'
    exit 1
}

switch ($Task) {
    'demo' { & $vpy demo.py --model $Model --steps $Steps --device $Device }
    'test' { & $vpy -m pytest -q }
    'list' { & $vpy demo.py --list }
    'info' { & $vpy (Join-Path $PSScriptRoot 'verify_torch.py') | ConvertFrom-Json | Format-List }
    'ui' {
        # 后端同时提供已构建好的前端页面 / backend also serves the prebuilt UI
        & $vpy -c "import starlette, uvicorn" 2>$null
        if ($LASTEXITCODE -ne 0) { & $vpy -m pip install -r requirements.txt -q }
        Write-Status 'STEP' "Starting UI at http://127.0.0.1:$Port  (Ctrl+C to stop)"
        & $vpy -m server --port $Port --open
    }
    'build-ui' {
        if (-not (Test-Command 'npm')) { Write-Status 'FAIL' 'Node.js/npm not found. Install: winget install OpenJS.NodeJS.LTS'; exit 1 }
        Push-Location (Join-Path $script:ProjectRoot 'web')
        try {
            & npm install
            if ($LASTEXITCODE -eq 0) { & npm run build }
        } finally { Pop-Location }
    }
    'ui-dev' {
        if (-not (Test-Command 'npm')) { Write-Status 'FAIL' 'Node.js/npm not found. Install: winget install OpenJS.NodeJS.LTS'; exit 1 }
        $web = Join-Path $script:ProjectRoot 'web'
        if (-not (Test-Path (Join-Path $web 'node_modules'))) { Push-Location $web; & npm install; Pop-Location }
        Write-Status 'STEP' "Backend on :$Port, Vite dev server on :5173 (hot reload)"
        $backend = Start-Process -FilePath $vpy -ArgumentList @('-m', 'server', '--port', "$Port") `
            -WorkingDirectory $script:ProjectRoot -PassThru -NoNewWindow
        Push-Location $web
        try { & npm run dev -- --open }
        finally {
            Pop-Location
            if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id -Force }
        }
    }
}
exit $LASTEXITCODE
