$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue

if ($pythonCommand) {
    $pythonExe = $pythonCommand.Source
} else {
    $codexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $codexPython) {
        $pythonExe = $codexPython
    } else {
        throw "未找到 Python。请安装 Python 3.10+，然后重新运行此脚本。"
    }
}

Write-Host "Qwen3.8 Flash Next Explorer" -ForegroundColor Magenta
Write-Host "打开浏览器访问: http://127.0.0.1:8000"
Write-Host "按 Ctrl+C 停止服务。"

& $pythonExe (Join-Path $projectRoot "server.py") --host 127.0.0.1 --port 8000
