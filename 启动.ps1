# 启动华华助手，并把控制台输出同时写进 runtime\app_console_<时间>.log
#
# 为什么要这个脚本：2026-11-23 那次「工具明明启动了、日志也在写，但一局都没记下来」，
# 排查时发现**控制台输出没落盘**，进程退出原因完全不可见。现在原样 tee 一份，
# 崩了能直接看到 traceback。
#
# 用法：双击 启动.bat（本文件不用手动跑）。

$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

$runtime = Join-Path $root 'runtime'
if (-not (Test-Path -LiteralPath $runtime)) {
    New-Item -ItemType Directory -Path $runtime -Force | Out-Null
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $runtime "app_console_$stamp.log"

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host '[华华助手] 虚拟环境不存在，退回系统 python（可能缺依赖）' -ForegroundColor Yellow
    $python = 'python'
}

$logPath = Join-Path $root 'runtime\game_log_path.txt'
Write-Host '========================================================'
Write-Host ' 华华助手  S8 画中雪原·作画'
Write-Host '========================================================'
Write-Host " Python  : $python"
Write-Host " 工作目录: $root"
Write-Host " 控制台日志: $log"
Write-Host ''
Write-Host ' 请保持这个窗口开着 —— 关掉窗口等于停止记录。'
Write-Host ' 画完一局后，日志会同时在下面显示并写入上面的文件。'
Write-Host '========================================================'
Write-Host ''

# -u 关闭缓冲，日志文件才是实时的（否则崩了就丢最后一段）
# 先把文件建出来写表头：Tee-Object 只在**有输出**时才建文件，
# 而 GUI 正常启动时一句都不打印，于是崩溃前连日志文件都不存在。
"[华华助手] 启动于 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  python=$python" |
    Out-File -FilePath $log -Encoding utf8

# -u 关闭缓冲，日志文件才是实时的（否则崩了就丢最后一段）
& $python -u (Join-Path $root 'run_app.py') 2>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE

Write-Host ''
Write-Host '========================================================'
if ($code -eq 0) {
    Write-Host ' [华华助手] 已正常退出' -ForegroundColor Green
} else {
    Write-Host " [华华助手] 异常退出，退出码 $code" -ForegroundColor Red
    Write-Host ' 把上面这段（或日志文件）发我，能直接定位原因。'
}
Write-Host " 日志已保存: $log"
Write-Host '========================================================'
