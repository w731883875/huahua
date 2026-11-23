@echo off
rem 华华助手启动器（S8 画中雪原·作画）
rem
rem 控制台输出会同时写进 runtime\app_console_<时间>.log，
rem 这样工具崩了/退出了也能看到原因，不会像 2026-11-23 那次一样查不到。
rem
rem 注意：必须让这个窗口一直开着，关窗口就等于停止记录。

chcp 65001 >nul
cd /d "%~dp0"

if not exist "run_app.py" (
    echo [错误] 找不到 run_app.py，请把本文件放在 tools\huahua 根目录。
    echo.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0启动.ps1"

echo.
echo 按任意键关闭窗口...
pause >nul
