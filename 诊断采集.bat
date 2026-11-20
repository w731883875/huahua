@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 采集链路诊断：会打印进程完整性级别、WGC 是否能建采集项、以及实际抓帧走的路径。
echo.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -u -B _wgc_probe.py %*
) else (
    python -u -B _wgc_probe.py %*
)
echo.
echo 报告同时留了一份在 runtime\capture_selftest.log（软件启动时自动写）。
echo [按任意键关闭]
pause >nul
