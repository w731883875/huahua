@echo off
cd /d "%~dp0"
rem Keep this file pure ASCII: cmd.exe parses .bat as ANSI/GBK on zh-CN Windows,
rem and UTF-8 Chinese comments here previously broke the launch.
rem
rem HUAHUA_SELF_SHOT: 12 seconds after start, take an in-process screenshot of
rem our own windows into runtime\_selfshot\. Needed because the tool window sets
rem WDA_EXCLUDEFROMCAPTURE, so every external capture (snipping, OBS, remote
rem desktop) only sees pure black and cannot tell whether the UI painted.
set "HUAHUA_SELF_SHOT=runtime\_selfshot"
if not exist "runtime" mkdir "runtime"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" run_app.py
) else (
    start "" pythonw run_app.py
)
exit /b
