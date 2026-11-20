@echo off
REM One-click workspace permission repair. Right-click -> Run as administrator,
REM or just double-click: the script raises the UAC prompt itself.
REM ASCII only on purpose - cmd.exe reads .bat in the OEM code page.
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\fix_workspace_permissions.ps1" -Path "%~dp0."
echo.
pause
