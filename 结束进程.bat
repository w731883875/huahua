@echo off
chcp 65001 >nul
setlocal

echo ============================================
echo   Stop Huahua Assistant and Torchlight
echo ============================================
echo.

echo [1/3] pythonw.exe (Huahua Assistant / Desktop Pet)
taskkill /F /IM pythonw.exe /T >nul 2>&1
if errorlevel 1 (echo    not running) else (echo    stopped)

echo [2/3] torchlight_infinite.exe
taskkill /F /IM torchlight_infinite.exe /T >nul 2>&1
if errorlevel 1 (echo    not running) else (echo    stopped)

echo [3/3] verify
tasklist /FI "IMAGENAME eq torchlight_infinite.exe" | find /I "torchlight" >nul
if errorlevel 1 (echo    game: gone) else (echo    game: STILL RUNNING - run this file as administrator)
tasklist /FI "IMAGENAME eq pythonw.exe" | find /I "pythonw" >nul
if errorlevel 1 (echo    huahua: gone) else (echo    huahua: STILL RUNNING - run this file as administrator)

echo.
echo Done.
pause
