@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PORT=8020
if not "%~1"=="" set PORT=%~1

set PY=python
where python >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    pause
    exit /b 1
  )
  set PY=py -3
)

echo ============================================================
echo   Stock live prediction  --  http://127.0.0.1:%PORT%
echo   Trading hours : real-time tracking (3s refresh)
echo   After close   : replays the last trading day's 1-min bars
echo   Press Ctrl+C to stop
echo ============================================================
echo.

start "" "http://127.0.0.1:%PORT%/"
%PY% live.py --port %PORT%

echo.
echo Server stopped.
pause
