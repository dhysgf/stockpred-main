@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo   Starting public share (web.py + cloudflared tunnel)
echo   Close this window or press Ctrl+C to stop everything.
echo.
python tools\serve_public.py %*
echo.
echo   Service stopped.
pause
