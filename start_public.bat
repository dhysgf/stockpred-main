@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 优先用项目内 .venv 的解释器：依赖装在里面，宿主机 Python 被重置也不影响
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

echo.
echo   Starting public share (web.py + cloudflared tunnel)
echo   Python: %PY%
echo   Close this window or press Ctrl+C to stop everything.
echo.

"%PY%" tools\serve_public.py %*
if errorlevel 1 (
  echo.
  echo   [失败] 如果提示缺 pandas/matplotlib，先双击 setup_env.bat 装依赖。
)
echo.
echo   Service stopped.
pause
