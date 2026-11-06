@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 一次性安装依赖到项目内 .venv（之后双击 start_public.bat 即可）
rem 为什么不用全局 Python：宿主环境会被重置，实测隔夜 pandas/numpy/sklearn/matplotlib 全丢。

echo.
echo   [1/2] 创建 .venv （已存在则跳过）
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 (
    echo   创建失败：没找到 python，请先装 Python 3.11+ 并加入 PATH。
    pause
    exit /b 1
  )
)

echo   [2/2] 安装依赖（首次约 3~8 分钟，取决于网速）
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt --retries 3 --timeout 60
if errorlevel 1 (
  echo.
  echo   安装失败。可重试：".venv\Scripts\python.exe" -m pip install -r requirements.txt
  pause
  exit /b 1
)

echo.
echo   依赖就绪。验证：
".venv\Scripts\python.exe" -c "import pandas,numpy,sklearn,matplotlib,snownlp;print('  pandas',pandas.__version__,'| numpy',numpy.__version__,'| sklearn',sklearn.__version__,'| matplotlib',matplotlib.__version__)"
echo.
echo   下一步：双击 start_public.bat 启动对外服务。
pause
