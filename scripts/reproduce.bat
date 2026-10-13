@echo off
REM 天玑 TIANJI · 一键复现（Windows，无需 Docker）
REM
REM   scripts\reproduce.bat           完整复现
REM   scripts\reproduce.bat --quick   只跑测试 + 门禁 + 最小示例
REM
REM 依赖：Python 3.10+。全程不污染系统环境（使用本地 .venv）。

setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

set PY=python
set VENV=.venv
set QUICK=0
if "%1"=="--quick" set QUICK=1

echo.
echo === 1/6 创建虚拟环境 ===
if not exist "%VENV%\Scripts\python.exe" (
    %PY% -m venv %VENV%
)
call "%VENV%\Scripts\activate.bat"
python -m pip install -q --disable-pip-version-check -r requirements.lock.txt
python -c "import numpy; print('numpy', numpy.__version__)"

echo.
echo === 2/6 单元测试 ===
python -m pytest tests/ -q --no-header
if errorlevel 1 goto :fail

echo.
echo === 3/6 交付门禁（10 道） ===
python -m tianji.cli gates
if errorlevel 1 goto :fail

echo.
echo === 4/6 最小可运行示例 ===
python scripts\quickstart.py
if errorlevel 1 goto :fail

if "%QUICK%"=="1" (
    echo.
    echo 完成（--quick 模式）
    exit /b 0
)

echo.
echo === 5/6 训练（120 步，约 15 秒） ===
python scripts\train.py --steps 120 --d-model 64 --layers 4 --seq 32 --batch 8 --out runs\reproduce
if errorlevel 1 goto :fail

echo.
echo === 6/6 推理 ===
python scripts\generate.py --ckpt runs\reproduce\model.npz --tokenizer runs\reproduce\tokenizer.txt --prompt "to be or not" -n 50 --temperature 0.7
if errorlevel 1 goto :fail

echo.
echo 全部复现步骤通过
echo 产物：runs\reproduce\{model.npz, model.npz.meta.json, tokenizer.txt, train_log.json}
exit /b 0

:fail
echo.
echo [失败] 复现脚本在第 %ERRORLEVEL% 步出错
exit /b 1
