@echo off
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"

rem pip 需要可写的临时目录，显式指到用户临时目录，避免在项目里留下 pip-unpack-* 垃圾
set "TMP=%LOCALAPPDATA%\Temp\dsh-pip"
set "TEMP=%TMP%"
if not exist "%TMP%" mkdir "%TMP%" >nul 2>&1

echo ============================================================
echo   Setup environment for the ML final project
echo   Folder: %CD%
echo ============================================================
echo.

rem ---------- [1/4] check python 3.12 ----------
echo [1/4] Checking Python 3.12 ...
py -3.12 -c "import sys; print('      found Python', sys.version.split()[0])" 2>nul
if errorlevel 1 (
    echo.
    echo   ERROR: Python 3.12 not found by the "py" launcher.
    echo   Install Python 3.12 from https://www.python.org/downloads/
    echo   and tick "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

rem ---------- [2/4] create venv ----------
echo [2/4] Creating virtual environment .venv ...
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo   ERROR: failed to create .venv
        pause
        exit /b 1
    )
    echo       .venv created
) else (
    echo       .venv already exists, skip
)

set PYEXE=.venv\Scripts\python.exe

rem ---------- [3/4] install packages ----------
echo [3/4] Installing packages (Tsinghua mirror) ...
"%PYEXE%" -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check --no-cache-dir
"%PYEXE%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check --no-cache-dir
if errorlevel 1 (
    echo.
    echo   Mirror failed, trying the official PyPI ...
    "%PYEXE%" -m pip install -r requirements.txt --disable-pip-version-check --no-cache-dir
    if errorlevel 1 (
        echo   ERROR: install failed. Check your network / proxy and run again.
        pause
        exit /b 1
    )
)

rem ---------- [4/4] self test ----------
echo [4/4] Self test ...
"%PYEXE%" -c "import numpy, pandas, sklearn, matplotlib; print('      numpy       ', numpy.__version__); print('      pandas      ', pandas.__version__); print('      scikit-learn', sklearn.__version__); print('      matplotlib  ', matplotlib.__version__)"
if errorlevel 1 (
    echo   ERROR: some package cannot be imported
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   DONE
echo   Run your script with:
echo       .venv\Scripts\python.exe your_script.py
echo ============================================================
echo.
pause
