@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title Demo - Movie Night Agent
cls
echo ================================================================
echo   [1/2] Chat demo: request - constraints - recall - rerank
echo ================================================================
echo.
.venv\Scripts\python.exe -m src.agent --demo --verbose
echo.
echo ================================================================
echo   [2/2] Experiments: ranking, group rerank, cold start
echo ================================================================
timeout /t 2 >nul
.venv\Scripts\python.exe eval\run_all.py
echo.
echo Done.
pause
