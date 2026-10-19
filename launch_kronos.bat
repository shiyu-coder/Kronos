@echo off
title Kronos Financial Predictor
color 0A

set PROJ=C:\Users\Jacob Higgins\projects\kronos-predictor
set PYTHON=%PROJ%\venv\Scripts\python.exe
set STREAMLIT=%PROJ%\venv\Scripts\streamlit.exe

echo ============================================
echo   KRONOS FINANCIAL PREDICTOR
echo   Powered by NeoQuasar/Kronos-base
echo   PyTorch 2.7.0+cpu - CPU inference
echo ============================================
echo.

:: Check if venv exists
if not exist "%PYTHON%" (
    echo ERROR: Virtual environment not found.
    echo Please run install_deps.bat first.
    pause
    exit /b 1
)

:: Check if streamlit is installed
if not exist "%STREAMLIT%" (
    echo ERROR: Streamlit not found in venv.
    echo Running installer...
    "%PROJ%\venv\Scripts\pip.exe" install streamlit yfinance plotly
)

echo Starting Kronos app...
echo Browser will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.

:: Launch streamlit (opens browser automatically)
cd /d "%PROJ%"
"%STREAMLIT%" run kronos_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
