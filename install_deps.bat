@echo off
:: Installs the pinned CPU environment. Pins live in requirements.txt +
:: scanner\requirements-scanner.txt (see REPO_MAP.md). No CUDA torch here.
set PROJ=%~dp0
if "%PROJ:~-1%"=="\" set PROJ=%PROJ:~0,-1%
set PYTHON=%PROJ%\venv\Scripts\python.exe

if not exist "%PYTHON%" (
    echo Creating virtual environment at "%PROJ%\venv"...
    py -3.12 -m venv "%PROJ%\venv" 2>nul || python -m venv "%PROJ%\venv"
)
if not exist "%PYTHON%" (
    echo [ERROR] Could not create venv. Install Python 3.10+ and re-run.
    exit /b 1
)

call "%PROJ%\scanner\setup_dependencies.bat"
