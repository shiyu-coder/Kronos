@echo off
for %%I in ("%~dp0..") do set PROJ=%%~fI
cd /d "%PROJ%"
if not exist "%PROJ%\venv\Scripts\python.exe" (
  echo [ERROR] Expected venv at "%PROJ%\venv\Scripts\python.exe"
  echo Run setup_dependencies.bat first.
  exit /b 1
)

call "%PROJ%\venv\Scripts\python.exe" -m scanner.main %*
