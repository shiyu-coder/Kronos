@echo off
for %%I in ("%~dp0..") do set PROJ=%%~fI
cd /d "%PROJ%"
if not exist "%PROJ%\venv\Scripts\python.exe" (
  echo [ERROR] Expected venv at "%PROJ%\venv\Scripts\python.exe"
  exit /b 1
)

call "%PROJ%\venv\Scripts\python.exe" -m pip install --upgrade pip
call "%PROJ%\venv\Scripts\python.exe" -m pip install -r requirements.txt
call "%PROJ%\venv\Scripts\python.exe" -m pip install -r scanner\requirements-scanner.txt

echo Setup complete.
