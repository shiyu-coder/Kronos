@echo off
rem Scheduled daily research_ops run (registered as "Kronos Daily Research Ops").
rem Runs the full evidence cycle and delivers the condensed brief to Telegram.
if exist "%~dp0RETIRED.json" (
  echo Kronos scanner is retired; scheduled research_ops skipped.
  exit /b 0
)
cd /d "%~dp0.."
if not exist "scanner\logs" mkdir "scanner\logs"
echo [%date% %time%] scheduled research_ops starting >> "scanner\logs\scheduled_research_ops.log"
".\venv\Scripts\python.exe" -m scanner.scheduled_research_ops >> "scanner\logs\scheduled_research_ops.log" 2>&1
set "RESEARCH_OPS_EXIT=%ERRORLEVEL%"
echo [%date% %time%] scheduled research_ops finished (exit %RESEARCH_OPS_EXIT%) >> "scanner\logs\scheduled_research_ops.log"
exit /b %RESEARCH_OPS_EXIT%
