@echo off
setlocal
chcp 65001 >nul
set "PROJECT_ROOT=%~dp0.."
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
  "%PROJECT_ROOT%\.venv\Scripts\python.exe" "%~dp0sync_to_external.py" %*
) else (
  echo Create venv first: python scripts\bootstrap_env.py --backend cpu
  exit /b 1
)
exit /b %ERRORLEVEL%
