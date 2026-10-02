@echo off
cd /d "%~dp0" || goto failed

docker compose up -d --build
if errorlevel 1 goto failed

docker compose ps
if errorlevel 1 goto failed

pause
exit /b 0

:failed
pause
exit /b 1
