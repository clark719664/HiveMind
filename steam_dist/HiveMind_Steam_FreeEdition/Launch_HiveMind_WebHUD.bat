@echo off
title HiveMind: Neural Breach // Interactive Cyberpunk Web HUD (Port 8090)
echo ============================================================================
echo   LAUNCHING HIVEMIND CYBERPUNK WEB HUD AT http://127.0.0.1:8090
echo ============================================================================
cd /d "%~dp0\..\.."
start "" "http://127.0.0.1:8090"
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -m hivemind.cli.main serve-hack-hud --port 8090
) else (
    python -m hivemind.cli.main serve-hack-hud --port 8090
)
pause
