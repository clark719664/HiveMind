@echo off
title HiveMind Free Steam RTX Tool v1.0.0 (AppID: 3133700)
echo ============================================================================
echo   HIVEMIND FREE STEAM RTX TOOL ^& HACK-MINER v1.0.0 (AppID: 3133700)
echo   Zero-Intrusion RTX GPU Swarm Utility ^| 4-Bit NF4 QLoRA ^| Sparse MoE
echo ============================================================================
cd /d "%~dp0\..\.."
if "%1"=="--headless-check" (
    if exist ".venv\Scripts\python.exe" (
        .venv\Scripts\python.exe -m hivemind.cli.main hardware-check
    ) else (
        python -m hivemind.cli.main hardware-check
    )
    exit /b 0
)
echo [1/2] Running Local RTX GPU & Win32 Zero-Intrusion Daemon Check...
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -m hivemind.cli.main hardware-check
    echo [2/2] Launching Interactive Cyberpunk RTX Control HUD at http://127.0.0.1:8090 ...
    start "" "http://127.0.0.1:8090"
    .venv\Scripts\python.exe -m hivemind.cli.main serve-hack-hud --port 8090
) else (
    python -m hivemind.cli.main hardware-check
    start "" "http://127.0.0.1:8090"
    python -m hivemind.cli.main serve-hack-hud --port 8090
)
pause
