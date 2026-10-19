@echo off
title HiveMind: Neural Breach // Hugging Face Hack-Miner (Free Steam Edition v1.0.0)
echo ============================================================================
echo   HIVEMIND: NEURAL BREACH // HUGGING FACE HACK-MINER (SEASON 1 ^& 2)
echo   Steam AppID: 3133700 | Free-to-Play | Powered by Qwen2.5 4-Bit NF4 + QLoRA
echo ============================================================================
cd /d "%~dp0\..\.."
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -m hivemind.cli.main play-hack-game --season all
) else (
    python -m hivemind.cli.main play-hack-game --season all
)
pause
