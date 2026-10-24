#!/usr/bin/env bash
# ============================================================================
# Project HiveMind - 1-Click Non-Technical User Launcher (macOS / Linux)
# Operation Polyglot-Singularity-128 (8-Domain Hugging Face & 128-Expert MoE)
#
# NON-MONETARY NOTICE (0.00 EUR / 0.00 USD):
# No financial payment, money, or cryptocurrency is ever offered or paid.
# Symbolic 'points' (PTS / $HIVE) confer exclusively reciprocal computing
# processing benefits when the live production HiveMind Swarm is deployed.
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "============================================================================"
echo "  PROJECT HIVEMIND - 1-CLICK CITIZEN SCIENCE & GPU CAMPAIGN LAUNCHER"
echo "  Operation Polyglot-Singularity-128 (Safe < 1.5 GB VRAM Envelope)"
echo "============================================================================"
echo ""
echo "[LEGAL NOTICE] 0.00 EUR / 0.00 USD: There is NO financial payment or money."
echo "Earned points (PTS / \$HIVE) grant only reciprocal AI computing processing"
echo "benefits when the live production HiveMind Swarm is deployed and shared."
echo ""

DEFAULT_HANDLE="${USER:-citizen-scientist}"
if [ -t 0 ]; then
    read -r -p "Enter your GitHub username or nickname [default: ${DEFAULT_HANDLE}]: " INPUT_HANDLE || true
    HANDLE="${INPUT_HANDLE:-$DEFAULT_HANDLE}"
else
    HANDLE="${1:-$DEFAULT_HANDLE}"
fi

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [ -x "$ROOT_DIR/../../../.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/../../../.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

mkdir -p "$ROOT_DIR/artifacts"
echo ""
echo "[Step 1/3] Running the 8-Domain Hugging Face & 128-Expert Multi-MoE Campaign for @${HANDLE}..."
PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/run_multi_moe_hf_campaign.py" \
    --contributor-handle "$HANDLE" \
    --tier standard | tee "$ROOT_DIR/artifacts/latest_user_campaign_summary.md"

echo ""
echo "[Step 2/3] Running the Season 9 & Season 10 GPU + HiveMind Swarm Campaigns (CAMP-S9/S10 1..4)..."
PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/run_season9_10_gpu_swarm_campaigns.py" \
    --contributor-handle "$HANDLE" | tee -a "$ROOT_DIR/artifacts/latest_user_campaign_summary.md"

echo ""
echo "[Step 3/3] SUCCESS! Copy the Markdown receipts above (or from artifacts/latest_user_campaign_summary.md)"
echo "and paste them into the Community Campaign Hub:"
echo "  👉 https://github.com/xaviercallens/HiveMind/discussions/86"
