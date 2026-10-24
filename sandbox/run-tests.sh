#!/bin/sh
set -eu
python /opt/guard/isolation-check.py
mkdir -p "$HOME"
cp -a /input/. /work/
cd /work
# Select CPU interpretation before importing Triton's standard kernel helpers.
export TRITON_INTERPRET=1
python -m coverage run --source=src/hivemind -m pytest tests/ -v --tb=short
python -m coverage report --fail-under=95
