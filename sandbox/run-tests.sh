#!/bin/sh
set -eu
python /opt/guard/isolation-check.py
mkdir -p "$HOME"
cp -a /input/. /work/
cd /work
python -m coverage run --source=src/hivemind -m pytest tests/ -v --tb=short
python -m coverage report --fail-under=95
