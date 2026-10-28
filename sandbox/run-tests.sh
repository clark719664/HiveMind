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
# Experiment S-1 (issue #266) on this CPU class: offline build of the pinned solver, 10 runs per problem, report to stdout.
python /opt/guard/s1-repro.py
