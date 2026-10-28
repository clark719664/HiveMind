#!/bin/sh
set -eu
python /opt/guard/isolation-check.py
mkdir -p "$HOME"
cp -a /input/. /work/
cd /work
# Experiment S-1 (issue #266) on this CPU class: offline build of the pinned solver, 10 runs per problem, report to stdout.
# Runs first so its report is in the log even when the project's own suite fails on this container's limits.
python /opt/guard/s1-repro.py
rm -rf /work/target-s1
# Select CPU interpretation before importing Triton's standard kernel helpers.
export TRITON_INTERPRET=1
python -m coverage run --source=src/hivemind -m pytest tests/ -v --tb=short
python -m coverage report --fail-under=95
