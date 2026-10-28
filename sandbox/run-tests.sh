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
# Upstream's E15 namespace-sandbox tests (scripts/, outside the coverage source) need a mount namespace that can see /tmp
# and /work; both are separate tmpfs mounts here, so unshare's non-recursive bind hides them and the tests fail for the
# container's layout, not the code. They are not run here; upstream CI runs them on an unconfined runner.
echo "NOTE: tests/test_solverbee_{service,mcp,sandbox}.py are not run in this container (E15 namespace sandbox cannot see tmpfs mounts)."
python -m coverage run --source=src/hivemind -m pytest tests/ -v --tb=short \
    --ignore=tests/test_solverbee_service.py --ignore=tests/test_solverbee_mcp.py --ignore=tests/test_solverbee_sandbox.py
python -m coverage report --fail-under=95
