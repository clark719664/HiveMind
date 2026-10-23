#!/usr/bin/env bash
# E8-pilot: two real hosts over the public internet (this box + GCP TPU VM gwenlaya-tpu-1, us-west4-a).
# For each seed: start the syncer here, a local learner here, and the remote learner on the TPU VM
# reached through an SSH reverse tunnel (-R 8765:localhost:8765), then a single-host baseline.
# Usage: scripts/run_e8_pilot_wan.sh [seeds...]   (default: 1 2 3)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/xavkal/xdev/AutoevolveAI/.venv/bin/python}"
ZONE="${ZONE:-us-west4-a}"
VM="${VM:-gwenlaya-tpu-1}"
PORT="${PORT:-8765}"
K="${K:-1}"
GRACE="${GRACE:-adaptive}"
export HM_INNER_STEPS="${HM_INNER_STEPS:-8}"
MERGES="${MERGES:-rda direct}"
OUTDIR="${OUTDIR:-$HERE/artifacts/experiments/e8_pilot_runs}"
mkdir -p "$OUTDIR"
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2 3)
for SEED in "${SEEDS[@]}"; do
  for MERGE in $MERGES; do
    OUT="$OUTDIR/wan_seed${SEED}_${MERGE}.json"
    echo "=== seed $SEED merge $MERGE: syncer + local learner + remote learner (WAN) ==="
    PYTHONPATH="$HERE/src" "$PY" "$HERE/scripts/run_experiment_E8_pilot.py" syncer --port "$PORT" --learners 2 --k "$K" \
      --merge "$MERGE" --grace "$GRACE" --rounds 12 --seed "$SEED" --timeout 900 --out "$OUT" > "$OUTDIR/syncer_seed${SEED}_${MERGE}.log" 2>&1 &
    SYNC_PID=$!
    for i in $(seq 1 50); do curl -s -o /dev/null "http://localhost:$PORT/global" && break; sleep 0.2; done
    PYTHONPATH="$HERE/src" "$PY" "$HERE/scripts/run_experiment_E8_pilot.py" learner --syncer "http://localhost:$PORT" \
      --learner-id 0 --host-label "local-$(hostname)" > "$OUTDIR/learner0_seed${SEED}_${MERGE}.log" 2>&1 &
    L0_PID=$!
    gcloud compute tpus tpu-vm ssh "$VM" --zone "$ZONE" \
      --command "cd ~/hm && HM_INNER_STEPS=$HM_INNER_STEPS ~/venv-hm/bin/python run_experiment_E8_pilot.py learner --syncer http://localhost:$PORT --learner-id 1 --host-label gcp-${ZONE}-tpu-vm" \
      -- -R "$PORT:localhost:$PORT" > "$OUTDIR/learner1_remote_seed${SEED}_${MERGE}.log" 2>&1 || echo "remote learner exited non-zero (see log)"
    wait "$L0_PID" || true
    wait "$SYNC_PID" || echo "syncer exited non-zero"
    tail -2 "$OUTDIR/learner1_remote_seed${SEED}_${MERGE}.log" | head -1
    PYTHONPATH="$HERE/src" "$PY" "$HERE/scripts/run_experiment_E8_pilot.py" baseline --learners 2 --merge "$MERGE" --rounds 12 --seed "$SEED" \
      --out "$OUTDIR/baseline_seed${SEED}_${MERGE}.json" | tail -1
  done
done
echo "done: $OUTDIR"
