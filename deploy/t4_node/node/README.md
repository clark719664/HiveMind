# The T4 node's batch layer: restart after preemption, telemetry, and QLoRA training only when idle

Installed on 2026-10-25 at the owner's request. Inference always has priority; this layer must never slow end users down.

## What exists now

| Piece | What it does | State |
| --- | --- | --- |
| Cloud Scheduler job `hm-t4-keepalive` | every 3 min calls the Compute API `start` on this Spot VM (no-op while it runs) | **enabled**, first attempt succeeded |
| GCE `startup-script` metadata | keeps the pre-existing `systemctl restart ssh` line, then runs `/opt/hivemind/t4/bin/startup.sh` in the background | **set** |
| `bin/startup.sh` | idempotent boot orchestrator: ollama + nginx, then the inference producers, then the BOINC layer, then governor + monitor; waits until the gateway lists the tiers; logs to `hivemind_telemetry/startup.log` | installed |
| systemd units (`node/*.service`) | BOINC db / Apache (loopback) / daemons / client, governor, work generator, monitor | **installed, enabled**; db, apache, daemons, client, workgen, governor, monitor are **running** |
| `t4_monitor.py` (root) | every 15 s one JSON line (GPU, load, disk, nginx in-flight, unit states, gateway-visible tiers, governor state, QLoRA progress) in `hivemind_telemetry/t4_node.jsonl`; restarts enabled-but-down units (max 3 per 10 min); writes a `PREEMPTION_NOTICE` event the moment the metadata server announces one | running |
| `t4_governor.py` | polls nginx's loopback `stub_status` once a second; on an in-flight request it sets BOINC's GPU mode to `never` (the task stops and later resumes from its segment checkpoint) and records how long the GPU took to free; after 30 s of quiet, and only with enough free VRAM and a cool card, it sets `always` | running, **held** (see below) |
| `t4_workgen.py` | keeps one chained QLoRA workunit queued; each starts from the latest validated adapter (`latest_adapter.pt`), the validator requires the chain to start there | running; round 1 queued |
| Validator | its GPU replay waits for a quiet moment too (nginx idle for 5 s, max 30 min) | active |

## Training is currently HELD

The governor is deployed but `…/hivemind_telemetry/HOLD` exists, so the BOINC GPU mode stays `never` and no training task runs.
Reason: the yield path has not yet been measured with a real task, and the owner said traffic is arriving. To allow
training: `rm /mnt/disks/disk-socrateai-local-1/hivemind_telemetry/HOLD`. To pause it again: `touch` the same file.

Not yet measured (do not quote numbers for these): the time from "request arrives" to "GPU free" (`gpu_freed_ms` in
`governor.jsonl`), the effect of a running task on inference latency, and the QLoRA loss trend over chained rounds.

## Known limits

- The governor sees nginx in-flight requests only: traffic that bypasses nginx (local clients of Ollama on 127.0.0.1:11434) is invisible to it.
- A yield kills the task process; work since the last segment checkpoint (up to ~30 s) is redone on resume.
- With the full inference mix resident (~10.3 GiB), one task (~3.5 GiB) fits; the validator's replay is serialised behind idle for that reason.
- The older bee/coder producers still run as session-started processes; their units are installed and enabled for the next boot, but the
  watchdog only reports them (use `--manage-legacy` after the cutover described in `../live/README.md`). A teammate session's `@reboot` cron line starts the
  same services, so the cutover (removing that line and stopping the hand-started processes) is to be done together, in a quiet window.
- The BOINC app launchers point at the `/opt/hivemind/t4/src` snapshot (app version 101); re-run `install_node.sh` after code changes.
- Everything is Spot-VM-bound: the Scheduler job restarts the VM; the services come back by the units and `startup.sh`.

## Operating

```bash
/opt/hivemind/t4/bin/boinccmd_node.sh --get_tasks          # the node's BOINC client
tail -f /mnt/disks/disk-socrateai-local-1/hivemind_telemetry/governor.jsonl
cat    /mnt/disks/disk-socrateai-local-1/hivemind_telemetry/latest.json
touch/rm /mnt/disks/disk-socrateai-local-1/hivemind_telemetry/HOLD   # pause / allow training
gcloud scheduler jobs pause hm-t4-keepalive --location=us-east4      # before stopping the VM on purpose
```
