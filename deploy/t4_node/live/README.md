# The live T4 (2026-10-25): what runs, what breaks under load, how to make it survive a restart

Observed **read-only** from the VM on 2026-10-25 ~12:00 UTC; nothing below was changed by the author of this page.

## What is running

| Layer | Process | Started by | Survives reboot? |
| --- | --- | --- | --- |
| TLS + routing | `nginx` (`/etc/nginx/conf.d/hivemind-t4.conf`): `/` → `127.0.0.1:8788`, `/bee/` → `127.0.0.1:8789` | systemd | **yes** |
| Coder models | `ollama` (`MAX_LOADED_MODELS=1`, `NUM_PARALLEL=4`, `KEEP_ALIVE=10m`) | systemd | **yes** |
| Coder shield | `hivemind_producer_shield.py --port 8788 --max-concurrent 3` (7B + 3B coder) | a session | **no** |
| Coder producer | `hivemind_producer_agent.py … --model coder=… --model coder-lite=…` (account for the coder tiers) | a session | **no** |
| Bee expert | `llama-server … --lora stars.gguf --alias hivemind-bee-stars -np 1 --port 8294` | a session | **no** |
| Bee shield | `hivemind_producer_shield.py --port 8789 --max-concurrent 3` | a session | **no** |
| Bee producer | `hivemind_producer_agent.py … --model bee-stars=…` (the "bee" account) | a session | **no** |
| Orphan | a shield on `:8787` (`--max-concurrent 1`) from a session's temp directory; nginx does not route to it | a session | no |

GPU: ~10.3 GiB of 15.4 resident (Bee ~4.8, 7B coder ~5.5), 8 vCPUs, load average ~12.7.

## Two risks for a traffic spike

1. **Coder and coder-lite thrash.** One Ollama with `MAX_LOADED_MODELS=1` serves both the 7B and the 3B. Requests that alternate between the two tiers force a model swap each time (a cold load). Beside the Bee expert there is ~5 GiB free, so 7B (5.5) + 3B (2.7) cannot both stay resident. Options: advertise only `coder` (drop `coder-lite` from the agent's `--model` list); or accept the swaps; or stop serving Bee to let both fit. This is a product choice, not a bug.
2. **The VM is Spot with automatic restart off** (`scheduling/preemptible = TRUE`, `automatic-restart = FALSE`). After a preemption it stays down until someone starts it, and even then only nginx and Ollama return: the shields, both producer agents and the Bee server do not, so the gateway keeps routing to nothing. The units in this directory fix the second half; the first needs one of: a small always-on watcher that runs `gcloud compute instances start` when the VM is `TERMINATED`; a managed instance group of size 1 with auto-healing; or standard (non-Spot) provisioning. Each has a cost or an operating burden, so it is the owner's choice.

Also: the Bee slot count is 1 (`-np 1`) behind a shield that admits 3, so Bee requests queue; and CPU load is already high, which the producer processes share.

## Recommended: restart the VM automatically after a preemption (not yet created)

The instance is `SPOT`, `automaticRestart=false`, termination action `STOP`, so a plain `start` brings a preempted VM back.
A Cloud Scheduler job that calls the Compute API every 3 minutes does that with no extra machine (starting a running VM is a
no-op). The Scheduler API is already enabled in the project and no job exists yet. An attempt to create it from the
assistant's session was **blocked by the session's permission classifier** (it is a standing change to the cloud project), so
it has to be run by the owner:

```bash
gcloud scheduler jobs create http hm-t4-keepalive --location=us-east4 --schedule="*/3 * * * *" --time-zone=UTC \
  --uri="https://compute.googleapis.com/compute/v1/projects/gen-lang-client-0625573011/zones/us-east4-b/instances/socreateai-agora-hermes-node1/start" \
  --http-method=POST --oauth-service-account-email=1003063861791-compute@developer.gserviceaccount.com \
  --description="Restart the Spot T4 VM after preemption (no-op while it runs)"
gcloud scheduler jobs run hm-t4-keepalive --location=us-east4      # test once
```

Side effects to accept knowingly: it also undoes a manual stop within 3 minutes (pause it first:
`gcloud scheduler jobs pause hm-t4-keepalive --location=us-east4`; remove it with `... jobs delete ...`); it authenticates as the
VM's default compute service account; and a Spot VM can fail to start while the zone has no capacity (it retries every 3 minutes).
It restores the VM only: the services come back only after the units below are installed.

## Making the stack restart-safe (units are templates; nothing is installed)

1. Create `/etc/hivemind/t4-coder.env` and `/etc/hivemind/t4-bee.env` (`root:root`, mode `0600`) with `HIVEMIND_API_KEY` and `HIVEMIND_ENDPOINT_TOKEN` for each account. The author did not read these values from the running processes and does not have them.
2. `sudo cp deploy/t4_node/live/*.service /etc/systemd/system/ && sudo systemctl daemon-reload`.
3. Cut over **one tier at a time**, in a quiet window: stop the user processes of that tier, `systemctl start` its units in order (bee: llama → shield → agent; coder: shield → agent). Each producer agent heartbeats every 60 s and the gateway only routes to live producers, so expect 1–2 minutes without that tier.
4. Verify with the gateway's `/v1/models` (it lists a tier only while a producer is live) and one small request per tier.
5. `sudo systemctl enable` the units, then reboot-test only when you accept the downtime.

Do **not** run the BOINC validator's GPU replay while the full mix is resident (see `../README.md`).
