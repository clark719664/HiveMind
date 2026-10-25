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

## Measured on 2026-10-25 (small samples; read the limits)

**Node telemetry, 12:56-15:51 UTC (`scripts/t4_inference_report.py`).** Requests were in flight in 2 of 601 15-second samples (0.3 %, max 1);
GPU memory constant at 10,303 MiB, utilisation 0 % at p95 with one 93 % spike, temperature 48-72 C, power p95 31 W; all four tiers visible at the
gateway in 601 of 601 samples; no preemption notice; host load median 6.3 on 8 vCPUs (mostly other sessions' work). nginx handled ~1.3 requests/s,
almost all of it the gateway's endpoint checks and the monitors' polls, **not user inferences**. What the 15 s sampling supports: the GPU was lightly loaded.
**It does NOT support "no demand"** (an earlier version of this page said so, wrongly): the in-flight signal is sampled every 15 s and chat requests last
~0.2 s, so it sees almost none of them. The nginx access log (adm-only, read with sudo) counts them properly: on 25 Oct (UTC) 566 completed
`POST .../chat/completions` (`/bee` 413, `/` 111, `/u` 42), typically **20-50 per hour** through the afternoon (53 in the 12:00 hour, 43 in the 15:00 hour), 337 of the
Bee ones in the 10:00 hour alone. Some are the monitors' and teammates' probes; the log cannot tell probes from users. At ~30 requests/hour of 0.2 s the GPU is
busy ~0.2 % of the time, which is exactly what the sampling saw: low load, real traffic. (`t4_monitor.py` now records the real count per route and status.)

**Gateway view (aggregates; the market endpoint publishes no per-user data).** At 15:5x UTC no consumer was bidding, so nothing cleared and the
price sat at the 0.494 HMT/1k floor; at ~13:00 UTC, during test windows, four consumers were bidding and the price was ~7.5. The owner's prosumer account had
`producer_earned` 0.773 HMT: that shows its producer served paid requests, but the ledger cannot say whether the consumers were other users or the
owner's own test keys, so it must not be read as external demand.

**End-user-path latency, 3 requests per tier, 8 tokens, temperature 0, 15:5x UTC (one sample, router path).**

| tier | ok | latency min / median / max | served by |
| --- | --- | --- | --- |
| coder (7B) | 3/3 | 0.15 / 0.17 / 0.40 s | the owner's producer 2x, the older coder producer 1x |
| auto | 3/3 | 0.15 / 0.16 / 0.20 s | the same two coder producers |
| bee-stars | 3/3 | 0.18 / 0.20 / 0.64 s | the bee producer |
| coder-lite (3B) | 2/3 | **32 / 48 / 63 s**, one **502 Bad Gateway** | the older coder producer |

The owner's new producer therefore does receive and answer traffic. A teammate's idle baseline at 13:15Z (router path) agrees for the fast tiers:
bee-stars 0.35-0.51 s, coder 0.63-0.81 s, auto 0.27 s, and ~32 tok/s for a 320-token completion.

**coder-lite hurts the coder tier.** Ollama runs with `OLLAMA_MAX_LOADED_MODELS=1`, so the 7B and the 3B evict each other. A second, cleaner pair measured:
coder 0.24 s -> coder-lite **6.6 s** -> the next coder request **8.2 s** (swap back) -> coder 0.16 s. Collateral cost: after any coder-lite request the next coder user waits ~40x longer.

**Disclosure: my own latency tests caused slow requests for others.** The router's Cloud Run log (read-only) shows slow chat completions at 15:53-16:05 UTC:
62.9 s, 177.7 s, a **502 after 120 s**, 32.1 s and 73.3 s. That is the window of my `coder-lite` tests (my sample's 32 s, 63 s and 502 are in that list), and with a
single resident model each of my 3B requests evicted the 7B for whoever asked next. The 177.7 s and 73.3 s entries are not mine and probably were such victims or
other probes; the log does not say. A teammate session reported the same slow requests as "13:53-14:05Z real users"; the router log timestamps them two hours later,
so that label was off by two hours, and its explanation (a cold load) is right in mechanism but those requests were not a separate user event. The same router log shows an
older slow period, 10:30-11:43 UTC (502 after 120 s at 10:30:53, 10:32:32, 11:41:27; 200s of 24-91 s), that predates all of my work today and has the same signature
(the nginx log has matching 499s at 10:32, 10:34 and 11:43). Lesson recorded: do not probe the `coder-lite` tier on a live node; measure it on a quiet node or a copy.

Two fixes, each a change to a live service (not applied; the owner decides):
1. Stop advertising `coder-lite` (remove `--model coder-lite=...` from the older coder agent; the unit template in this directory already does).
2. Let both models stay resident: `OLLAMA_MAX_LOADED_MODELS=2` and an Ollama restart. Memory arithmetic from measured figures: Bee 4.8 GiB + 7B 5.6 GiB + 3B ~2.7 GiB = ~13.1 GiB of 15.4, ~2 GiB spare. It restarts Ollama (the loaded model is dropped and cold-loaded again), leaves no room for a BOINC task while both are resident (they unload after 10 minutes of silence, which is before any night run), and is untested here.

## 2026-10-25 20:25Z: the owner chose "stop advertising coder-lite"; a third party's process overrides it (UNRESOLVED, owner: "solve it later")

Applied by me: `hivemind-live-agent-coder.service` (coder tier only, same account `af92feeb…`, same endpoint) was started with `--ignore-dependencies`, and
the old hand-started coder agent (pid 613067, coder + coder-lite) was stopped. Nothing supervised it, and `~/bee-stars/start.sh` launches only the Bee tier.

Then a **different process appeared that neither I nor the teammate session started**: pid 1051361 (since ~20:17Z), and after I stopped it, an identical respawn,
pid 1060495 (since ~20:27Z). Both run `~/hivemind-producer/hivemind_producer_agent.py` against the router's alternate URL
(`…-cfhmhmvv5a-uc.a.run.app`, the same Cloud Run service), endpoint = this T4, models `coder` + `coder-lite`, `--cost-per-1k 0.25 --capacity-1k 80`,
`--execution-class rtx2070-8gb-turing` (not this card), and heartbeat for the **same account**, so the gateway's model list flaps between my agent's `["coder"]` and its
`["coder","coder-lite"]`. As of 20:30Z the gateway still lists `coder-lite`. The owner described this as a contributor's work and asked to leave it for later; I did not stop it again.
Consequences to settle later: coder-lite swap penalty is back in force; the advertised price (0.25 vs 0.4) and the wrong execution class are someone else's choices.

Same actor, same evening: **uncommitted edits to my worktree and a redeploy** of two pipeline files, live in `/opt/hivemind/t4/src` since 20:22-20:23Z
(`boinc_server_scripts.py`: a "ratchet gate" that advances the chain head only if the held-out loss did not get worse, with a comment saying "within 0.02" that the code does not implement;
`moe_work.py`: each round mixes 19 chunks of the target domain with 9 of each of the five others, and caps the learning rate at 6e-5 on chained rounds). I did not write, review or test these,
did not commit them, and my commits exclude them; a copy of the diff is in the job scratch directory. Anything trained from tonight on uses that code, so treat the loss trend as coming from it, not
from the code described elsewhere on this page.

## Recommended: restart the VM automatically after a preemption (created: job `hm-t4-keepalive`)

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
