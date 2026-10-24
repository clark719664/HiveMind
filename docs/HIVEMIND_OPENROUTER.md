# HiveMind OpenRouter

HiveMind OpenRouter is a gateway in the style of OpenRouter. Anyone can send chat requests to HiveMind
MoE models through the standard OpenAI chat-completions protocol, from an IDE or any OpenAI-compatible
client. Anyone with a GPU can serve those models and earn **HiveMind points (HMT)** for the tokens they
serve. HMT is spent by consumers to pay for inference.

## Roles

| Role | What you do | What you get |
|---|---|---|
| Consumer | Send chat requests to `hivemind/...` models | Pay HMT per 1,000 tokens, up to your declared maximum price |
| Producer | Run an OpenAI-compatible server and heartbeat it to the gateway | Earn HMT per 1,000 tokens served, less fees |
| Prosumer | Both | Earn and spend from the same balance |

Registering creates an account and returns an API key once. The gateway stores only an HMAC of the key,
so a lost key cannot be recovered; register again with a new handle.

## Lifecycle of HMT

1. **Mint.** HMT enters the system only through an admin grant (`POST /v1/hivemind/admin/grant`). Nothing
   else creates it.
2. **Spend.** A consumer's request is priced at the current market price for its model. Before forwarding,
   the gateway checks that the balance covers the estimated charge. The actual charge is metered from the
   tokens used and settled in one atomic ledger transaction.
3. **Earn.** The producer receives `(1 − f)` of the charge, where `f` is the fee (default `0.10` validator
   share plus `0.05` burn share, so `f = 0.15`). The rest goes to the validator system account and the burn
   system account.
4. **Audit.** Every ledger row is hash-chained to the previous one. `verify_chain` detects any edited row.

## The win-win equilibrium, in plain words

Prices are in HMT per 1,000 tokens. For each model, the gateway solves the market once per epoch.

**Consumers** each declare the most they will pay per 1,000 tokens, `v_j`. They share the model's available
capacity in proportion to what they bid, the Kelly proportional-allocation rule. The market price `ω` is the
single value that satisfies

    Σ_j max(0, 1 − ω / v_j) = 1

Each consumer `j` then gets the share `x_j = max(0, 1 − ω / v_j)`. A consumer with a low maximum price gets
nothing; a consumer with a high one gets a larger share. Notice that `ω` does not depend on how much capacity
is available, so the price is set by what consumers are willing to pay.

**Producers** each have a cost per 1,000 tokens, `c_i`, and a capacity `k_i`. Whatever a producer serves
earns `(1 − f) · ω` per 1,000 tokens after fees. So:

- if `c_i < (1 − f) · ω`, serving your full capacity is the best choice for you;
- if `c_i = (1 − f) · ω`, you are indifferent and supply nothing;
- if `c_i > (1 − f) · ω`, serving loses you money and you supply nothing.

**Why this is win-win.** Each side gets a deal it would not improve by changing unilaterally:

- A consumer pays no more than its declared maximum and gets a share of the capacity it paid for. Its surplus
  is never negative.
- A producer is paid more than its cost, so its profit is never negative, and it cannot gain by offering a
  different amount of capacity.
- Every HMT paid is accounted for: producer revenue, validator fees and burn add up to the payments.

If the market cannot clear this way (fewer than two consumers, or no producer would supply at the price),
the gateway falls back to a **posted price** set just above the cheapest producer's break-even price, with a
5% margin. Consumers whose maximum is at least that price are served. If no consumer is willing to pay it,
the mechanism is `no_trade`. Every equilibrium the gateway publishes carries a certificate that checks these
properties; see `GET /v1/hivemind/market?model=...`. The endpoint needs a Bearer key and is cached per 10 s epoch. Only funded accounts bid, each bid capped at what its balance can pay for one default request (`balance x 1000 / 1024`), so zero-balance sybil accounts cannot move the price. Every request escrows its reserve (`sys:escrow`) before a producer is contacted and is billed at most that reserve.

The win-win property is a property of the posted rules, not a promise about any particular month. It holds
for every epoch the gateway computes, and the certificate is recomputed each time.

## Using HiveMind from your IDE

Register once, then use the API key as the bearer token and the gateway as the OpenAI base URL. Model ids are
`hivemind/<model>`, and `hivemind/auto` picks the live model with the most supply.

Replace `GATEWAY_URL` with your gateway's base URL and `hm-YOUR_KEY` with your API key. Each snippet below
is a starting point. **Verify against the tool's current documentation** before relying on it, because
configuration formats change between releases.

### OpenCode

`opencode.json` (verify against the tool's current documentation):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "hivemind": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "HiveMind OpenRouter",
      "options": {
        "baseURL": "GATEWAY_URL/v1",
        "apiKey": "hm-YOUR_KEY"
      },
      "models": {
        "hivemind/auto": {
          "name": "HiveMind auto"
        }
      }
    }
  }
}
```

### VS Code (Continue extension)

`config.yaml` for the Continue extension (verify against the tool's current documentation):

```yaml
models:
  - name: HiveMind auto
    provider: openai
    model: hivemind/auto
    apiBase: GATEWAY_URL/v1
    apiKey: hm-YOUR_KEY
```

### Hermes and any other OpenAI-compatible client

Most OpenAI-compatible clients accept a base URL and a key. Set them through environment variables
(verify against the tool's current documentation):

```bash
export OPENAI_BASE_URL="GATEWAY_URL/v1"
export OPENAI_API_KEY="hm-YOUR_KEY"
```

You can test the connection with the official OpenAI Python client, which reads these variables:

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY
reply = client.chat.completions.create(
    model="hivemind/auto",
    messages=[{"role": "user", "content": "Say hello."}],
)
print(reply.choices[0].message.content)
```

Responses carry an extra top-level `hivemind` object in the JSON body with the price, the charge, the
producer id, the mechanism and the request id. Requests that exceed your declared maximum price return HTTP 402 with an
OpenAI-style error body, and no charge is made.

## Producer quickstart with Ollama

A producer runs any OpenAI-compatible server. This example uses Ollama with `qwen3:8b`.

1. Pull the model and start Ollama, then point the endpoint at Ollama's OpenAI-compatible API
   (verify against Ollama's current documentation):

   ```bash
   ollama pull qwen3:8b
   ollama serve
   ```

2. Make the endpoint reachable at a public `https` address. The gateway's SSRF guard refuses `http://`
   URLs and any address that is not globally routable. Put a TLS reverse proxy or tunnel in front of the
   local port. Check the setup with `curl https://YOUR_ENDPOINT/v1/models`.

3. Export your producer key and send a heartbeat:

   ```bash
   export HIVEMIND_API_KEY=hm-YOUR_PRODUCER_KEY
   python scripts/hivemind_producer_agent.py \
     --gateway GATEWAY_URL \
     --endpoint https://YOUR_ENDPOINT \
     --model qwen3-8b=qwen3:8b \
     --cost-per-1k 0.4 \
     --capacity-1k 50 \
     --execution-class t4 \
     --once
   ```

   `--model` takes `public_id=local_name`. The public id is what consumers request as `hivemind/<public_id>`.
   The local name is what your server expects. Leave out `--once` to keep heartbeating every
   `--interval-s` seconds (default 60). Heartbeats are how the gateway knows you are live, and live
   producers are the only ones it routes to.

Choose `--cost-per-1k` honestly. If it is above the market price, no requests reach you. If it is below
your real cost, you lose money on every token you serve.

## The public deployment

Live since 2026-10-24 on Cloud Run (project `gen-lang-client-0625573011`, `us-central1`, owner-approved public access):

    https://hivemind-openrouter-1003063861791.us-central1.run.app

Base URL for IDE tools: `https://hivemind-openrouter-1003063861791.us-central1.run.app/v1`. Register with
`POST /v1/hivemind/register` (`{"handle": "...", "role": "consumer|producer|prosumer"}`); the API key is shown once.
New accounts hold 0 HMT. Only the owner can grant HMT, so a new consumer cannot spend until a grant is made.
Health probes use `/health` (Cloud Run's front end swallows `/healthz`).

**HMT are points, not money.** They have no cash value, cannot be withdrawn and are not a cryptocurrency.

**What a producer can see.** A producer receives the full prompt of every request routed to it. Do not send secrets
or private code to `hivemind/...` models. End-to-end privacy is not provided.

**What is not verified yet.** The gateway meters tokens by bytes, so a producer cannot inflate a bill. It does not
check that an answer is *correct*: a dishonest producer can return poor text and still be paid. Spot-check
verification of outputs is task HM-S13-05. Registration is rate limited but has no per-identity cost, so a determined
sybil can create many accounts. Funded-only bidding keeps free accounts from moving the price, but see HM-S13-06.

## GCP deployment notes

The gateway is packaged in `deploy/gcp_openrouter/`:

- `Dockerfile` builds a Python 3.12 image with a checksum-verified Litestream binary and runs as a non-root user.
- `entrypoint.sh` restores the SQLite ledger from a private, versioned GCS bucket if a replica exists, then runs
  the gateway under `litestream replicate`, which streams every committed transaction to the bucket (about one
  second behind) and flushes on shutdown.
- `server.py` builds the app from environment variables and fails at start-up if the secret is missing.
- `requirements.txt` lists the runtime dependencies (fastapi, uvicorn, pydantic, httpx, cryptography).
- `deploy.sh` assembles a clean build context and runs `gcloud run deploy hivemind-openrouter` as a public service
  with a dedicated service account, exactly **one** instance (a second would fork the ledger) and CPU always
  allocated so replication continues between requests.

Environment variables:

| Variable | Purpose |
|---|---|
| `HIVEMIND_ROUTER_SECRET` | HMAC secret, at least 32 bytes. Mounted from Secret Manager. |
| `HIVEMIND_ROUTER_DB` | SQLite path, `/data/router.db` on Cloud Run |
| `HIVEMIND_ROUTER_REPLICA_URL` | Litestream replica, `gcs://<bucket>/router`. Unset means an ephemeral ledger. |
| `HIVEMIND_ROUTER_ADMIN_KEY_HASH` | HMAC of the admin key; enables admin grants when set |
| `HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS` | Right-most `X-Forwarded-For` hops to trust for the registration rate limit (1 on Cloud Run) |

**Durability, measured.** The persistence path was tested twice: locally (database deleted, restored from the
replica, same API key and balance) and live (an account funded on one revision kept its balance after a new revision
rolled out). The recovery point is about one second of transactions on a hard crash. Two instances overlapping for a
few seconds during a revision rollout is the remaining risk, so do not deploy under load. A multi-instance design
needs a shared database (HM-S13-03).

**Known gaps in the SSRF guard.** The gateway resolves a producer's host at registration and again just before every
forward, and it judges IPv6 forms that carry an IPv4 address (mapped, NAT64, 6to4, Teredo, IPv4-compatible) by that
IPv4 address. It does not pin the connection to the checked address, so a DNS answer that changes between the check
and the connect remains possible; closing it needs an egress proxy (HM-S13-07).

The router secret must not change once accounts exist, because account ids and key hashes depend on it.
Rotating it orphans every existing account.
