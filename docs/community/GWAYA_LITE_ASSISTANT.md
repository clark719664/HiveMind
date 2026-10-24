# GWAYA-Lite assistant: a small-first coding ladder behind a fail-closed gate (CLI + localhost gateway)

> Project HiveMind is a voluntary, non-commercial research project. Nothing here involves money. Any recognition for contributions is non-monetary
> reciprocal compute accounting, as described in `LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`.

`scripts/gwaya_lite.py` is the deployable form of what experiment E16 measured (`artifacts/experiments/E16_gwaya_lite_t4_report.json`,
pre-registration `E16_gwaya_lite_t4_prereg.json`, owner guide `GWAYA_T4_DEPLOYMENT.md`). One stdlib-only file, three subcommands: `ask`, `serve`, `doctor`.

## What it is, and what it is not

* It asks a small local model (Qwen2.5-Coder 1.5B, then 3B, then 7B, Q4_K_M, served by your local Ollama daemon) for Python code and **checks every candidate** with the
  gate from `scripts/e15_gate.py`: syntax, the vendored GWAYA zero-stub audit (no `pass`/`...`/`NotImplementedError` bodies, no `mock_*`/`fake_*` names), then your tests in a sandbox.
  It escalates to the next tier **only** when the whole loop of the current tier (greedy, two samples, up to three repair rounds fed with the gate feedback) found nothing that passes.
* It is **not** a correctness oracle. In the E16 measurement 5-7% of `VERIFIED` solutions still failed the hidden tests (precision of VERIFIED 92.6% / 92.8% / 94.8% for 1.5B / 3B / 7B).
  `VERIFIED` means "audit clean and *your* tests passed in the sandbox". Write tests that matter to you.
* It is **not** GWAYA's Bubblewrap oracle. The sandbox is the E15 namespace sandbox (user/mount/pid/net namespaces, tmpfs over `$HOME`, no network, all capabilities dropped,
  rlimits). It has **no seccomp filter**, shares the host kernel and is not a security boundary against a determined attacker. Do not run it against hostile prompts as a service.
* It is **not** tested in a real editor (see section 5), supports Python only, and answers single tasks (the gateway uses the last user message; earlier chat turns are ignored).
* It never runs a model in the cloud and sends nothing off the machine: the Ollama client refuses non-local URLs and ignores proxy variables.

## 1. Install (no pip)

Needs: Linux, Python 3.10+, the Ollama daemon on `127.0.0.1:11434`, a system `python3` outside `$HOME` and unprivileged user namespaces (the E15 gate requirements; see
`GWAYA_CODING_ASSISTANT.md`). Nothing to install from PyPI. Pull the models yourself (owner action):

```bash
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5-coder:7b-instruct
python3 scripts/gwaya_lite.py doctor     # one-screen status: Ollama, models, sandbox self-test, free GPU memory, the load/unload plan
```

## 2. `ask`

```bash
# no tests: the best verdict is AUDITED
python3 scripts/gwaya_lite.py ask "Write a function is_palindrome(s) ignoring case"

# with tests (single-line asserts): the best verdict is VERIFIED
python3 scripts/gwaya_lite.py ask "Write a function add_two(a, b) returning the sum" --test 'assert add_two(1, 2) == 3' --test 'assert add_two(-1, 1) == 0'
python3 scripts/gwaya_lite.py ask "Parse a CSV line into a list of ints" --tests my_asserts.txt --ladder 3b,7b --ctx 4096 --max-rounds 3 --json
```

Flags: `--tests FILE` (one `assert ...` per line, `#` comments allowed), `--test 'assert ...'` (repeatable), `--ladder 1.5b,3b,7b`, `--ctx 4096`, `--max-rounds 3`,
`--max-tokens 512` (cap 1024), `--max-resident-mib 7500`, `--keep-warm`, `--json`, `--ollama-url` (local only). The first test is shown to the model in the prompt (as in E16);
all tests gate. The code goes to stdout, followed by a PROVENANCE block: tier used, candidates tried, repair rounds, generated tokens, seconds, the gate result, the sandbox note and the E16 caveat sentence.

## 3. `serve` (localhost OpenAI-compatible gateway)

```bash
python3 scripts/gwaya_lite.py serve                       # http://127.0.0.1:8099, loopback only, no auth
python3 scripts/gwaya_lite.py serve --port 8099 --ladder 3b,7b --ctx 4096
```

| Endpoint | Behaviour |
|---|---|
| `GET /health` | `{"status":"ok", ladder, ctx, auth}` |
| `GET /v1/models` | one model, `gwaya-lite` |
| `POST /v1/chat/completions` | OpenAI schema. `stream:false` returns a `chat.completion`; `stream:true` returns SSE (`chat.completion.chunk` frames of about 40 characters, then `data: [DONE]`). The model call finishes before the stream starts, so streaming only chunks the final text. |

* **Tests**: put a fenced block in the last user message, it is stripped from the model prompt and used as the gate tests:

  ````
  Write add_two(a, b) returning the sum.
  ```tests
  assert add_two(1, 2) == 3
  assert add_two(2, 2) == 4
  ```
  ````

  or send `"gwaya": {"tests": ["assert ..."], "ladder": ["3b", "7b"]}` in the body. Only single-line `assert` statements are accepted (anything else is a 400).
* **Reply**: `choices[0].message.content` is the final code in a ```` ```python ```` block plus a short provenance footer; the top-level `gwaya` field carries `verdict`, `tier`, `candidates`, `repair_rounds`, `seconds`, `generated_tokens`, `gate_verdict`, `tests`, `caveat`.
  A `FAILED` or `UNVERIFIED` reply still shows the best attempt, labelled as not accepted. Always read the verdict, not just the code.
* **Limits**: request bodies 64 KiB (413 above), `max_tokens` capped at 1024 (default 512), prompts 16 000 characters, 100 tests. Requests are **serialised by a lock** (one GPU job at a time, bounded memory); concurrent clients wait.
* **Binding and auth**: the default is `127.0.0.1` with **no authentication**, so any local user or process on the machine can use it (and spend your GPU); do not forward the port. A non-loopback bind is
  refused unless both `--allow-nonlocal` and `--api-key-file FILE` (a Bearer token of at least 16 characters) are given; Bearer auth is then required on every route. On a loopback bind the gateway also
  rejects requests with a non-loopback `Host` (DNS rebinding) or `Origin` (a web page in your browser posting to localhost). There is no TLS; do not send the token over an untrusted network.
* **Logging**: metadata only (verdict, tier, counts, seconds). Prompts, tests and code are logged only with `--log-prompts`.
* Ollama failures return 502; a model that is not installed returns 502 with its `ollama pull` command.

### An OpenAI-compatible client (no SDK)

```python
import json, urllib.request
body = {"model": "gwaya-lite", "messages": [{"role": "user", "content":
        "Write add_two(a, b) returning the sum.\n```tests\nassert add_two(1, 2) == 3\n```"}]}
req = urllib.request.Request("http://127.0.0.1:8099/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
out = json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=600))
print(out["gwaya"]["verdict"], out["gwaya"]["tier"]); print(out["choices"][0]["message"]["content"])
```

```bash
curl -s --noproxy '*' http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write square(n).\n```tests\nassert square(3) == 9\n```"}],"stream":false}'
```

## 4. Verdicts

| Verdict | Meaning | Exit code | When |
|---|---|---|---|
| `VERIFIED` | audit clean **and** the user-supplied tests passed in the sandbox | 0 | tests given, a candidate passed the gate |
| `AUDITED (syntax + zero-stub audit only; NOT tested)` | no tests: syntax parsed and the zero-stub audit was clean; nothing about behaviour is claimed | 0 | no tests given |
| `UNVERIFIED` | the gate or sandbox could not run (fail-closed); nothing was checked and the tool says so | 2 | sandbox unavailable or disabled, auditor error |
| `FAILED` | no candidate passed the gate after the whole ladder; the last error is shown and the code shown is not accepted | 1 | every tier exhausted |
| (usage / environment error) | bad flags, Ollama down, model not installed, refused bind | 3 | before any verdict |

Every `VERIFIED`/`AUDITED` answer carries the E16 sentence: 5-7% of VERIFIED solutions failed hidden tests, so VERIFIED is never "correct".

## GPU budget (8 GB cards)

Planning numbers are read from the E16 report at run time (`probes[].measured_mib`, `nvidia-smi` memory delta on a Tesla T4, Ollama 0.34.1, Q4_K_M, 100% on GPU), not typed by hand:

| Tier | num_ctx 4096 (MiB) | num_ctx 8192 (MiB) | PR #21 estimate at 4096 / 8192 (MiB) |
|---|---|---|---|
| 1.5B | 1651 | 2107 | 1782 / 1894 |
| 3B | 2713 | 3297 | 2584 / 2728 |
| 7B | 5483 | 6387 | 5290 / 5514 |

* `--max-resident-mib` (default 7500) bounds the sum of the resident Ollama models. Before loading a tier the tool unloads other resident models, largest first, until the sum plus the new tier fits
  (for example the 7B at 4096 beside the 1.5B is 7134 MiB and stays; the 7B beside the 3B would be 8196 MiB, so the 3B is unloaded first). A tier that alone exceeds the budget is refused. The active tier gets `keep_alive 5m`.
* Everything is unloaded on exit, on an exception, on SIGTERM/SIGINT and when `serve` shuts down (`ask --keep-warm` is the only exception). If the host card is shared, check `doctor`: it prints free GPU memory.
* Other values of `--ctx` use the 4096 value plus 28/36/56 KiB of KV cache per token (1.5B/3B/7B). 7B decode measured 28.5 tok/s at 4096.

Measured effect (E16, MBPP-sanitized n = 257, one T4): the loop raised pass@1 by 8.6 / 7.4 / 3.9 points for 1.5B / 3B / 7B and never lowered it; the ladder reached 85.2% against 85.6% for the 7B loop alone at about 78% of the model time.

## 5. Editor recipes (CONFIG EXAMPLES ONLY: not tested in a real editor in this environment)

These are settings shapes for an "OpenAI-compatible provider"; field names change between versions, check your tool's current documentation. Use the gateway for chat/edit tasks, not for tab autocomplete
(a gate loop takes seconds). To get a verdict other than `AUDITED`, the user message must contain a ```` ```tests ```` block, which editors do not add on their own.

* **VS Code Continue** (`config.yaml` models entry, shape only):
  ```yaml
  models:
    - name: GWAYA-Lite
      provider: openai
      model: gwaya-lite
      apiBase: http://127.0.0.1:8099/v1
      apiKey: not-needed-on-loopback
  ```
* **Cursor**: Settings, Models, add an OpenAI-compatible model: base URL `http://127.0.0.1:8099/v1`, model name `gwaya-lite`, any non-empty API key. Cursor routes some requests through its own servers,
  which cannot reach `127.0.0.1`; whether your version can use a localhost endpoint at all is unverified here.
* **Open WebUI**: Admin Settings, Connections, OpenAI API: URL `http://127.0.0.1:8099/v1`, key `none`. If Open WebUI runs in Docker, `127.0.0.1` is the container; reaching the host needs a non-loopback bind,
  which requires `--allow-nonlocal --api-key-file` and a firewall rule limiting the source.

## 6. Limitations

* One runtime (Ollama), Q4_K_M, one benchmark (MBPP), one T4; numbers on your card will differ. The stub audit and sandbox are the E15 ones, with the gaps listed above.
* The gate only knows the tests you give it. With one weak assert the precision is about what E16 measured (92.6-94.8%); with none you get `AUDITED`, which says nothing about behaviour.
* Module-level code of a candidate also runs once in the sandbox (the tests harness executes it). Greedy decoding on Ollama is stable in E16's repeat check but not guaranteed.
* Serialised requests: a 7B escalation holds the lock for tens of seconds. There is no queue limit besides the HTTP server's own threads.
* Prompts are limited to 16 000 characters, tests to single-line asserts. No streaming of tokens while the model works.
* If the sandbox is unavailable every verdict is `UNVERIFIED`; do not "fix" that by disabling checks (`E15_GATE_DISABLE=1` exists only for tests).

## 7. Five small community contribution ideas

1. Run `doctor` and a ten-task `ask` session on another 8 GB card (RTX 3060 8 GB, 4060) and send the output together with the E16-style memory numbers.
2. Add a `--tests-from-docstring` helper that turns `>>> f(1)` doctest examples in the prompt into assert lines, and measure how often it raises VERIFIED precision.
3. Write a small editor extension or shell function that appends a ```` ```tests ```` block from a selected test file to the chat message.
4. Replace the lock with a bounded queue (HTTP 429 when full) and add a `/v1/queue` status route, with a test using the fake Ollama.
5. Port the sandbox probes to Bubblewrap on a machine that has `bwrap` and report where verdicts differ from the E15 sandbox on the same candidates.
