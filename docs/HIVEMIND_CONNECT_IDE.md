# Connect your editor to the HiveMind OpenRouter

HiveMind speaks the standard OpenAI chat protocol, so any editor or tool that accepts a custom OpenAI-compatible
provider can use it. You need three things: a **base URL**, an **API key** and a **model id**.

| | Value |
|---|---|
| Base URL | `https://hivemind-openrouter-1003063861791.us-central1.run.app/v1` |
| API key | `hm-...` (shown once when you register) |
| Model id | `hivemind/coder` (7B, everyday Python), `hivemind/coder-lite` (3B, quick edits), `hivemind/auto` (busiest live model) |

Which models are live changes with the producers online. Ask the gateway instead of trusting this page:

```bash
curl -s https://hivemind-openrouter-1003063861791.us-central1.run.app/v1/models
```

On 2026-10-24 it listed `hivemind/auto`, `hivemind/coder` and `hivemind/coder-lite`. The larger tiers in
`/v1/hivemind/catalog` (`coder-plus`, `moe-general`, `moe-coder`) have no producer yet (roadmap item `HM-S13-14`), so
asking for one returns an error.

> **Read this first.** HMT are points with no cash value. A producer sees the full text of every prompt routed to it, so
> do not send secrets or private code. The gateway does not check that an answer is correct (`HM-S13-05`). New accounts
> start at 0 HMT and only the owner can grant points, so you may need a grant before your first request.

## 1. Get an API key

Pick one.

- **In the browser.** Open the gateway's home page. It has a live console, a balance check and copy-paste snippets that
  already contain your gateway address.
- **From a terminal** (the key is printed once; keep it in an owner-only file):

  ```bash
  python scripts/hivemind_connect.py register --handle your-handle --role consumer --save-key ~/.hivemind_key
  export HIVEMIND_API_KEY=$(cat ~/.hivemind_key)
  ```

- **With Google** (`/signup`) once the operator has set a Google client id; see [Sign in with Google](HIVEMIND_OPENROUTER.md).

Check the key, your balance and the live models:

```bash
python scripts/hivemind_connect.py check        # reads HIVEMIND_API_KEY
```

or with cURL:

```bash
curl -s -H "Authorization: Bearer $HIVEMIND_API_KEY" https://hivemind-openrouter-1003063861791.us-central1.run.app/v1/hivemind/me
```

## 2. Print a ready-made snippet

`scripts/hivemind_connect.py snippet <tool>` prints a configuration with your gateway address filled in. Tools:
`opencode`, `continue`, `cline`, `aider`, `zed`, `jetbrains`, `python`, `env`.

```bash
python scripts/hivemind_connect.py snippet opencode --model coder
```

The sections below show the same settings by hand. Editor settings change between releases, so check each tool's own
documentation if a field is rejected. Status of each one on our side:

| Tool | Status |
|---|---|
| OpenCode | **Tested live** against the public gateway with a Tesla T4 producer (chat answers; see the tool-call note) |
| OpenAI Python client, cURL | **Tested live** |
| VS Code (Continue, Cline), JetBrains, Zed, Aider | Snippets generated and format-checked by our tests, **not yet run against a real editor** |

## OpenCode (tested)

Save as `opencode.json` in your project, or in `~/.config/opencode/opencode.json` for all projects.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "hivemind": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "HiveMind OpenRouter",
      "options": {
        "baseURL": "https://hivemind-openrouter-1003063861791.us-central1.run.app/v1",
        "apiKey": "hm-YOUR_KEY"
      },
      "models": {
        "hivemind/coder": {
          "name": "HiveMind coder",
          "limit": { "context": 8192, "output": 700 }
        }
      }
    }
  }
}
```

Then start `opencode`, run `/models` and choose **HiveMind coder**.

Things we hit while testing, and what to do:

- **Keep the `limit` block.** Without an output limit OpenCode asks for a very long answer. The gateway reserves points for the
  whole possible answer, so a small balance is refused (a 50-point account needed about 123 per request).
- **The 7B model often writes a tool call as plain JSON text** instead of calling the tool, so file edits and shell commands
  do not run. Use it for questions, explanations and code you paste in yourself, or turn tools off for this provider.
- **Keep prompts short.** The default context of the serving GPU is 4,096 tokens and OpenCode's own prompt is about 2,000.
- Several OpenCode requests can arrive at the same moment (a title and the answer). A producer that allows only one request at a
  time answers 502; producers should allow at least 2.

## VS Code

### Continue extension

Open the Continue config (`config.yaml`, command palette: *Continue: Open Config*) and add:

```yaml
models:
  - name: HiveMind coder
    provider: openai
    model: hivemind/coder
    apiBase: https://hivemind-openrouter-1003063861791.us-central1.run.app/v1
    apiKey: hm-YOUR_KEY
    roles: [chat, edit, apply]
    defaultCompletionOptions:
      maxTokens: 700
```

Do not assign HiveMind to the `autocomplete` role: there is no inline-completion endpoint yet (`HM-S13-10`).

### Cline or Roo Code

Settings, then: API Provider **OpenAI Compatible**; Base URL the address above; API Key your key; Model ID `hivemind/coder`;
maximum output tokens 700 if offered. Cline relies on tool calls, so expect the plain-text tool-call problem described above.

## JetBrains (PyCharm, IntelliJ)

Either install the **Continue** plugin and use the YAML above, or add an OpenAI-compatible provider under
*Settings, Tools, AI Assistant, Models* with the base URL, key and model id from the table.

## Zed

In `settings.json`:

```json
{
  "language_models": {
    "openai_compatible": {
      "HiveMind": {
        "api_url": "https://hivemind-openrouter-1003063861791.us-central1.run.app/v1",
        "available_models": [
          { "name": "hivemind/coder", "display_name": "HiveMind coder", "max_tokens": 32768, "max_output_tokens": 700 }
        ]
      }
    }
  }
}
```

Enter the key in Zed's provider settings.

## Terminal tools and scripts

```bash
# Aider
export OPENAI_API_BASE=https://hivemind-openrouter-1003063861791.us-central1.run.app/v1
export OPENAI_API_KEY=hm-YOUR_KEY
aider --model openai/hivemind/coder

# Anything that reads the standard variables
export OPENAI_BASE_URL=https://hivemind-openrouter-1003063861791.us-central1.run.app/v1
export OPENAI_API_KEY=hm-YOUR_KEY
```

```python
from openai import OpenAI
client = OpenAI()  # reads the two variables above
reply = client.chat.completions.create(
    model="hivemind/coder", max_tokens=700,
    messages=[{"role": "user", "content": "Write clamp(x, lo, hi) in Python."}],
)
print(reply.choices[0].message.content)
print(reply.model_dump().get("hivemind"))  # price, charge and producer of this call
```

## Check that the code is right: verified mode

A 7B model is wrong some of the time. `scripts/hivemind_verified.py` asks the router for code, runs **your** tests on **your**
machine in a sandbox, sends failures back for up to three repairs and prints only code that passed:

```bash
python scripts/hivemind_verified.py --task "Write clamp(x, lo, hi)" --test "assert clamp(5,0,3)==3" --test "assert clamp(-1,0,3)==0"
```

On 257 held-out MBPP-sanitized tasks the 7B tier passed 76.3% on the first attempt and 82.9% with this loop. Without a
sandbox the result is `BLOCKED`, never "verified". Details: [Python and game developers](HIVEMIND_PYTHON_GAME_DEVS.md).

## Troubleshooting

| You see | Cause and fix |
|---|---|
| `401` | Wrong or missing key. The key starts with `hm-`; the header is `Authorization: Bearer hm-...`. |
| `402` "below the ... reserve" | Balance too low for the answer you allowed. Lower `max_tokens` (the `limit.output` above) or ask for a grant. |
| `402` price above your maximum | Raise it with `POST /v1/hivemind/consumer/max-price`. No charge is made. |
| `404` or "model not found" | That id has no live producer. Check `/v1/models`. |
| `502` | The producer failed or was busy. No charge is made; retry. |
| Answer is JSON like `{"name": "write", ...}` | The model wrote a tool call as text. Turn the editor's tools off for HiveMind. |
| Slow first answer | The producer loads the model into the GPU on the first request. |

## Share your GPU instead

If you have a graphics card, you can earn points by serving a model. See the producer quickstart in
[HIVEMIND_OPENROUTER.md](HIVEMIND_OPENROUTER.md) and [HIVEMIND_PYTHON_GAME_DEVS.md](HIVEMIND_PYTHON_GAME_DEVS.md).
