# Connect your IDE to the HiveMind OpenRouter

Use the public OpenRouter from OpenCode, VS Code (Continue, Cline), JetBrains, Zed, Aider or any tool that accepts an
OpenAI-compatible provider.

| | |
|---|---|
| Base URL | `https://hivemind-openrouter-1003063861791.us-central1.run.app/v1` |
| Key | `python scripts/hivemind_connect.py register --handle you --role consumer --save-key ~/.hivemind_key` |
| Model | `hivemind/coder` (7B) or `hivemind/coder-lite` (3B); check `/v1/models` for what is live |

- Print a ready-made snippet: `python scripts/hivemind_connect.py snippet opencode --model coder`
- Tested live: OpenCode, the OpenAI Python client, cURL. Other editors: format-checked only.
- Keep an output limit in the editor settings, or a small balance is refused.
- 7B models often write tool calls as plain text, so use them for chat-style help.
- HMT are points with no cash value; producers see your prompts, so send no secrets.

Full guide with every editor and a troubleshooting table: [docs/HIVEMIND_CONNECT_IDE.md](https://github.com/xaviercallens/HiveMind/blob/main/docs/HIVEMIND_CONNECT_IDE.md).
