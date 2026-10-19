# HiveMind PC continuation prompt

Continue the HiveMind audit implementation from the user's fork:
https://github.com/clark719664/HiveMind/tree/fix/audit-runtime-hardening

Read CONTRIBUTING.md and any applicable AGENTS.md first. Record the fetched commit
SHA in your report. Use a new worktree/branch from that ref; preserve any existing
Claude work and uncommitted files. Never reset or overwrite the user's checkout,
force-push, merge upstream, or publish synthetic success reports. The integration
branch preserves the original PR40 and PR41 ancestry. Existing review branches:

- PR40: feat/real-quic-stun-turn, network changes, reviewed checkpoint
  3452d4e93693f0b10b71122b7f0fe25fd6801894; its CI passed.
- PR41: fix/real-lean-21-theorems, real compiler enforcement; checkpoint
  e866e6bdfc0133435dc78dc8bb9c7abb9aefef67 passed remote CI.
- fix/steam-native-safety: 3b7f497f29fe5aed49766a12089fbeb8386a3bbf,
  isolated Steam bridge fixes based on upstream main.

## What is already implemented

Protocol-correct authenticated STUN/TURN parsing and retransmission, bounded QUIC
remote endpoint API with TLS verification and resumed sessions, verified WSS
fallback without unsafe ambiguous-delivery replay; unified replay nonce state;
correct speculative rejection sampling; payload/checkpoint integrity and trusted
assignment checks; NF4 packed storage accounting; atomic community imports and
restart-safe ledger; true supplied-text CPU LoRA training API; real Lean compiler
and axiom auditing; WebGL renderer and lifecycle handling; safer Steam ABI and
idempotent local credit; installed-wheel resources and writable caches; explicit
training preemption and CPU offload with fail-closed error reporting.

Do not redo these blindly. Read the code and regression tests, then validate the
hardware-dependent paths below. CPU-only cloud tests do not establish GPU,
Steam-service, browser raster, or WAN behavior.

## Reproduce the baseline

Use a fresh Python 3.11 environment. The CPU-tested combination is Torch 2.6.0,
Transformers 4.57.6, PEFT 0.17.1, NumPy 2.1.3. Install the declared project/dev
dependencies. For CUDA select the official Torch build appropriate to the already
installed driver; do not change drivers or security configuration silently.
Use the pinned lean4/lean-toolchain, then run:

    cd lean4
    lake build
    cd ..
    python -m pytest tests -q --tb=short

Keep optional public-network tests opt-in. Record exact versions, commands,
results, skips, logs, and commit. Tests may generate paper JSON reports: inspect
and exclude generated fixture reports from source commits.

## Remaining PC validation and implementation

1. GPU preemption: follow docs/GPU_PREEMPTION_VALIDATION.md. Test actual CUDA
   allocation and registered model buffers with the original owner still alive,
   optimizer state, disk/offload failures, Windows idle readings, and safe restart.
   No complete-VRAM-release or latency guarantee without measurement.
2. GPU kernels and NF4: current packed representation is reconstructed densely
   for execution. This is not a fused NF4 matmul or proof of memory residency
   savings. Implement genuine kernels only if feasible, compare against dense
   reference across dimensions/dtypes/edge cases, measure allocated/reserved
   bytes and latency, and keep unsupported devices explicit. Current DMA ring is
   bounded software buffering, not demonstrated zero-copy device DMA.
3. Steam: use a legally available Steamworks SDK and an authorized application
   ID with Steam running. Never use the repository's legacy 3133700 constant to
   publish; it belongs to another product. Verify ctypes interface ABI and native
   return/error behavior. HMAC fallback receipts are local simulation, not Valve
   authentication, remote leaderboard updates, or payment. Finish only with
   authorized SDK/service access. Do not accept terms or create credentials for
   the user. Validate distribution on a clean Windows installation; existing ZIP
   launchers require the repository/Python and are not a standalone installer.
4. WebGL: launch the local HUD in a real browser; capture screenshots and console
   errors, check shader compilation, rendered nodes/edges, pointer/keyboard
   selection, resize, context loss/restoration, and mobile/low-end performance.
   Node lifecycle tests passed, but cloud browser raster access was blocked.
   Keep fixture/demo badges and reject fake hardware certification.
5. HF training: test train_tokenized_domain_adapter with an approved local
   tokenizer/pretrained model and actual authorized text, held-out evaluation,
   saved adapter reload and deterministic recovery. Tiny CPU LoRA tests passed.
   Wire real training into the user-facing game flow if desired; its current
   micro-forge is intentionally labeled synthetic and cannot certify itself.
   Do not upload private datasets or models without permission.
6. Network: test two separately owned machines with TLS identities and actual
   bytes/ACKs, UDP-blocked WSS fallback, packet loss, retransmission, session
   resumption, and NAT/TURN relay. Existing endpoint tests use real local sockets
   and TLS; they do not prove Internet discovery or NAT orchestration is wired
   into every application path. Do not change firewall/router/security settings
   without specific approval. Do not expose an unauthenticated service publicly.

## Review and publication

Only update user-owned branches with fast-forward commits after checking for
concurrent edits. Preserve original attribution. Split other contributors' fixes
into focused review proposals against current upstream main. No upstream merge.
The cloud GitHub integration returned HTTP403 for new upstream PR creation;
this is an access limitation, not a code failure. The user can submit the prepared
fork comparison through their normal authorized GitHub interface. Do not bypass
that integration denial by extracting credentials or alternate hidden routes.

Stop and report blockers requiring SDK rights, unavailable hardware, credentials,
security changes or owner approval. Report real failures and unfinished features,
not a percentage-complete or a claim that every original marketing statement is
now true. Commit tested checkpoints and give the user exact SHAs and commands.

## Final cloud verification

The final source suite passed 470 tests, with 2 opt-in live-network skips and
4 PEFT warnings, in 59.77 seconds. Real Lean was available; hardware was CPU-only.
Generated Steam HTML and paper run reports were excluded from publication.
Generate the HUD locally from the updated source renderer; do not treat the
previous checked-in static snapshot as the updated UI.
