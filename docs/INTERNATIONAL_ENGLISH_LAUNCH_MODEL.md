# 🌍 International English Launch Readiness: Mathematical, Social & Economic Model (`HM-S19-INTL`)

- **Specification ID:** `HM-S19-INTL`
- **Reference Implementation:** [`src/hivemind/economy/international_launch_model.py`](../src/hivemind/economy/international_launch_model.py)
- **Live Diagnostic Endpoint:** `GET /v1/hivemind/international-launch-readiness`
- **Verification Suite:** [`tests/test_international_launch_model.py`](../tests/test_international_launch_model.py)
- **Date:** 2026-10-27

---

## 1. Honest Readiness Verdict: Are We Ready Today for a Large English Viral Wave?

**Short Answer:**
- **For an unthrottled mass English consumer launch TODAY (Oct 27):** **NO (`ready_for_unthrottled_mass_english_launch_today = False`).**
- **For a 4-Stage "Producer-First" International Rollout starting Nov 1–3 (`Stage 1 -> Stage 2 -> Stage 3`):** **YES (`ready_for_staged_producer_first_rollout = True`).**

### Why an Immediate Unthrottled English Launch Today Would Fail (The "Consumer Tsunami")
In our French community pilot (`r/artificielle`, `47` accounts), we achieved an extraordinary **Supply Participation Ratio**:
$$\sigma_{\text{FR}} = \frac{N_{\text{producers}} + N_{\text{prosumers}}}{N_{\text{total}}} = \frac{8 + 14}{47} = 46.8\%$$
In contrast, when an open-source AI project goes viral on large English-speaking aggregators (`r/LocalLLaMA`, `r/MachineLearning`, Hacker News `Show HN`, X/Twitter), empirical conversion funnels exhibit a **severe consumer skew**:
- **`90%–96%` Pure Consumers ("Free API Tourists"):** Sign up in 10 seconds in the browser, burn their free starter grant on heavy coding prompts, and never install a producer script.
- **`2%–4%` Immediate Producers ($\alpha_p$):** Plug in a GPU on Day 1.
- **`5%–10%` Latent Prosumers:** Have an `RTX 3060/3090/4090`, `RX 7900 XTX`, or `Apple M2/M3 Max` under their desk, start as consumers, and **only convert into Producers if the economic & social incentives nudge them**.

---

## 2. Mathematical Model ($M/M/c$ Erlang-C Queueing & Stability Boundary)

Let $c = M_d$ be the number of active daytime GPU producer endpoints, $\tau_{\text{agg}} = \sum_{i=1}^c \tau_i$ be the aggregate daytime token throughput (measured today at $\tau_{\text{agg}} \approx 380\text{ tok/s}$ across $c = 8$ producers), and $\bar{L} = \frac{240,162}{1,427} = 168.3\text{ tokens/request}$ be the empirical mean request size on `hivemind-openrouter`.

1. **Aggregate Daytime Service Rate ($\mu_{\text{tot}}$):**
   $$\mu_{\text{tot}} = \frac{\tau_{\text{agg}}}{\bar{L}} = \frac{380.0}{168.3} = 2.2579\text{ requests/sec}$$
2. **Traffic Intensity ($\rho$) & Erlang-C Waiting Probability ($P_Q$):**
   For consumer request arrival rate $\lambda_c$, utilization is $\rho = \frac{\lambda_c}{\mu_{\text{tot}}}$ and offered load is $a = c \rho$. Under an $M/M/c$ queue, the probability that an arriving request must wait in queue is:
   $$P_Q(c, \rho) = \frac{\frac{a^c}{c!(1 - \rho)}}{\sum_{k=0}^{c-1} \frac{a^k}{k!} + \frac{a^c}{c!(1 - \rho)}}$$
   and the 95th-percentile total response time is:
   $$T_{95}(\lambda_c) = T_{\text{net}} + T_{\text{service}} + \max\!\left(0,\; \frac{-\ln(0.05 / P_Q)}{\mu_{\text{tot}} - \lambda_c}\right)$$
3. **Hard Stability Boundary ($\rho_{\max} = 0.78$):**
   - **Today ($c = 8$, $\tau_{\text{agg}} = 380\text{ tok/s}$):** Maximum safe arrival rate is $\lambda_{\max} = 0.78 \times 2.2579 = 1.76\text{ req/s}$ (`~220` simultaneously active human developers).
   - **Under an unthrottled English spike (`8,000` visitors $\implies 1,200$ signups):** Without economic dampeners, arrival rate reaches $\lambda_c \approx 5.91\text{ req/s} \implies \rho = 2.62 > 1.0$, causing immediate queue divergence ($P_Q = 1.0$, $T_{95} \to \infty$, `HTTP 502/504`).

---

## 3. Economic & Game-Theoretic Model (Defeating the Free-Rider Tragedy of the Commons)

All `HMT` (`HiveMind Reflexion Points`) remain strictly **non-monetary compute-reciprocity credits (`0.00 EUR / 0.00 USD`)**. To prevent pure-consumer signups from draining the swarm, we introduce **Three Economic Mechanisms** in [`international_launch_model.py`](../src/hivemind/economy/international_launch_model.py):

### Mechanism 1 — Supply-Pegged Progressive Starter Grant (`15 HMT` Instant + `85 HMT` Prosumer Unlock)
Instead of minting `100 HMT` unconditionally on signup (`200,000 HMT` of unbacked demand for `2,000` tourists), every new account receives:
1. **`G_{\text{instant}} = 15.0\text{ HMT}` on Signup (`85%` reduction in unbacked starter inflation):**
   - Enough for **~37,500 tokens on `hivemind/coder` (7B/9B)** or **~75,000 tokens on `hivemind/coder-lite` (3B) / `hivemind/crowd-v1` (1B)** — i.e., `25–45` full coding prompts to experience the swarm.
2. **`G_{\text{unlock}} = +85.0\text{ HMT}` Instant Prosumer Bonus:**
   - Unlocked the moment the user runs the 60-second GPU benchmark (`scripts/run_community_gpu_benchmark.py`) or sends **1 verified Producer/Miner heartbeat** (`/v1/hivemind/producer/heartbeat`).
3. **Supply-Pegged Daily Drip for Pure Consumers without GPUs:**
   $$G_{\text{drip}}(t) = G_{\text{base}} \cdot \min\!\left(1.0,\; \frac{S_{\text{epoch}}(t)}{D_{\text{epoch}}(t)}\right)$$
   When consumer demand $D_{\text{epoch}}$ exceeds producer supply $S_{\text{epoch}}$, free daily drips automatically contract, guaranteeing **zero unbacked inflation**.

### Mechanism 2 — Congestion-Adjusted Kelly & Surge Pricing ($\omega(\rho)$)
When utilization $\rho$ rises above $\rho_0 = 0.60$, the clearing price per 1,000 tokens automatically scales along a convex congestion curve up to $\rho_{\text{cap}} = 0.95$:
$$\omega(\rho) = p_0 \cdot \left(1 + \gamma \left(\frac{\max(0, \rho - \rho_0)}{\rho_{\text{cap}} - \rho_0}\right)^2\right), \qquad \gamma = 2.5$$
- At $\rho \le 0.60$ (calm): $\omega = p_0 = 0.40\text{ HMT/1k}$ (producer earns $0.85 \times 0.40 = 0.34\text{ HMT/1k}$).
- At $\rho = 0.90$ (English viral peak): $\omega(0.90) = 2.837 \times p_0 = 1.135\text{ HMT/1k}$ (**producer reward jumps `$2.84\times$` to `0.965 HMT/1k`!**).
- **Game-Theoretic Effect:** High consumer traffic makes daytime GPU mining **$2.8\times$–$3.5\times$ more rewarding**, pulling idle prosumers online right when the swarm needs capacity most!

### Mechanism 3 — Two-Lane Priority QoS (`Prosumer Fast-Lane` vs. `Tourist Fair-Queue`)
- **Lane 1 (`Prosumers & Producers` — contributed $\ge 1$ GPU workunit/heartbeat in the last 7 days):**
  - `70%` reserved daytime capacity, `< 200 ms` TTFT priority routing, and full access to Big-Producer models (`hivemind/moe-coder` `Qwen3.5-30B-A3B` & `hivemind/coder-plus` `14B`).
- **Lane 2 (`Pure Consumers` on `15 HMT` starter grant):**
  - Routed to fast `std_8gb` & speculative tiers (`coder-lite` 3B, `crowd-v1` 1B, `coder` 7B/9B), `12 req/min` rate-limit, and an interactive conversion prompt whenever a queue forms.

---

## 4. Social Diffusion Model & The Critical Prosumer Conversion Rate ($\kappa^*$)

Let a viral English wave bring $N_{\text{new}}$ signups, where $\alpha_p$ (`~3%`) are immediate GPU producers and $(1 - \alpha_p)$ (`~97%`) start as consumers. A fraction $\kappa \in [0, 1]$ of those initial consumers convert into **Night/Idle Prosumers** via our 1-click installer + `+85 HMT` unlock + Fast-Lane QoS.

For the wave to be **net-positive (self-sustaining)** in daytime token throughput, the newly added Prosumer + Producer supply must cover the newly added Consumer demand:
$$\kappa \ge \kappa^* = \frac{\bar{\lambda}_c \cdot \bar{L}}{\tau_{\text{prosumer}} \cdot \eta_{\text{duty}}} - \frac{\alpha_p}{1 - \alpha_p}$$

Using our empirical telemetry ($\bar{L} = 168.3\text{ tok/req}$, $\bar{\lambda}_c = 0.0022\text{ req/s}$ per active consumer, $\tau_{\text{prosumer}} = 48\text{ tok/s}$ on a consumer GPU with $\eta_{\text{duty}} = 0.35$ daytime micro-idle duty cycle):
- Without any direct producers ($\alpha_p = 0$), $\kappa^* = \frac{0.3703}{16.80} = 2.20\%$.
- With $\alpha_p = 2.5\%–4.0\%$ direct producers and a modest **$\kappa = 7.5\%–9.0\%$ Consumer $\to$ Prosumer conversion rate**, every English wave **expands** net swarm capacity rather than depleting it!

### Simulated International Launch Waves (`evaluate_international_launch_readiness()`)

| Launch Wave | Visitors / Signups | Initial Producers ($\alpha_p$) | Converted Prosumers ($\kappa$) | Pure Consumers | $\rho$ **Without** Controls | $\rho$ **With** 4 Controls | Post-Wave Daytime Capacity | Post-Wave Peak TFLOPS | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Supply-First Seed** (`r/BOINC`, `r/homelab`, `CrowdGPT`, `3,000 Devs`) | `1,200` / `240` | `29` (`12.0%`) | `38` (`18.0%`) | `173` | `0.50` | **`0.13`** | **`2,062 tok/s`** | **`1,758 TFLOPS`** | ✅ **Builds Supply Cushion** |
| **Stage 2: Technical ML/GPU Launch** (`r/LocalLLaMA`, `r/MachineLearning`, `r/SelfHosted`) | `8,000` / `1,200` | `48` (`4.0%`) | `104` (`9.0%`) | `1,048` | **`1.47` (CRASH)** | **`0.46` (SAFE)** | **`5,625 tok/s`** | **`4,632 TFLOPS`** | ✅ **Self-Sustaining (`p50 < 250ms`)** |
| **Stage 3: Broad International Viral Wave** (`Show HN`, `X/Twitter`, `Dev.to`) | `25,000` / `3,500` | `88` (`2.5%`) | `256` (`7.5%`) | `3,156` | **`3.21` (CRASH)** | **`0.61` (SAFE)** | **`13,269 tok/s`** | **`11,314 TFLOPS` (`11.3 PFLOPS`)** | ✅ **Global Scale (`12.9%` Supply Ratio)** |

---

## 5. Action Plan: WHEN and HOW to Execute the International English Launch

| Stage | Dates | Target Audience & Channels | Key Messaging Hook | Mandatory Go/No-Go Gate |
| :--- | :---: | :--- | :--- | :--- |
| **Stage 0: Hardening & Economic Shield** | **Days 1–5** (`Oct 27 – Oct 31`) | Existing `47` Pioneer Nodes + `@Vxtzq` CrowdGPT Integration | Internal engineering: ship Qwen 3.5 `14B`/`30B-A3B` (`#239`–`#243`), `15+85 HMT` progressive drip, and 2-Lane Priority QoS. | **DO NOT post mass English consumer links yet.** Pass all Season 19 Day 1–5 gates (`#239`–`#243`). |
| **Stage 1: Producer-First Seed Wave** | **Days 6–7** (`Nov 1 – Nov 2`) | **`r/BOINC`, `r/homelab`, `r/gpumining`, `CrowdGPT` (100 nodes), & 3,000 Internal Developers** | *"Turn your idle 8 GB–24 GB GPU into a verified Qwen 3.5 MoE node at night (`5,632 Cobblestones` proven) & unlock unlimited Fast-Lane coding."* | Reach **`>= 35` active GPU producers**, **`>= 1,800 TFLOPS` peak**, and **`>= 1,800 tok/s`** daytime serving capacity. |
| **Stage 2: Technical English AI/GPU Wave** | **Days 8–10** (`Nov 3 – Nov 5`) | **`r/LocalLLaMA`, `r/MachineLearning`, `r/SelfHosted`, HuggingFace** | Lead with the **v3.9.0 Paper & Engineering Breakthrough**: *"How we trained & served Qwen 3.5 MoE across 47 consumer GPUs (<6.5 GB WDDM VRAM) with SVD Rank-32 $\to$ 16 Night Distillation."* | Sustain $\rho < 0.65$, `p50 < 250 ms`, `0%` HTTP 5xx, and Consumer $\to$ Prosumer conversion $\kappa \ge 7.5\%$. |
| **Stage 3: Broad International Developer Launch** | **Days 12–14** (`Nov 7 – Nov 9`) | **Hacker News (`Show HN`), X/Twitter, Dev.to, LinkedIn** | *"Show HN: HiveMind@Home — A citizen BOINC + OpenRouter swarm where your idle night GPU pays for your daytime coding AI (`AGPL-3.0`)."* | Sustain **`>= 5,000 tok/s`** daytime capacity, **`> 10 PFLOPS`** peak nocturnal swarm, and $\rho \le 0.78$. |
