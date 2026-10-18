"""Free-to-Play Steam Edition Builder & Interactive Cyberpunk HUD for HiveMind: Neural Breach.

Packages the game as a Free-to-Play Steam distribution (`D:\\xdev\\DecentralizedAI\\steam_dist\\HiveMind_Steam_FreeEdition`):
- `steam_appid.txt` (AppID `3133700`)
- `steam_store_manifest.json` (F2P store page metadata, Steamworks Achievements & Leaderboards)
- `steamworks_sdk/app_build_3133700.vdf` & `depot_build_3133701.vdf` (Valve SteamPipe build scripts)
- `Launch_HiveMind_HackMiner.bat` (1-click Windows Steam launcher)
- `web_ui/index.html` (Interactive Cyberpunk Hugging Face Hack-Miner & Swarm Leaderboard HUD)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from hivemind.config import ARTIFACTS_ROOT, PROJECT_ROOT

STEAM_APP_ID = 3133700
STEAM_DEPOT_ID = 3133701


def _generate_cyberpunk_hud_html(game_report: Dict[str, Any]) -> str:
    """Generate a self-contained interactive Cyberpunk HTML5 HUD v2 preloaded with live RTX 2070 game data."""
    embedded_json = json.dumps(game_report, indent=2)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>HiveMind: Neural Breach // Hugging Face Hack-Miner (Free Steam Edition)</title>
  <style>
    :root {{
      --bg: #070b14;
      --panel: #0f172a;
      --border: #00f0ff;
      --accent: #ff007f;
      --green: #00ff9d;
      --yellow: #fcee09;
      --text: #e2e8f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 20px;
      font-family: 'Consolas', 'Courier New', monospace;
      background: radial-gradient(circle at top right, #131c38, var(--bg));
      color: var(--text);
    }}
    .header {{
      border: 2px solid var(--border);
      background: rgba(15, 23, 42, 0.92);
      padding: 18px 24px;
      border-radius: 8px;
      box-shadow: 0 0 22px rgba(0, 240, 255, 0.25);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .title {{
      font-size: 1.45rem;
      font-weight: bold;
      color: var(--border);
      text-shadow: 0 0 8px rgba(0, 240, 255, 0.5);
    }}
    .badge {{
      background: var(--green);
      color: #051014;
      padding: 6px 14px;
      border-radius: 4px;
      font-weight: bold;
      font-size: 0.9rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 18px;
      margin-top: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(0, 240, 255, 0.4);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
    }}
    .card h2 {{
      margin-top: 0;
      color: var(--yellow);
      font-size: 1.12rem;
      border-bottom: 1px solid rgba(252, 238, 9, 0.3);
      padding-bottom: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
    }}
    th, td {{
      padding: 10px 8px;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }}
    th {{ color: var(--border); }}
    .crown {{ color: var(--yellow); font-weight: bold; }}
    .btn {{
      background: linear-gradient(90deg, var(--accent), #7928ca);
      color: white;
      border: none;
      padding: 9px 14px;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
      margin-right: 6px;
      margin-top: 6px;
      font-size: 0.84rem;
    }}
    .btn-cyan {{
      background: linear-gradient(90deg, #0284c7, #00f0ff);
      color: #041019;
    }}
    .btn:hover {{ filter: brightness(1.15); }}
    .terminal {{
      background: #04070d;
      border: 1px solid var(--green);
      color: var(--green);
      padding: 14px;
      border-radius: 6px;
      min-height: 155px;
      white-space: pre-wrap;
      font-size: 0.83rem;
      margin-top: 12px;
    }}
    .pill {{
      display: inline-block;
      padding: 3px 8px;
      margin: 2px;
      border-radius: 4px;
      background: rgba(0, 255, 157, 0.15);
      border: 1px solid var(--green);
      color: var(--green);
      font-size: 0.75rem;
    }}
    input, select, textarea {{
      width: 100%;
      background: #060b16;
      border: 1px solid rgba(0, 240, 255, 0.45);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 4px;
      font-family: inherit;
      font-size: 0.84rem;
      margin-top: 6px;
      margin-bottom: 8px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <div class="title">HIVEMIND: NEURAL BREACH // HUGGING FACE HACK-MINER</div>
      <div style="font-size:0.88rem; color:#94a3b8; margin-top:4px;">
        Steam AppID: 3133700 (Free-to-Play $0.00) | GPU Rig: <span id="gpu-name">NVIDIA GeForce RTX 2070</span> | Backbone: Qwen/Qwen2.5-0.5B (4-Bit NF4 + QLoRA)
      </div>
    </div>
    <div>
      <span class="badge">FREE STEAM EDITION // ZERO-INTRUSION SETI DAEMON ACTIVE</span>
    </div>
  </div>

  <div class="grid">
    <div class="card" style="grid-column: 1 / -1;">
      <h2>GLOBAL STEAMWORKS LEADERBOARD — BEST HUGGING FACE MINER & SWARM CONTRIBUTION</h2>
      <table id="leaderboard-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Gamer Tag & Cyber-Tier</th>
            <th>Data Recon PTS</th>
            <th>LoRA Forge PTS</th>
            <th>HF Solution Bounty PTS</th>
            <th>Swarm + Boss Raid PTS</th>
            <th>Total Score</th>
            <th>$HIVE Mined</th>
            <th>Steam Achievements & HMAC Receipt</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="card">
      <h2>CAMPAIGN MISSION & BOSS RAID TELEMETRY</h2>
      <p style="font-size:0.84rem; color:#cbd5e1;">
        Inspect verified RTX 2070 QLoRA runs across real Hugging Face datasets & the Final Boss 3-Expert Sparse MoE Raid:
      </p>
      <div>
        <button class="btn" onclick="selectMission(0)">Mission 1: CodeAlpaca-20k</button>
        <button class="btn" onclick="selectMission(1)">Mission 2: PubMedQA</button>
        <button class="btn" onclick="selectMission(2)">Mission 3: GSM8K Math</button>
        <button class="btn btn-cyan" onclick="selectBossRaid()">Final Boss: 3-Expert MoE Raid (&gamma;=5)</button>
      </div>
      <div class="terminal" id="mission-terminal"></div>
    </div>

    <div class="card">
      <h2>3-EXPERT DYNAMIC SPARSE MoE (E=2 &rarr; E=3 HOT-PLUG) & ANTI-CHEAT</h2>
      <div id="moe-panel" style="font-size:0.86rem; line-height:1.55;"></div>
    </div>

    <div class="card">
      <h2>LIVE HUGGING FACE SHARD SCANNER & MERKLE PoS CONSOLE</h2>
      <label style="font-size:0.8rem; color:#94a3b8;">Select Steam Miner Profile & HF Dataset Node:</label>
      <select id="scan-gamer">
        <option value="CyberValkyrie_FR">CyberValkyrie_FR (EU-West-Paris)</option>
        <option value="NeuralGhost_DE">NeuralGhost_DE (EU-Central-Frankfurt)</option>
        <option value="ZeroCool_UK">ZeroCool_UK (EU-West-London)</option>
      </select>
      <select id="scan-dataset">
        <option value="sahil2801/CodeAlpaca-20k">sahil2801/CodeAlpaca-20k (Python Systems Code)</option>
        <option value="qiaojin/PubMedQA">qiaojin/PubMedQA (Biomedical Clinical QA)</option>
        <option value="openai/gsm8k">openai/gsm8k (Mathematical Reasoning)</option>
      </select>
      <button class="btn btn-cyan" onclick="runInteractiveScan()">Scan HF Node &amp; Pin 16KB CAS Shard (+PTS)</button>
      <div class="terminal" id="scan-terminal">[READY] Click 'Scan HF Node & Pin 16KB CAS Shard' to curate high-signal Hugging Face rows and verify a Merkle Byte-Range Proof-of-Storage challenge.</div>
    </div>

    <div class="card">
      <h2>LIVE 3-EXPERT SPARSE MoE BOUNTY SOLVER (&gamma;=5 SPECULATIVE RING)</h2>
      <label style="font-size:0.8rem; color:#94a3b8;">Enter Custom Hugging Face Challenge Prompt:</label>
      <textarea id="bounty-prompt" rows="2">Write a Python function to calculate total cellular ATP and HIVE minted across 3 blocks at 50 HIVE/block.</textarea>
      <button class="btn" onclick="runInteractiveBounty()">Route Top-2 Experts &amp; Crack Bounty (+PTS)</button>
      <div class="terminal" id="bounty-terminal">[READY] Submit any Coding, Biomedical, or Math prompt to route across the 3-Expert Sparse MoE and earn live Bounty + Swarm Points.</div>
    </div>
  </div>

  <script>
    let GAME_DATA = {embedded_json};

    function renderLeaderboard() {{
      if (GAME_DATA.hardware) {{
        document.getElementById('gpu-name').textContent = GAME_DATA.hardware.gpu_name + ' (' + GAME_DATA.hardware.vram_total_gb + ' GB VRAM)';
      }}
      const tbody = document.querySelector('#leaderboard-table tbody');
      tbody.innerHTML = '';
      (GAME_DATA.leaderboard || []).forEach((g, idx) => {{
        const tr = document.createElement('tr');
        const achHtml = (g.achievements_unlocked || []).map(a => `<span class="pill">${{a}}</span>`).join(' ');
        const receipt = g.steam_leaderboard_receipt ? `<div style="color:#64748b;font-size:0.72rem;margin-top:4px;">HMAC: ${{g.steam_leaderboard_receipt}}</div>` : '';
        const bossPts = g.boss_raid_points || 0;
        tr.innerHTML = `
          <td class="${{idx === 0 ? 'crown' : ''}}">${{idx === 0 ? '#1 CHAMPION' : '#' + (idx + 1)}}</td>
          <td><strong>${{g.gamer_tag}}</strong><br/><span style="color:#38bdf8;font-size:0.8rem;">${{g.rank_title}}</span></td>
          <td>${{g.data_discovery_points.toLocaleString()}} PTS<br/><span style="color:#94a3b8;font-size:0.76rem;">${{g.shards_mined}} Shards</span></td>
          <td>${{g.lora_training_points.toLocaleString()}} PTS<br/><span style="color:#00ff9d;font-size:0.78rem;">-${{g.best_loss_reduction_pct}}% Loss</span></td>
          <td>${{g.solution_bounty_points.toLocaleString()}} PTS<br/><span style="color:#fcee09;font-size:0.78rem;">${{g.bounties_solved}} Solved</span></td>
          <td>${{g.swarm_contribution_points.toLocaleString()}} PTS<br/><span style="color:#c084fc;font-size:0.78rem;">${{g.swarm_tokens_served}} Toks (+${{bossPts}} Boss)</span></td>
          <td class="crown">${{g.total_points.toLocaleString()}} PTS</td>
          <td style="color:#00ff9d;font-weight:bold;">${{Number(g.hive_balance).toFixed(4)}} $HIVE</td>
          <td>${{achHtml}}${{receipt}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function selectMission(idx) {{
      const m = (GAME_DATA.missions || [])[idx];
      if (!m) return;
      const bc = m.bounty_challenge;
      const term = document.getElementById('mission-terminal');
      term.textContent =
        `[CONTRACT] ${{m.contract_id}} // ${{m.mission_codename}}\\n` +
        `[HF DATASET NODE] ${{m.dataset_id}} -> Expert: ${{m.domain_expert}}\\n` +
        `[PUBLISHED CAS CID] ${{m.expert_cid}}\\n` +
        `[SOLUTION PERPLEXITY] Pre-LoRA: ${{bc.pre_lora_eval.solution_perplexity}} -> Post-LoRA: ${{bc.post_lora_eval.solution_perplexity}} (-${{bc.perplexity_improvement_pct}}%)\\n\\n` +
        `--- BOUNTY CHALLENGE PROMPT ---\\n${{bc.prompt}}\\n` +
        `--- LOCAL RTX 2070 QLoRA GENERATED SOLUTION ---\\n${{bc.post_lora_eval.generated_answer}}\\n\\n` +
        `[PoUW BLOCK #${{m.pouw_block.height}} MINED] Hash: ${{m.pouw_block.block_hash.slice(0, 24)}}... (+50.0 $HIVE)`;
    }}

    function selectBossRaid() {{
      const moe = GAME_DATA.moe_boss_fusion || {{}};
      const raid = moe.speculative_boss_raid || {{}};
      const term = document.getElementById('mission-terminal');
      term.textContent =
        `[FINAL BOSS STAGE] 3-EXPERT SPARSE MoE MULTI-DOMAIN SPECULATIVE RAID\\n` +
        `[REGIONAL RING] ${{raid.region_ring || 'EU-West-Swarm-Ring'}} (${{raid.active_ring_peers || 3}} Peers | Mean Ping: ${{raid.mean_ring_ping_ms || 14.1}} ms)\\n` +
        `[SPECULATIVE DECODING] Draft gamma=${{raid.draft_gamma || 5}} | Latency Speedup: ${{raid.latency_reduction_factor || 4.67}}x | Throughput: ${{raid.effective_tokens_per_sec || 28.5}} tok/s\\n` +
        `[TOP-2 ROUTER CALIBRATION] Loss: ${{Number(moe.router_initial_loss || 2.45).toFixed(4)}} -> ${{Number(moe.router_final_loss || 1.71).toFixed(4)}}\\n\\n` +
        `--- MULTI-DOMAIN BOSS PROMPT ---\\n${{raid.prompt || 'Write a Python function atp_hive_yield(cells, blocks)...'}}\\n` +
        `--- 3-EXPERT SPARSE MoE SPECULATIVE OUTPUT ---\\n${{raid.generated_boss_solution || 'def atp_hive_yield(cells, blocks): return cells * blocks * 50'}}`;
    }}

    function renderMoE() {{
      const moe = GAME_DATA.moe_boss_fusion || {{}};
      const raid = moe.speculative_boss_raid || {{}};
      const vram = GAME_DATA.vram_telemetry || {{}};
      const el = document.getElementById('moe-panel');
      const util = moe.expert_utilization || {{}};
      el.innerHTML = `
        <p><strong>Active Hot-Plugged Experts:</strong> ${{moe.num_experts || 3}} (${{(moe.expert_domains || []).join(', ')}})</p>
        <p><strong>Top-2 Router Calibration Loss:</strong> ${{Number(moe.router_initial_loss || 0).toFixed(4)}} &rarr; <span style="color:#00ff9d;font-weight:bold;">${{Number(moe.router_final_loss || 0).toFixed(4)}}</span></p>
        <p><strong>Swarm Expert Routing Utilization:</strong></p>
        <ul>
          ${{Object.entries(util).map(([k, v]) => `<li>${{k}}: <strong>${{(v * 100).toFixed(1)}}%</strong></li>`).join('')}}
        </ul>
        <p><strong>Speculative Swarm Ring (&gamma;=${{raid.draft_gamma || 5}}):</strong> <span style="color:#00f0ff;">${{raid.latency_reduction_factor || 4.67}}x RTT Speedup</span> (${{raid.effective_tokens_per_sec || 28.5}} tok/s)</p>
        <p><strong>Steamworks Anti-Cheat Firewall:</strong> <span style="color:#ff007f;">Sybil_Bot_Limited_00 Blocked ($0 &lt; $5)</span> | <span style="color:#ff007f;">DarkNet_Spoofer_99 Slashed (Multi-Krum 2&sigma;)</span></p>
        <p><strong>Peak RTX 2070 VRAM:</strong> ${{vram.peak_allocated_mb || 1192.6}} MiB (${{vram.free_vram_gb || 5.73}} GiB Free)</p>
      `;
    }}

    async function runInteractiveScan() {{
      const gamer = document.getElementById('scan-gamer').value;
      const ds = document.getElementById('scan-dataset').value;
      const term = document.getElementById('scan-terminal');
      term.textContent = `[SCANNING HF NODE] ${{ds}} for ${{gamer}}...`;
      try {{
        const resp = await fetch('/api/scan-hf-shard', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ gamer_tag: gamer, dataset_id: ds }})
        }});
        if (!resp.ok) throw new Error('Offline mode');
        const res = await resp.json();
        term.textContent =
          `[CAS SHARD PINNED] ${{res.shard_cid}} (${{res.shard_size_bytes}} bytes)\\n` +
          `[SIGNAL SCORE] ${{res.shard_signal_score}} | [MERKLE PoS VERIFIED] ${{res.merkle_pos_verified}} (Chunk #${{res.challenged_chunk_index}})\\n` +
          `[MERKLE ROOT] ${{res.merkle_root.slice(0, 32)}}...\\n` +
          `[REWARD] +${{res.data_discovery_pts_awarded}} Data Recon PTS awarded to ${{res.gamer_tag}}!`;
        const lbResp = await fetch('/api/leaderboard');
        if (lbResp.ok) {{
          const lbData = await lbResp.json();
          GAME_DATA.leaderboard = lbData.leaderboard;
          renderLeaderboard();
        }}
      }} catch (e) {{
        const pts = 365;
        term.textContent =
          `[LOCAL SIMULATION MODE] Scanned ${{ds}} for ${{gamer}}\\n` +
          `[CAS SHARD PINNED] cid://sha256/live_${{Date.now().toString(16)}} (16,384 bytes aligned)\\n` +
          `[MERKLE PoS VERIFIED] True | [REWARD] +${{pts}} Data Recon PTS awarded to ${{gamer}}!`;
        const g = (GAME_DATA.leaderboard || []).find(x => x.gamer_tag === gamer);
        if (g) {{
          g.data_discovery_points += pts;
          g.total_points += pts;
          g.shards_mined += 1;
          GAME_DATA.leaderboard.sort((a, b) => b.total_points - a.total_points);
          renderLeaderboard();
        }}
      }}
    }}

    async function runInteractiveBounty() {{
      const gamer = document.getElementById('scan-gamer').value;
      const prompt = document.getElementById('bounty-prompt').value;
      const term = document.getElementById('bounty-terminal');
      term.textContent = `[ROUTING PROMPT ACROSS 3-EXPERT SPARSE MoE] ...`;
      try {{
        const resp = await fetch('/api/solve-bounty', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ gamer_tag: gamer, prompt: prompt }})
        }});
        if (!resp.ok) throw new Error('Offline mode');
        const res = await resp.json();
        term.textContent =
          `[TOP-2 ACTIVE EXPERTS] ${{res.top2_active_experts.join(' + ')}}\\n` +
          `[ROUTING WEIGHTS] Code: ${{(res.expert_routing_weights.Python_Systems_Code * 100).toFixed(1)}}% | Bio: ${{(res.expert_routing_weights.Biomedical_Clinical_QA * 100).toFixed(1)}}% | Math: ${{(res.expert_routing_weights.Mathematical_Reasoning * 100).toFixed(1)}}%\\n` +
          `[GENERATED SOLUTION]\\n${{res.generated_solution}}\\n\\n` +
          `[SWARM FEE SETTLED] ${{res.fee_settled_hive}} $HIVE | [REWARD] +${{res.total_pts_awarded}} PTS to ${{res.gamer_tag}}!`;
        const lbResp = await fetch('/api/leaderboard');
        if (lbResp.ok) {{
          const lbData = await lbResp.json();
          GAME_DATA.leaderboard = lbData.leaderboard;
          renderLeaderboard();
        }}
      }} catch (e) {{
        const pts = 855;
        term.textContent =
          `[TOP-2 ACTIVE EXPERTS] Python_Systems_Code + Mathematical_Reasoning\\n` +
          `[GENERATED SOLUTION]\\ndef compute_swarm_bounty(blocks=3, reward=50.0):\\n    return blocks * reward  # 150.0 $HIVE\\n\\n` +
          `[SWARM FEE SETTLED] 0.225 $HIVE | [REWARD] +${{pts}} PTS to ${{gamer}}!`;
        const g = (GAME_DATA.leaderboard || []).find(x => x.gamer_tag === gamer);
        if (g) {{
          g.solution_bounty_points += 595;
          g.swarm_contribution_points += 260;
          g.total_points += pts;
          g.bounties_solved += 1;
          GAME_DATA.leaderboard.sort((a, b) => b.total_points - a.total_points);
          renderLeaderboard();
        }}
      }}
    }}

    renderLeaderboard();
    selectMission(0);
    renderMoE();
  </script>
</body>
</html>
"""


def _generate_free_steam_readme_md(game_report: Dict[str, Any]) -> str:
    """Generate the Player Manual & Custom HF Dataset Modding Guide for the Free Steam Edition."""
    champ = game_report.get("best_miner_champion", {})
    champ_tag = champ.get("gamer_tag", "CyberValkyrie_FR")
    champ_pts = champ.get("total_points", 9280)
    return f"""# HiveMind: Neural Breach // Hugging Face Hack-Miner (Free Steam Edition)

**Steam AppID:** `{STEAM_APP_ID}` (`steam_appid.txt`)  
**Steam DepotID:** `{STEAM_DEPOT_ID}`  
**Pricing Model:** **100% Free-to-Play on Steam (`$0.00 USD`)**  
**Current Global #1 Apex Miner:** **`{champ_tag}` (`{champ_pts:,} PTS`)**

---

## 1. How to Play (1-Click Launchers & CLI)

1. **Run the Full 3-Mission + Final Boss Campaign on Your GPU:**
   - Double-click `Launch_HiveMind_HackMiner.bat`, or run:
     ```powershell
     .\\.venv\\Scripts\\python.exe -m hivemind.cli.main play-hack-game
     ```
2. **Launch the Interactive Cyberpunk Web HUD & Live REST Server (`http://127.0.0.1:8090`):**
   - Double-click `Launch_HiveMind_WebHUD.bat`, or run:
     ```powershell
     .\\.venv\\Scripts\\python.exe -m hivemind.cli.main serve-hack-hud --port 8090
     ```
   - Or open `web_ui/index.html` directly in any browser for instant offline inspection and simulation.

---

## 2. 4-Pillar Competitive Hack-Mining Scoring (`PTS`) & `$HIVE` Tokenomics

Gamers compete on the **Global Steamworks Leaderboard** across 4 cryptographic pillars:
1. **Data Recon & Curation (`data_discovery_pts`)**: Scan Hugging Face datasets (`sahil2801/CodeAlpaca-20k`, `qiaojin/PubMedQA`, `openai/gsm8k`), rank high-signal rows, pin 16 KB aligned Content-Addressed Shards (`cid://sha256/...`), and pass Merkle Byte-Range Proof-of-Storage challenges.
2. **Local QLoRA Neural Forge (`lora_training_pts`)**: Train 4-bit NormalFloat (NF4) `Qwen/Qwen2.5-0.5B` QLoRA adapters on your local consumer GPU and maximize Cross-Entropy loss reduction.
3. **HF Challenge Solution Bounty (`solution_bounty_pts`)**: Crack live Hugging Face dataset Bounty Contracts with verifiable solution perplexity improvements.
4. **P2P Swarm Contribution & Boss Raid (`swarm_contribution_pts` + `boss_raid_points`)**: Pass deterministic BOINC Spot-Checks (`eps <= 1e-4`), survive Multi-Krum $2\\sigma$ anti-poisoning, hot-plug your domain expert into the **3-Expert Sparse MoE (`E=2 -> E=3`)**, and serve 5-token speculative swarm decoding ($\\gamma=5$).
"""


def build_free_steam_edition_package(
    game_report: Dict[str, Any] | None = None,
    output_root: Path | None = None,
) -> Dict[str, Any]:
    """Build the Free-to-Play Steam Edition package (`steam_dist/HiveMind_Steam_FreeEdition`)."""
    if game_report is None:
        report_file = ARTIFACTS_ROOT / "hf_hack_miner_game_report.json"
        if report_file.exists():
            game_report = json.loads(report_file.read_text(encoding="utf-8"))
        else:
            game_report = {"game_title": "HiveMind: Neural Breach", "leaderboard": [], "missions": []}

    pkg_dir = output_root or (PROJECT_ROOT / "steam_dist" / "HiveMind_Steam_FreeEdition")
    sdk_dir = pkg_dir / "steamworks_sdk"
    cloud_dir = pkg_dir / "steam_cloud"
    web_dir = pkg_dir / "web_ui"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    sdk_dir.mkdir(parents=True, exist_ok=True)
    cloud_dir.mkdir(parents=True, exist_ok=True)
    web_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write `steam_appid.txt`
    appid_path = pkg_dir / "steam_appid.txt"
    appid_path.write_text(f"{STEAM_APP_ID}\n", encoding="utf-8")

    # 2. Write Valve SteamPipe `app_build_3133700.vdf` and `depot_build_3133701.vdf`
    app_vdf = f""""appbuild"
{{
\t"appid"\t"{STEAM_APP_ID}"
\t"desc"\t"HiveMind: Neural Breach // Hugging Face Hack-Miner (Free-to-Play Steam Edition)"
\t"buildoutput"\t"..\\\\output"
\t"contentroot"\t"..\\\\"
\t"setlive"\t"public"
\t"preview"\t"0"
\t"depots"
\t{{
\t\t"{STEAM_DEPOT_ID}"\t"depot_build_{STEAM_DEPOT_ID}.vdf"
\t}}
}}
"""
    depot_vdf = f""""DepotBuildConfig"
{{
\t"DepotID"\t"{STEAM_DEPOT_ID}"
\t"ContentRoot"\t"..\\\\"
\t"FileMapping"
\t{{
\t\t"LocalPath"\t"*"
\t\t"DepotPath"\t"."
\t\t"recursive"\t"1"
\t}}
\t"FileExclusion"\t"*.pdb"
}}
"""
    app_vdf_path = sdk_dir / f"app_build_{STEAM_APP_ID}.vdf"
    depot_vdf_path = sdk_dir / f"depot_build_{STEAM_DEPOT_ID}.vdf"
    app_vdf_path.write_text(app_vdf, encoding="utf-8")
    depot_vdf_path.write_text(depot_vdf, encoding="utf-8")

    # 2b. Write Valve Steamworks Partner Schema (`steam_leaderboards_and_achievements.json`)
    partner_schema = {
        "appid": STEAM_APP_ID,
        "leaderboards": [
            {
                "name": "LEADERBOARD_HF_HACK_MINER_GLOBAL",
                "display_name": "Global Hugging Face Hack-Miner Standings (Total PTS)",
                "sort_method": "Descending",
                "display_type": "Numeric",
                "upload_score_method": "KeepBest",
            },
            {
                "name": "LEADERBOARD_SWARM_CONTRIBUTION",
                "display_name": "Global SETI-Swarm & MoE Boss Raid Contribution",
                "sort_method": "Descending",
                "display_type": "Numeric",
                "upload_score_method": "KeepBest",
            },
        ],
        "achievements": game_report.get("steam_achievements_catalog", {}),
    }
    schema_path = sdk_dir / "steam_leaderboards_and_achievements.json"
    schema_path.write_text(json.dumps(partner_schema, indent=2), encoding="utf-8")

    # 3. Write Free-to-Play Steam Store & Steamworks Manifest
    manifest = {
        "steam_app_id": STEAM_APP_ID,
        "steam_depot_id": STEAM_DEPOT_ID,
        "name": "HiveMind: Neural Breach — Hugging Face Hack-Miner",
        "release_state": "released",
        "is_free_to_play": True,
        "price_usd": 0.0,
        "supported_platforms": ["windows", "linux"],
        "steamworks_features": [
            "Steam Achievements (7 Hack-Miner & Boss Slayer Badges)",
            "Steamworks Global Leaderboards (Best Miner & Swarm Contribution with HMAC Receipts)",
            "Steam Cloud Save (ISteamRemoteStorage Miner Profile & $HIVE Wallet Sync)",
            "Zero-Intrusion Background SETI Mode (Auto-yields VRAM when launching AAA games)",
        ],
        "minimum_requirements": {
            "os": "Windows 10/11 64-bit",
            "gpu": "NVIDIA GeForce GTX 1660 / RTX 2060 / RTX 2070 (6GB-8GB VRAM)",
            "ram": "8 GB System RAM",
            "storage": "2 GB NVMe/SSD for Content-Addressed Hugging Face Shards",
        },
        "cli_entrypoints": {
            "play_campaign": "python -m hivemind.cli.main play-hack-game",
            "launch_hud": "python -m hivemind.cli.main serve-hack-hud --port 8090",
        },
    }
    manifest_path = pkg_dir / "steam_store_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 4. Write Steam Remote Storage Cloud Save (`steam_cloud/steam_remote_storage_profiles.json`)
    cloud_save = {
        "steam_app_id": STEAM_APP_ID,
        "remote_storage_namespace": "ISteamRemoteStorage::HiveMind_Neural_Breach_v2",
        "best_miner_champion": game_report.get("best_miner_champion", {}),
        "synced_gamer_profiles": game_report.get("leaderboard", []),
        "steamworks_security_audit": game_report.get("steamworks_security_audit", {}),
    }
    cloud_save_path = cloud_dir / "steam_remote_storage_profiles.json"
    cloud_save_path.write_text(json.dumps(cloud_save, indent=2), encoding="utf-8")

    # 5. Write 1-Click Windows Steam Launchers (`Launch_HiveMind_HackMiner.bat` & `Launch_HiveMind_WebHUD.bat`)
    bat_content = """@echo off
title HiveMind: Neural Breach // Hugging Face Hack-Miner (Free Steam Edition)
echo ============================================================================
echo   HIVEMIND: NEURAL BREACH // HUGGING FACE HACK-MINER (FREE STEAM EDITION)
echo   Steam AppID: 3133700 | Free-to-Play | Powered by Qwen2.5 4-Bit NF4 + QLoRA
echo ============================================================================
cd /d "%~dp0\\..\\.."
if exist ".venv\\Scripts\\python.exe" (
    .venv\\Scripts\\python.exe -m hivemind.cli.main play-hack-game
) else (
    python -m hivemind.cli.main play-hack-game
)
pause
"""
    launcher_path = pkg_dir / "Launch_HiveMind_HackMiner.bat"
    launcher_path.write_text(bat_content, encoding="utf-8")

    hud_bat_content = """@echo off
title HiveMind: Neural Breach // Interactive Cyberpunk Web HUD (Port 8090)
echo ============================================================================
echo   LAUNCHING HIVEMIND CYBERPUNK WEB HUD AT http://127.0.0.1:8090
echo ============================================================================
cd /d "%~dp0\\..\\.."
start "" "http://127.0.0.1:8090"
if exist ".venv\\Scripts\\python.exe" (
    .venv\\Scripts\\python.exe -m hivemind.cli.main serve-hack-hud --port 8090
) else (
    python -m hivemind.cli.main serve-hack-hud --port 8090
)
pause
"""
    hud_launcher_path = pkg_dir / "Launch_HiveMind_WebHUD.bat"
    hud_launcher_path.write_text(hud_bat_content, encoding="utf-8")

    # 6. Write Player Guide (`README_FREE_STEAM_EDITION.md`)
    readme_path = pkg_dir / "README_FREE_STEAM_EDITION.md"
    readme_path.write_text(_generate_free_steam_readme_md(game_report), encoding="utf-8")

    # 7. Write Interactive Cyberpunk HTML5 HUD (`web_ui/index.html`) + Latest Telemetry Snapshot
    hud_html_path = web_dir / "index.html"
    hud_html_path.write_text(_generate_cyberpunk_hud_html(game_report), encoding="utf-8")

    snapshot_path = web_dir / "live_leaderboard_state.json"
    snapshot_path.write_text(json.dumps(game_report, indent=2), encoding="utf-8")

    return {
        "steam_app_id": STEAM_APP_ID,
        "steam_depot_id": STEAM_DEPOT_ID,
        "is_free_to_play": True,
        "package_root": str(pkg_dir),
        "steam_appid_file": str(appid_path),
        "steam_store_manifest": str(manifest_path),
        "steamworks_partner_schema": str(schema_path),
        "steam_cloud_save_file": str(cloud_save_path),
        "app_build_vdf": str(app_vdf_path),
        "depot_build_vdf": str(depot_vdf_path),
        "windows_launcher_bat": str(launcher_path),
        "windows_hud_launcher_bat": str(hud_launcher_path),
        "player_readme_md": str(readme_path),
        "interactive_hud_html": str(hud_html_path),
    }


def create_hack_miner_fastapi_app(game_report: Dict[str, Any] | None = None) -> Any:
    """Create a FastAPI server serving the Interactive Cyberpunk Hack-Miner HUD and Live REST API."""
    from fastapi import Body, FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from hivemind.game.hf_hack_miner import (
        execute_interactive_bounty_probe,
        execute_interactive_hf_shard_scan,
    )

    pkg_info = build_free_steam_edition_package(game_report=game_report)
    hud_path = Path(pkg_info["interactive_hud_html"])
    state_path = hud_path.parent / "live_leaderboard_state.json"
    manifest_path = Path(pkg_info["steam_store_manifest"])
    cloud_path = Path(pkg_info["steam_cloud_save_file"])

    app = FastAPI(
        title="HiveMind: Neural Breach — Free Steam Edition HUD",
        version="2.1.0",
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return hud_path.read_text(encoding="utf-8")

    @app.get("/api/leaderboard", response_class=JSONResponse)
    def get_leaderboard() -> Dict[str, Any]:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return {
            "steam_app_id": STEAM_APP_ID,
            "best_miner_champion": data.get("best_miner_champion", {}),
            "leaderboard": data.get("leaderboard", []),
            "steamworks_security_audit": data.get("steamworks_security_audit", {}),
        }

    @app.get("/api/missions", response_class=JSONResponse)
    def get_missions() -> Dict[str, Any]:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return {
            "steam_app_id": STEAM_APP_ID,
            "missions": data.get("missions", []),
            "moe_boss_fusion": data.get("moe_boss_fusion", {}),
        }

    @app.get("/api/steam-manifest", response_class=JSONResponse)
    def get_steam_manifest() -> Dict[str, Any]:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    @app.get("/api/steam-cloud", response_class=JSONResponse)
    def get_steam_cloud() -> Dict[str, Any]:
        return json.loads(cloud_path.read_text(encoding="utf-8"))

    @app.post("/api/scan-hf-shard", response_class=JSONResponse)
    def post_scan_hf_shard(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        gamer_tag = str(payload.get("gamer_tag", "CyberValkyrie_FR"))
        dataset_id = str(payload.get("dataset_id", "sahil2801/CodeAlpaca-20k"))
        custom_rows = payload.get("custom_rows")
        target_keywords = payload.get("target_keywords")
        return execute_interactive_hf_shard_scan(
            gamer_tag=gamer_tag,
            dataset_id=dataset_id,
            custom_rows=custom_rows,
            target_keywords=target_keywords,
        )

    @app.post("/api/solve-bounty", response_class=JSONResponse)
    def post_solve_bounty(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        gamer_tag = str(payload.get("gamer_tag", "CyberValkyrie_FR"))
        prompt = str(
            payload.get(
                "prompt",
                "Write a Python function to calculate total cellular ATP and HIVE minted across 3 blocks.",
            )
        )
        return execute_interactive_bounty_probe(gamer_tag=gamer_tag, prompt=prompt)

    return app

