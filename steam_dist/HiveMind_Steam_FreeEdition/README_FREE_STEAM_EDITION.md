# HiveMind: Neural Breach // Hugging Face Hack-Miner (Free Steam Tool & Local Prototype v1.0.0)

**Steam AppID (Local Config):** `3133700` (`steam_appid.txt`)  
**Steam DepotID (Local Config):** `3133701`  
**Distribution & Steam Status:** Local Python RTX Tool & Hack-Miner prototype; requires the repository and its `.venv` dependencies. The default test App ID `3133700` belongs to [Fantasy Grounds - Raiders of the Ghost Zone](https://store.steampowered.com/app/3133700/) on the public Steam Store, not HiveMind; see [Steam integration status](../../docs/STEAM_STATUS.md).  
**Current #1 Apex Miner (Local Report):** **`CyberValkyrie_FR` (`33,236 PTS`)**  
**Active Sparse MoE Fusion:** **`6-Expert Top-2 Sparse MoE` (`E=2 -> E=6` Hot-Plug + 2-Slot Pinned RAM DMA)**

---

## 1. How to Play Locally (1-Click Launchers & CLI)

1. **Run the Full 6-Mission (Season 1 + Season 2) + Omega Boss Campaign on Your GPU:**
   - Double-click `Launch_HiveMind_HackMiner.bat`, or run:
     ```powershell
     .\.venv\Scripts\python.exe -m hivemind.cli.main play-hack-game --season all
     ```
2. **Launch the Interactive Cyberpunk Web HUD & Live REST Server (`http://127.0.0.1:8090`):**
   - Double-click `Launch_HiveMind_WebHUD.bat`, or run:
     ```powershell
     .\.venv\Scripts\python.exe -m hivemind.cli.main serve-hack-hud --port 8090
     ```
   - Or open `web_ui/index.html` directly in any browser for instant offline inspection and simulation.

---

## 2. Season 1 & Season 2 Hugging Face Missions (`HM-M5-01` & `HM-M5-02`)

- **Season 1 Missions (`--season 1`):**
  1. `CONTRACT-HF-01-CODEALPACA` — `sahil2801/CodeAlpaca-20k` (`Python_Systems_Code`)
  2. `CONTRACT-HF-02-PUBMEDQA` — `qiaojin/PubMedQA` (`Biomedical_Clinical_QA`)
  3. `CONTRACT-HF-03-GSM8K` — `openai/gsm8k` (`Mathematical_Reasoning`)
- **Season 2 Missions (`--season 2`):**
  4. `CONTRACT-HF-04-RUST-SYSTEMS` — `anon/rust-code-instructions` (`Rust_Systems_Security`)
  5. `CONTRACT-HF-05-CYBERSEC-AUDIT` — `CyberNative/Code_Vulnerability_Security_DPO` (`CyberSecurity_Vulnerability_Recon`)
  6. `CONTRACT-HF-06-MULTILINGUAL-FR` — `CohereForAI/aya_collection` (`Multilingual_FR_Synthesis`)

---

## 3. 4-Pillar Competitive Hack-Mining Scoring (`PTS`) & `$HIVE` Tokenomics

Gamers compete on the **Global Steamworks Leaderboard** across 4 cryptographic pillars:
1. **Data Recon & Curation (`data_discovery_pts`)**: Scan Hugging Face datasets, rank high-signal rows, pin 16 KB aligned Content-Addressed Shards (`cid://sha256/...`), and pass Merkle Byte-Range Proof-of-Storage challenges.
2. **Local QLoRA Neural Forge (`lora_training_pts`)**: Train 4-bit NormalFloat (NF4) `Qwen/Qwen2.5-0.5B` QLoRA adapters on your local consumer GPU and maximize Cross-Entropy loss reduction.
3. **HF Challenge Solution Bounty (`solution_bounty_pts`)**: Crack live Hugging Face dataset Bounty Contracts with verifiable solution perplexity improvements.
4. **P2P Swarm Contribution & Omega Boss Raid (`swarm_contribution_pts` + `boss_raid_points`)**: Pass deterministic BOINC Spot-Checks (`eps <= 1e-4`), survive Multi-Krum $2\sigma$ anti-poisoning, hot-plug your domain expert into the **6-Expert Sparse MoE (`E=2 -> E=6`)**, and serve speculative swarm decoding ($\gamma=5..6$).
