# HiveMind: Neural Breach // Hugging Face Hack-Miner (Free Steam Edition)

**Steam AppID:** `3133700` (`steam_appid.txt`)  
**Steam DepotID:** `3133701`  
**Pricing Model:** **100% Free-to-Play on Steam (`$0.00 USD`)**  
**Current Global #1 Apex Miner:** **`CyberValkyrie_FR` (`10,659 PTS`)**

---

## 1. How to Play (1-Click Launchers & CLI)

1. **Run the Full 3-Mission + Final Boss Campaign on Your GPU:**
   - Double-click `Launch_HiveMind_HackMiner.bat`, or run:
     ```powershell
     .\.venv\Scripts\python.exe -m hivemind.cli.main play-hack-game
     ```
2. **Launch the Interactive Cyberpunk Web HUD & Live REST Server (`http://127.0.0.1:8090`):**
   - Double-click `Launch_HiveMind_WebHUD.bat`, or run:
     ```powershell
     .\.venv\Scripts\python.exe -m hivemind.cli.main serve-hack-hud --port 8090
     ```
   - Or open `web_ui/index.html` directly in any browser for instant offline inspection and simulation.

---

## 2. 4-Pillar Competitive Hack-Mining Scoring (`PTS`) & `$HIVE` Tokenomics

Gamers compete on the **Global Steamworks Leaderboard** across 4 cryptographic pillars:
1. **Data Recon & Curation (`data_discovery_pts`)**: Scan Hugging Face datasets (`sahil2801/CodeAlpaca-20k`, `qiaojin/PubMedQA`, `openai/gsm8k`), rank high-signal rows, pin 16 KB aligned Content-Addressed Shards (`cid://sha256/...`), and pass Merkle Byte-Range Proof-of-Storage challenges.
2. **Local QLoRA Neural Forge (`lora_training_pts`)**: Train 4-bit NormalFloat (NF4) `Qwen/Qwen2.5-0.5B` QLoRA adapters on your local consumer GPU and maximize Cross-Entropy loss reduction.
3. **HF Challenge Solution Bounty (`solution_bounty_pts`)**: Crack live Hugging Face dataset Bounty Contracts with verifiable solution perplexity improvements.
4. **P2P Swarm Contribution & Boss Raid (`swarm_contribution_pts` + `boss_raid_points`)**: Pass deterministic BOINC Spot-Checks (`eps <= 1e-4`), survive Multi-Krum $2\sigma$ anti-poisoning, hot-plug your domain expert into the **3-Expert Sparse MoE (`E=2 -> E=3`)**, and serve 5-token speculative swarm decoding ($\gamma=5$).
