"""Comprehensive fast unit and integration tests for BOINC, Client, and Game modules.

Covers >=95% of:
1. `src/hivemind/boinc/coordinator_api.py`
2. `src/hivemind/boinc/validator.py` & `src/hivemind/boinc/workunit.py`
3. `src/hivemind/client/worker_node.py`
4. `src/hivemind/client/daemon.py` & `src/hivemind/client/steamworks.py`
5. `src/hivemind/game/hf_hack_miner.py` & `src/hivemind/game/rtx_steam_tool_deployer.py`
6. `src/hivemind/game/steam_edition.py` & `src/hivemind/game/georing_webgl.py`
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import psutil
import pytest
import torch
from fastapi.testclient import TestClient

import hivemind.boinc.coordinator_api as coordinator_api_mod
import hivemind.boinc.validator as validator_mod
import hivemind.boinc.workunit as workunit_mod
import hivemind.client.daemon as daemon_mod
import hivemind.client.steamworks as steamworks_mod
import hivemind.client.worker_node as worker_node_mod
import hivemind.game.georing_webgl as georing_mod
import hivemind.game.hf_hack_miner as hf_miner_mod
import hivemind.game.rtx_steam_tool_deployer as rtx_deployer_mod
import hivemind.game.steam_edition as steam_edition_mod
from hivemind.boinc.coordinator_api import (
    BOINCCoordinatorService,
    create_coordinator_fastapi_app,
)
from hivemind.boinc.validator import DeterministicSpotCheckValidator, SpotCheckVerdict
from hivemind.boinc.workunit import (
    BOINCWorkunit,
    create_signed_parquet_shard,
    verify_and_load_parquet_shard,
)
from hivemind.client.daemon import (
    SETIAgentState,
    ZeroIntrusionDaemon,
    detect_running_game_process,
    get_windows_idle_seconds,
)
from hivemind.client.steamworks import (
    SteamAuthSessionTicket,
    SteamRichPresenceState,
    SteamworksSDKBridge,
)
from hivemind.client.worker_node import HiveMindWorkerNode
from hivemind.config import HiveMindConfig
from hivemind.core.compression import compress_lora_delta
from hivemind.core.diloco_inner import InnerLoopResult
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)
from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError


# ---------------------------------------------------------------------------
# Shared tiny synthetic Qwen2 + QLoRA fixture on CPU for sub-second execution
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_hf_network_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent any live Hugging Face Hub HTTP calls during tests."""
    monkeypatch.setattr(
        hf_miner_mod,
        "fetch_hf_domain_subset",
        lambda dataset_id, num_samples=12: [
            "### Instruction:\ndef binary_search low high while mid return mitochondria cell atp 50 hive block pub fn arc mutex hmac sha256 essaim\n"
        ]
        * max(4, min(num_samples, 6)),
    )


@pytest.fixture(scope="module")
def tiny_qwen_bundle():
    base_model, base_sha, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(
        base_model=base_model,
        rank_r=2,
        lora_alpha=4,
        gradient_checkpointing=False,
        seed=42,
    )
    init_lora = extract_lora_state_dict(bundle.model)
    return bundle, init_lora, base_sha


class _DummyEncoding(dict):
    def to(self, device: Any) -> "_DummyEncoding":
        return self


class _DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token: str | None = None
        self.eos_token: str = "<eos>"
        self.pad_token_id: int = 0

    def __call__(
        self,
        texts: str | list[str],
        truncation: bool = True,
        padding: str | bool = True,
        max_length: int = 8,
        return_tensors: str = "pt",
    ) -> _DummyEncoding:
        batch = [texts] if isinstance(texts, str) else list(texts)
        seq_len = max(4, min(max_length, 8))
        rows = []
        for idx, txt in enumerate(batch):
            seed_val = (sum(ord(c) for c in str(txt)) + idx) % 100 + 2
            rows.append([(seed_val + j) % 120 + 2 for j in range(seq_len)])
        tensor = torch.tensor(rows, dtype=torch.long)
        return _DummyEncoding({"input_ids": tensor, "attention_mask": torch.ones_like(tensor)})

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        return (
            "def binary_search low high while mid return mitochondria cell atp "
            "50 12 3 hive block = * pub fn arc mutex lock result ok hmac hashlib "
            "sha256 compare_digest verify bloc 200 4 essaim multi-krum"
        )


# ---------------------------------------------------------------------------
# 1. BOINC Workunit, Validator & Coordinator API Tests
# ---------------------------------------------------------------------------


def test_workunit_and_parquet_shard_errors(tmp_path: Path) -> None:
    shard_file = tmp_path / "shard_01.parquet"
    path, sha256, sig = create_signed_parquet_shard(
        shard_path=shard_file,
        branch_name="Code",
        rng_seed=101,
        num_sequences=4,
        seq_len=8,
        vocab_size=128,
    )
    wu = BOINCWorkunit(
        wu_id="wu-code-t1-abc123",
        branch_name="Code",
        outer_step=1,
        base_model_id="sha256_base",
        lora_checkpoint_url="ckpt.pt",
        data_shard_parquet=str(path),
        shard_sha256=sha256,
        shard_hmac_signature=sig,
        rng_seed=101,
        inner_steps_H=4,
        spot_check_interval_s=2,
        seq_len=8,
        num_sequences=4,
    )
    assert wu.to_dict()["wu_id"] == "wu-code-t1-abc123"

    loaded = verify_and_load_parquet_shard(path, sha256, sig, rng_seed=101)
    assert loaded.shape == (4, 8)

    with pytest.raises(FileNotFoundError):
        verify_and_load_parquet_shard(tmp_path / "missing.parquet", sha256, sig, rng_seed=101)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_and_load_parquet_shard(path, "0" * 64, sig, rng_seed=101)

    with pytest.raises(ValueError, match="HMAC signature"):
        verify_and_load_parquet_shard(path, sha256, "f" * 64, rng_seed=101)


def test_validator_edge_cases(tiny_qwen_bundle, tmp_path: Path) -> None:
    bundle, init_lora, base_sha = tiny_qwen_bundle
    validator = DeterministicSpotCheckValidator(tolerance_eps=1e-4)

    shard_path, sha256, sig = create_signed_parquet_shard(
        shard_path=tmp_path / "val_shard.parquet",
        branch_name="Code",
        rng_seed=77,
        num_sequences=4,
        seq_len=8,
        vocab_size=128,
    )
    wu = BOINCWorkunit(
        wu_id="wu-val-01",
        branch_name="Code",
        outer_step=1,
        base_model_id=base_sha,
        lora_checkpoint_url="ckpt.pt",
        data_shard_parquet=str(shard_path),
        shard_sha256=sha256,
        shard_hmac_signature=sig,
        rng_seed=77,
        inner_steps_H=2,
        spot_check_interval_s=1,
        seq_len=8,
        num_sequences=4,
    )
    worker = HiveMindWorkerNode(
        worker_id="worker-val-01",
        steam_id64="76561198000000010",
        persona_name="ValTester",
        model=bundle.model,
    )
    sub = worker.execute_workunit(wu, {k: v.clone() for k, v in init_lora.items()})
    tokens = verify_and_load_parquet_shard(shard_path, sha256, sig, rng_seed=77)

    verdict_ok = validator.verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=sub,
        token_sequences=tokens,
        steam_id64="76561198000000010",
    )
    assert verdict_ok.passed is True

    first_ckpt_step = sorted(s for s in sub.micro_checkpoints if s > 0)[0]
    first_key = next(iter(sub.micro_checkpoints[first_ckpt_step].state_snapshot.keys()))
    orig_ckpt_t = sub.micro_checkpoints[first_ckpt_step].state_snapshot[first_key].clone()
    sub.micro_checkpoints[first_ckpt_step].state_snapshot[first_key] = orig_ckpt_t + 0.5
    verdict_diff = validator.verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=sub,
        token_sequences=tokens,
        steam_id64="76561198000000014",
    )
    assert verdict_diff.passed is False
    sub.micro_checkpoints[first_ckpt_step].state_snapshot[first_key] = orig_ckpt_t

    saved_tensor = sub.micro_checkpoints[first_ckpt_step].state_snapshot.pop(first_key)
    verdict_missing_key = validator.verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=sub,
        token_sequences=tokens,
        steam_id64="76561198000000011",
    )
    assert verdict_missing_key.passed is False
    sub.micro_checkpoints[first_ckpt_step].state_snapshot[first_key] = saved_tensor

    first_delta_key = next(iter(sub.raw_delta.keys()))
    orig_delta = sub.raw_delta[first_delta_key].clone()
    sub.raw_delta[first_delta_key] = orig_delta + 5.0
    verdict_tampered_delta = validator.verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=sub,
        token_sequences=tokens,
        steam_id64="76561198000000012",
    )
    assert verdict_tampered_delta.passed is False
    sub.raw_delta[first_delta_key] = orig_delta

    sub.micro_checkpoints = {0: sub.micro_checkpoints[0]}
    verdict_no_ckpts = validator.verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=sub,
        token_sequences=tokens,
        steam_id64="76561198000000013",
    )
    assert verdict_no_ckpts.passed is False
    assert "zero micro-checkpoints" in verdict_no_ckpts.reason


def test_coordinator_service_and_fastapi_endpoints(tiny_qwen_bundle, tmp_path: Path) -> None:
    bundle, init_lora, base_sha = tiny_qwen_bundle
    cfg = HiveMindConfig.from_yaml()
    coordinator = BOINCCoordinatorService(
        config=cfg,
        initial_lora_state=init_lora,
        base_model_sha256=base_sha,
        shards_dir=tmp_path / "coord_shards",
        vocab_size=128,
    )

    with pytest.raises(PermissionError, match="Limited"):
        coordinator.register_or_get_node("76561198000000099", "w-lim", is_non_limited=False)

    wu1, state1 = coordinator.assign_workunit(
        worker_id="w-01",
        steam_id64="76561198000000001",
        branch_name="Code",
        inner_steps_H=2,
        spot_check_interval_s=1,
        num_sequences=4,
        seq_len=8,
    )
    wu_rr, _ = coordinator.assign_workunit(
        worker_id="w-02",
        steam_id64="76561198000000002",
        branch_name=None,
        inner_steps_H=2,
        spot_check_interval_s=1,
        num_sequences=4,
        seq_len=8,
    )
    assert wu1.branch_name == "Code"
    assert wu_rr.wu_id in coordinator.active_workunits

    worker_honest = HiveMindWorkerNode(
        worker_id="w-01",
        steam_id64="76561198000000001",
        persona_name="HonestMiner",
        model=bundle.model,
    )
    sub_honest = worker_honest.execute_workunit(wu1, state1)
    verdict_pass = coordinator.validate_and_record_submission(
        validator_model=bundle.model,
        workunit=wu1,
        submission=sub_honest,
        steam_id64="76561198000000001",
    )
    assert verdict_pass.passed is True

    # A workunit is consumed by the first spot-check verdict (FL-05 anti-replay), so the
    # Byzantine miner must receive its own workunit rather than reusing the honest one.
    wu_byz, state_byz = coordinator.assign_workunit(
        worker_id="w-byz",
        steam_id64="76561198000000066",
        branch_name="Code",
        inner_steps_H=2,
        spot_check_interval_s=1,
        num_sequences=4,
        seq_len=8,
    )
    worker_byz = HiveMindWorkerNode(
        worker_id="w-byz",
        steam_id64="76561198000000066",
        persona_name="ByzMiner",
        model=bundle.model,
    )
    sub_byz = worker_byz.execute_workunit(wu_byz, state_byz, byzantine_poison_scale=10.0)
    verdict_fail = coordinator.validate_and_record_submission(
        validator_model=bundle.model,
        workunit=wu_byz,
        submission=sub_byz,
        steam_id64="76561198000000066",
    )
    assert verdict_fail.passed is False

    with pytest.raises(PermissionError, match="Byzantine"):
        coordinator.register_or_get_node("76561198000000066", "w-byz")

    with pytest.raises(SwarmSecurityViolationError, match="WORKUNIT_REPLAY_DETECTED"):
        coordinator.validate_and_record_submission(
            validator_model=bundle.model,
            workunit=wu1,
            submission=sub_honest,
            steam_id64="76561198000000001",
        )

    with pytest.raises(ValueError, match="No validated worker deltas"):
        coordinator.step_branch_outer_loop("Mathematics")

    base_delta = coordinator.pending_branch_deltas["Code"]["w-01"]
    for extra_w in ("w-02", "w-03", "w-04"):
        coordinator.inject_raw_delta_for_aggregation(
            "Code", extra_w, {k: v.clone() for k, v in base_delta.items()}
        )
    outer_res = coordinator.step_branch_outer_loop("Code")
    assert outer_res.outer_step == 1

    app = create_coordinator_fastapi_app(coordinator)
    client = TestClient(app)

    assert client.get("/status").status_code == 200
    assert (
        client.post(
            "/workunit/request",
            json={"worker_id": "w-api-1", "steam_id64": "76561198000000001", "preferred_branch": "Code"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/workunit/request",
            json={"worker_id": "w-byz", "steam_id64": "76561198000000066"},
        ).status_code
        == 403
    )
    assert client.get("/reputation").status_code == 200

    r_chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "hivemind-qwen2.5-sparse-moe-top2",
            "messages": [{"role": "user", "content": "Write a Python function to compute HIVE blocks."}],
            "stream": False,
        },
    )
    assert r_chat.status_code == 200

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "hivemind-qwen2.5-sparse-moe-top2",
            "prompt": "Implement a Rust Arc Mutex counter for HIVE blocks.",
            "stream": True,
        },
    ) as r_sse:
        assert r_sse.status_code == 200
        body = "".join(list(r_sse.iter_text()))
        assert "data: [DONE]" in body


# ---------------------------------------------------------------------------
# 2. Client WorkerNode, ZeroIntrusionDaemon & Steamworks SDK Bridge Tests
# ---------------------------------------------------------------------------


def test_worker_node_lifecycle_and_cas_shard(tiny_qwen_bundle) -> None:
    bundle, init_lora, _ = tiny_qwen_bundle

    limited_worker = HiveMindWorkerNode(
        worker_id="w-limited",
        steam_id64="76561198000000055",
        persona_name="LimitedUser",
        model=bundle.model,
        account_spend_usd=1.0,
    )
    with pytest.raises(PermissionError, match="Steam Sybil guard"):
        limited_worker.execute_workunit(MagicMock(), init_lora)

    with pytest.raises(PermissionError, match="Steam Sybil guard"):
        limited_worker.execute_cas_shard(
            domain_name="Code",
            token_sequences=torch.ones((4, 8), dtype=torch.long),
            global_lora_state=init_lora,
        )

    valid_worker = HiveMindWorkerNode(
        worker_id="w-valid",
        steam_id64="76561198000000001",
        persona_name="ValidMiner",
        model=bundle.model,
        account_spend_usd=49.99,
    )
    assert valid_worker.get_seti_lifecycle_state(consumer_query_active=False) == (
        SETIAgentState.STATE_2_IDLE_SETI_MINER.value
    )
    assert valid_worker.get_seti_lifecycle_state(consumer_query_active=True) == (
        SETIAgentState.STATE_3_SWARM_CONSUMER.value
    )
    cas_res = valid_worker.execute_cas_shard(
        domain_name="Code",
        token_sequences=torch.ones((4, 8), dtype=torch.long),
        global_lora_state={k: v.clone() for k, v in init_lora.items()},
        inner_steps_H=2,
        spot_check_interval_s=1,
    )
    assert cas_res.steps_completed == 2


def test_daemon_os_hooks_and_process_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daemon_mod.sys, "platform", "linux")
    assert get_windows_idle_seconds() == 999.0

    monkeypatch.setattr(daemon_mod.sys, "platform", "win32")
    monkeypatch.setattr(
        daemon_mod.ctypes,
        "sizeof",
        MagicMock(side_effect=RuntimeError("Win32 hook failure")),
    )
    assert get_windows_idle_seconds() == 999.0

    class _DeadProc:
        @property
        def info(self) -> dict[str, str]:
            raise psutil.NoSuchProcess(pid=1234)

    class _GameProc:
        info = {"name": "CustomGame.exe"}

    monkeypatch.setattr(
        daemon_mod.psutil,
        "process_iter",
        lambda attrs: [_DeadProc(), _GameProc()],
    )
    assert detect_running_game_process(extra_watchlist={"customgame.exe"}) == "customgame.exe"

    d_game = ZeroIntrusionDaemon(simulated_active_game="cs2.exe", simulated_idle_seconds=100.0)
    assert d_game.should_preempt() is True

    d_input = ZeroIntrusionDaemon(
        idle_threshold_seconds=30.0,
        enforce_os_idle_hook=True,
        simulated_idle_seconds=5.0,
    )
    assert d_input.poll().should_yield_vram is True


def test_steamworks_sdk_bridge_all_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bridge = SteamworksSDKBridge()
    monkeypatch.setattr(steamworks_mod.platform, "system", lambda: "Darwin")
    bridge._try_load_native_steam_dll(sdk_search_dir=tmp_path)
    monkeypatch.setattr(steamworks_mod.platform, "system", lambda: "Linux")
    bridge._try_load_native_steam_dll(sdk_search_dir=tmp_path)

    fake_dll = tmp_path / "libsteam_api.so"
    fake_dll.write_bytes(b"fake")

    class _FakeCDLL:
        def __init__(self) -> None:
            self.SteamAPI_Init = MagicMock(return_value=True)
            self.SteamAPI_ISteamFriends_SetRichPresence = MagicMock()

    monkeypatch.setattr(steamworks_mod.ctypes, "CDLL", lambda path: _FakeCDLL())
    status_loaded = bridge._try_load_native_steam_dll(sdk_search_dir=tmp_path)
    assert status_loaded["native_dll_loaded"] is True

    rp1 = bridge.set_rich_presence(
        "76561198000000001",
        dataset_name="sahil2801/CodeAlpaca-20k",
        gpu_model="RTX 3080",
        hive_earned=88.5,
    )
    assert rp1.native_dll_broadcast is True

    bridge.native_dll_handle.SteamAPI_ISteamFriends_SetRichPresence.side_effect = RuntimeError("dll error")
    rp2 = bridge.set_rich_presence("76561198000000001", persona_name="SoloMission", mission_name=None)
    assert rp2.native_dll_broadcast is False

    rp3 = bridge.set_rich_presence(
        "76561198000000001",
        "qiaojin/PubMedQA",
        "RTX 2070",
        "Rank V: Singularity Core",
        55.0,
    )
    assert rp3.active_mission == "qiaojin/PubMedQA"

    stats = bridge.update_user_stats(
        steam_id64="76561198000000001",
        persona_name="CyberValkyrie_FR",
        shards_mined=4,
        bounties_cracked=3,
        best_loss_drop_pct=78.5,
        boss_raids_cleared=1,
        hive_mined=120.0,
    )
    assert stats["stat_shards_mined"] == 4

    receipt = bridge.upload_hack_miner_score(
        steam_id64="76561198000000001",
        persona_name="CyberValkyrie_FR",
        total_points=12500,
        swarm_contribution_points=4200,
        hive_balance=120.0,
        achievements=["FIRST_HF_SHARD_BREACH", "LORA_LOSS_CRUSHER"],
    )
    assert receipt.startswith("steam_lb_hmac_")
    assert len(bridge.overlay_notifications) >= 2

    monkeypatch.setattr(
        steamworks_mod.ctypes,
        "CDLL",
        MagicMock(side_effect=OSError("bad dll")),
    )
    bridge2 = SteamworksSDKBridge(sdk_search_dir=tmp_path)
    assert bridge2.native_dll_loaded is False

    bad_ticket = SteamAuthSessionTicket(
        steam_id64="76561198000000001",
        app_id=3133700,
        account_spend_usd=25.0,
        is_vac_banned=False,
        ticket_hex="00" * 32,
    )
    ok_sig, _ = bridge.verify_sybil_resistance(bad_ticket)
    assert ok_sig is False

    tel = bridge.record_validated_workunit(
        steam_id64="76561198000000001",
        persona_name="CyberValkyrie_FR",
        wu_id="wu-1",
        tflops_contributed=12.5,
        credits_minted=10.0,
    )
    assert "ACH_10_TFLOPS_SWARM" in tel.unlocked_achievements


# ---------------------------------------------------------------------------
# 3. Geo-Ring WebGL Geometry & Steam Edition Package / FastAPI Server Tests
# ---------------------------------------------------------------------------


def test_georing_webgl_helpers() -> None:
    assert georing_mod.guild_raid_arc_color("GUILD-01") == [1.0, 0.3, 0.52]
    custom_col = georing_mod.guild_raid_arc_color("GUILD-99-CUSTOM")
    assert len(custom_col) == 3

    pts = georing_mod.great_circle_arc_points([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], steps=0)
    assert len(pts) == 2

    geom = georing_mod.build_guild_raid_arc_geometry(
        [
            {
                "arc_id": "ARC-1",
                "source_guild": "GUILD-99",
                "target_guild": "GUILD-01",
                "points": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            }
        ]
    )
    assert geom[0]["steps"] == 1


def test_steam_edition_package_and_fastapi_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    art_dir = tmp_path / "artifacts_test"
    monkeypatch.setattr(steam_edition_mod, "ARTIFACTS_ROOT", art_dir)
    pkg_out = tmp_path / "steam_pkg"
    pkg_info = steam_edition_mod.build_free_steam_edition_package(
        game_report=None,
        output_root=pkg_out,
    )
    assert Path(pkg_info["interactive_hud_html"]).exists()

    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "hf_hack_miner_game_report.json").write_text(
        json.dumps({"game_title": "SavedReport", "leaderboard": [], "missions": []}),
        encoding="utf-8",
    )
    pkg_info2 = steam_edition_mod.build_free_steam_edition_package(
        game_report=None,
        output_root=pkg_out,
    )
    assert Path(pkg_info2["steam_store_manifest"]).exists()

    monkeypatch.setattr(
        steam_edition_mod,
        "build_free_steam_edition_package",
        lambda game_report=None: pkg_info,
    )
    app = steam_edition_mod.create_hack_miner_fastapi_app(
        game_report={"game_title": "Test", "leaderboard": [], "missions": []}
    )
    client = TestClient(app)

    for ep in (
        "/",
        "/api/leaderboard",
        "/api/missions",
        "/api/season4-missions",
        "/api/season5-missions",
        "/api/guild-tournament",
        "/api/georing-topology",
        "/api/rich-presence",
        "/api/rtx-tool-status",
        "/api/steam-manifest",
        "/api/steam-cloud",
    ):
        assert client.get(ep).status_code == 200

    assert client.post("/api/run-rtx-diagnostic", json={}).status_code == 200
    assert (
        client.post(
            "/api/guild-tournament",
            json={
                "guild_id": "GUILD-01",
                "multi_domain_holdout_perplexity": 4.5,
                "speculative_tokens_per_sec": 32.0,
                "gamer_tag": "CyberValkyrie_FR",
            },
        ).status_code
        == 200
    )
    assert client.post("/api/guild-tournament", json={"guild_id": "NO-SUCH-GUILD"}).status_code == 404
    assert (
        client.post(
            "/api/guild-tournament",
            json={"guild_id": "GUILD-01", "multi_domain_holdout_perplexity": "bad_float"},
        ).status_code
        == 400
    )

    assert (
        client.post(
            "/api/scan-hf-shard",
            json={
                "gamer_tag": "CyberValkyrie_FR",
                "dataset_id": "sahil2801/CodeAlpaca-20k",
                "custom_rows": ["### Instruction:\ndef f(): return 1\n"],
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/forge-lora-mission",
            json={"gamer_tag": "CyberValkyrie_FR", "contract_id": "CONTRACT-HF-17-CUDA-ROCM-PTX-OPT"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/solve-bounty",
            json={"gamer_tag": "CyberValkyrie_FR", "prompt": "Write a Python function to compute HIVE."},
        ).status_code
        == 200
    )

    from hivemind.swarm.scientific_swarm_eval import JoinedSwarmCampaignEngine

    monkeypatch.setattr(
        JoinedSwarmCampaignEngine,
        "execute_full_scientific_swarm_campaign",
        lambda self, **kw: {"proposed_joined_campaigns": [], "dual_leaderboard": {}},
    )
    assert (
        client.post(
            "/api/joined-campaign",
            json={"lead_gamer_tag": "CyberValkyrie_FR", "campaign_id": "CAMPAIGN-COOP-02-POLYMATH-FUSION"},
        ).status_code
        == 200
    )

    monkeypatch.setattr(steam_edition_mod, "SWARM_TELEMETRY_INTERVAL_SECONDS", 0.002)
    state_file = Path(pkg_info["interactive_hud_html"]).parent / "live_leaderboard_state.json"
    state_file.write_text("{corrupt_json", encoding="utf-8")
    with client.websocket_connect("/ws/swarm-telemetry") as ws:
        frame0 = ws.receive_json()
        frame1 = ws.receive_json()
        assert frame0["seq"] == 0
        assert frame1["seq"] == 1

    ws_route = [r for r in app.routes if getattr(r, "path", "") == "/ws/swarm-telemetry"][0]

    class _BrokenWS:
        async def accept(self) -> None:
            return None

        async def receive(self) -> dict[str, Any]:
            raise RuntimeError("receive broken")

        async def send_json(self, data: Any) -> None:
            await asyncio.sleep(0)
            raise RuntimeError("send broken")

    asyncio.run(ws_route.endpoint(_BrokenWS()))


# ---------------------------------------------------------------------------
# 4. HF Hack-Miner Campaign, Interactive Helpers & Perplexity Edge Cases
# ---------------------------------------------------------------------------


def test_hf_hack_miner_perplexity_and_profile_edge_cases(tmp_path: Path) -> None:
    assert hf_miner_mod.estimate_bigram_holdout_perplexity(["only_one_row"])["measured"] is False
    assert (
        hf_miner_mod.estimate_bigram_holdout_perplexity(["alpha beta gamma", "singleword"])["measured"]
        is False
    )

    assert len(hf_miner_mod.fetch_season4_curated_corpus("hivemind/triton-int4-fp8-kernels", 3)) == 3
    assert len(hf_miner_mod.fetch_offline_curated_corpus("hivemind/triton-int4-fp8-kernels", 3)) == 3
    assert hf_miner_mod.fetch_offline_curated_corpus("unknown/dataset", 3) == []
    assert hf_miner_mod.build_guild_tournament_state()["season"] == 5

    prof = hf_miner_mod.GamerMinerProfile(
        gamer_tag="CyberValkyrie_FR",
        steam_id64="76561198000000001",
        node_id="seti-node-paris-01",
        region="EU-West-Paris",
        specialty_focus="All",
        best_loss_reduction_pct=75.0,
        season5_missions_forged=1,
        boss_raids_cleared=1,
        domains_forged=[
            "Rust_Systems_Security",
            "CyberSecurity_Vulnerability_Recon",
            "Multilingual_FR_Synthesis",
        ],
    )
    prof.update_rank_and_achievements(is_top_miner=True, num_moe_experts=6)
    assert "LORA_LOSS_CRUSHER" in prof.achievements_unlocked
    assert "HF_COOP_RAID_COMMANDER" in prof.achievements_unlocked
    assert "OMEGA_6_EXPERT_SLAYER" in prof.achievements_unlocked

    store_dir = tmp_path / "swarm_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / "hive_gamified_ledger.json").write_text("{}", encoding="utf-8")
    eng = hf_miner_mod.HackMiningGameEngine(store_dir=store_dir, reset_ledger=True)
    assert len(eng.ledger.chain) == 1


def test_interactive_helpers_and_leaderboard_file_updates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from hivemind.swarm.scientific_swarm_eval import JoinedSwarmCampaignEngine

    monkeypatch.setattr(
        JoinedSwarmCampaignEngine,
        "execute_full_scientific_swarm_campaign",
        lambda self, **kw: {"proposed_joined_campaigns": [], "dual_leaderboard": {}},
    )
    monkeypatch.setattr(hf_miner_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(hf_miner_mod, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    hud_dir = tmp_path / "web_ui"
    hud_dir.mkdir(parents=True, exist_ok=True)
    hud_html = hud_dir / "index.html"
    hud_html.write_text("<html></html>", encoding="utf-8")

    initial_report = {
        "leaderboard": [
            {
                "gamer_tag": "CyberValkyrie_FR",
                "steam_id64": "76561198000000001",
                "rank_title": "Rank V",
                "total_points": 1000,
                "data_discovery_points": 100,
                "lora_training_points": 100,
                "solution_bounty_points": 100,
                "swarm_contribution_points": 100,
                "hive_balance": 50.0,
                "best_loss_reduction_pct": 65.0,
                "shards_mined": 1,
                "bounties_solved": 1,
                "swarm_tokens_served": 10,
                "achievements_unlocked": [],
            }
        ],
        "free_steam_package": {"interactive_hud_html": str(hud_html)},
    }
    report_path = tmp_path / "hf_hack_miner_game_report.json"
    report_path.write_text(json.dumps(initial_report), encoding="utf-8")

    scan_res = hf_miner_mod.execute_interactive_hf_shard_scan(
        gamer_tag="CyberValkyrie_FR",
        dataset_id="sahil2801/CodeAlpaca-20k",
        custom_rows=["### Instruction:\ndef f(): return 50\n"],
    )
    assert scan_res["merkle_pos_verified"] is True
    assert (hud_dir / "live_leaderboard_state.json").exists()

    r_cyber = hf_miner_mod.execute_interactive_bounty_probe(
        gamer_tag="CyberValkyrie_FR",
        prompt="cve security vulnerability sql injection hmac patch firewall timing audit",
    )
    assert "compare_digest" in r_cyber["generated_solution"]

    r_fr = hf_miner_mod.execute_interactive_bounty_probe(
        gamer_tag="CyberValkyrie_FR",
        prompt="expliquez calculez essaim bloc jeton scientifique français récompense preuve",
    )
    assert "Synthèse" in r_fr["generated_solution"]

    r_bio = hf_miner_mod.execute_interactive_bounty_probe(
        gamer_tag="CyberValkyrie_FR",
        prompt="cell mitochondria atp clinical biomedical disease gene protein patient",
    )
    assert "Mitochondrial" in r_bio["generated_solution"]

    orig_sorted = sorted
    monkeypatch.setattr(
        "builtins.sorted",
        lambda iterable, *a, **kw: [("Python_Systems_Code", 1.0)]
        if kw.get("reverse") and isinstance(iterable, dict | type({}.items()))
        else orig_sorted(iterable, *a, **kw),
    )
    r_else = hf_miner_mod.execute_interactive_bounty_probe(
        gamer_tag="CyberValkyrie_FR",
        prompt="generic query",
    )
    assert "binary_search_swarm" in r_else["generated_solution"]
    monkeypatch.undo()

    monkeypatch.setattr(
        JoinedSwarmCampaignEngine,
        "execute_full_scientific_swarm_campaign",
        lambda self, **kw: {"proposed_joined_campaigns": [], "dual_leaderboard": {}},
    )
    monkeypatch.setattr(hf_miner_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(hf_miner_mod, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        hf_miner_mod,
        "fetch_hf_domain_subset",
        lambda dataset_id, num_samples=12: ["### Rust:\npub fn arc mutex lock ok\n"] * 4,
    )
    f_s2 = hf_miner_mod.execute_interactive_lora_micro_forge(
        gamer_tag="CyberValkyrie_FR",
        contract_id="CONTRACT-HF-04-RUST-SYSTEMS",
    )
    assert f_s2["season"] == 2

    f_s5 = hf_miner_mod.execute_interactive_lora_micro_forge(
        gamer_tag="CyberValkyrie_FR",
        contract_id="CONTRACT-HF-17-CUDA-ROCM-PTX-OPT",
    )
    assert f_s5["season"] == 5
    assert f_s5["guild_wars_badge"] == "HF_COOP_RAID_COMMANDER"

    coop_rep = hf_miner_mod.execute_interactive_joined_campaign(
        lead_gamer_tag="CyberValkyrie_FR",
        campaign_id="CAMPAIGN-COOP-02-POLYMATH-FUSION",
        inner_steps=2,
    )
    assert coop_rep["lead_gamer_tag"] == "CyberValkyrie_FR"


def test_run_hf_hack_mining_game_all_seasons_fast(
    monkeypatch: pytest.MonkeyPatch, tiny_qwen_bundle, tmp_path: Path
) -> None:
    bundle, init_lora, _ = tiny_qwen_bundle
    monkeypatch.setattr(hf_miner_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(hf_miner_mod, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(steam_edition_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(steam_edition_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hf_miner_mod.torch.cuda, "is_available", lambda: False)

    # Allow dataclasses.asdict inside hf_hack_miner to serialize SteamRichPresenceState (str subclass)
    orig_asdict = hf_miner_mod.asdict
    monkeypatch.setattr(
        hf_miner_mod,
        "asdict",
        lambda obj: dict(obj.__dict__) if isinstance(obj, SteamRichPresenceState) else orig_asdict(obj),
    )

    monkeypatch.setattr(
        hf_miner_mod.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: _DummyTokenizer(),
    )

    def _fast_stream_load(*args: Any, **kwargs: Any):
        base_m, sha, is_nf4 = build_local_qwen2_backbone(
            seed=42,
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            device=torch.device("cpu"),
            use_4bit_nf4_if_cuda=False,
        )
        return base_m, sha, is_nf4

    monkeypatch.setattr(hf_miner_mod, "stream_load_hf_causal_lm", _fast_stream_load)

    orig_router_train = hf_miner_mod.train_moe_router_gating
    monkeypatch.setattr(
        hf_miner_mod,
        "train_moe_router_gating",
        lambda moe_model, domain_token_batches, steps=15, lr=5e-3: orig_router_train(
            moe_model=moe_model,
            domain_token_batches=domain_token_batches,
            steps=1,
            lr=lr,
        ),
    )
    orig_spec = hf_miner_mod.RegionalDHTSwarmRing.run_speculative_inference
    monkeypatch.setattr(
        hf_miner_mod.RegionalDHTSwarmRing,
        "run_speculative_inference",
        lambda self, moe_target_model, draft_model, prompt_ids, max_new_tokens=28, draft_gamma=5, simulated_network_rtt_ms=14.1: orig_spec(
            self,
            moe_target_model=moe_target_model,
            draft_model=draft_model,
            prompt_ids=prompt_ids,
            max_new_tokens=3,
            draft_gamma=2,
            simulated_network_rtt_ms=simulated_network_rtt_ms,
        ),
    )

    from hivemind.swarm.scientific_swarm_eval import JoinedSwarmCampaignEngine

    monkeypatch.setattr(
        JoinedSwarmCampaignEngine,
        "execute_full_scientific_swarm_campaign",
        lambda self, **kw: {"proposed_joined_campaigns": [], "dual_leaderboard": {}},
    )

    # 1. Full execution on Season 1 (3 missions -> E=2 upcycle + 1 hotplug) with build_steam_package=True
    rep_s1 = hf_miner_mod.run_hf_hack_mining_game(
        model_id="Qwen/Qwen2.5-0.5B",
        lora_rank=2,
        inner_steps=2,
        seq_len=8,
        season="s1",
        build_steam_package=True,
    )
    assert rep_s1["moe_boss_fusion"]["num_experts"] == 3
    assert "free_steam_package" in rep_s1

    # 2. Stub inner training/validation/MoE fusion for remaining season branch selectors (`s2`..`all`)
    two_contracts = hf_miner_mod.SEASON_1_HF_BOUNTY_CONTRACTS[:2]
    for attr in (
        "SEASON_2_HF_BOUNTY_CONTRACTS",
        "SEASON_3_HF_BOUNTY_CONTRACTS",
        "SEASON_4_HF_BOUNTY_CONTRACTS",
        "SEASON_5_HF_BOUNTY_CONTRACTS",
        "FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS",
        "SEASON_4_FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS",
        "SEASON_5_FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS",
        "ALL_HF_BOUNTY_CONTRACTS",
    ):
        monkeypatch.setattr(hf_miner_mod, attr, two_contracts)

    dummy_delta = {k: torch.ones_like(v, dtype=torch.float32) * 0.01 for k, v in init_lora.items()}
    dummy_comp = compress_lora_delta(dummy_delta, "w", "Code", 1, topk_ratio=0.5)

    class _FastStubModel:
        def eval(self) -> None:
            return None

        def generate(self, **kw: Any) -> torch.Tensor:
            return torch.ones((1, 8), dtype=torch.long)

    monkeypatch.setattr(
        hf_miner_mod,
        "attach_qlora_adapters",
        lambda **kw: SimpleNamespace(model=_FastStubModel(), base_model_sha256="sha_stub"),
    )
    monkeypatch.setattr(
        hf_miner_mod,
        "extract_lora_state_dict",
        lambda m: {k: v.clone() for k, v in init_lora.items()},
    )
    monkeypatch.setattr(hf_miner_mod, "load_lora_state_dict", lambda m, s: None)

    def _instant_inner_loop(**kw: Any) -> InnerLoopResult:
        h = kw.get("inner_steps_H", 2)
        return InnerLoopResult(
            worker_id=kw["worker_id"],
            branch_name=kw["branch_name"],
            outer_step=kw["outer_step"],
            steps_completed=h,
            initial_loss=3.0,
            final_loss=1.2,
            elapsed_seconds=0.01,
            theta_init={},
            theta_final={},
            raw_delta={},
            compressed_delta=dummy_comp,
            micro_checkpoints={h: SimpleNamespace(lora_sha256="sha_ok", state_snapshot={})},
        )

    monkeypatch.setattr(hf_miner_mod, "run_diloco_inner_loop", _instant_inner_loop)
    monkeypatch.setattr(
        hf_miner_mod.DeterministicSpotCheckValidator,
        "verify_submission",
        lambda self, **kw: SpotCheckVerdict(
            wu_id="wu",
            worker_id="w",
            steam_id64="s",
            segment_start_step=0,
            segment_end_step=2,
            verification_overhead_pct=10.0,
            max_abs_divergence=0.0,
            tolerance_eps=1e-4,
            hash_matched=True,
            passed=True,
            steam_account_flagged=False,
            reason="ok",
        ),
    )

    class _DummyMoE:
        num_experts = 2
        expert_domains = ["Python_Systems_Code", "Biomedical_Clinical_QA"]

        def hotplug_domain_expert(self, **kw: Any) -> None:
            self.num_experts += 1

    monkeypatch.setattr(
        hf_miner_mod,
        "upcycle_branches_to_sparse_moe",
        lambda **kw: (_DummyMoE(), None),
    )
    monkeypatch.setattr(
        hf_miner_mod,
        "train_moe_router_gating",
        lambda **kw: (2.5, 1.5, {"Python_Systems_Code": 0.5, "Biomedical_Clinical_QA": 0.5}),
    )
    monkeypatch.setattr(
        hf_miner_mod.RegionalDHTSwarmRing,
        "run_speculative_inference",
        lambda self, **kw: SimpleNamespace(
            generated_tokens=[10, 20, 30, 40],
            expert_routing_share={"Python_Systems_Code": 0.5, "Biomedical_Clinical_QA": 0.5},
            draft_gamma=5,
            latency_reduction_factor=5.2,
            region_ring="EU-West-Swarm-Ring",
            active_ring_peers=3,
            mean_ring_ping_ms=14.1,
            swarm_verification_passes=2,
            mean_acceptance_rate=0.9,
            effective_tokens_per_sec=28.5,
        ),
    )

    _cpu_dev = torch.device("cpu")
    for s_flag in ("s2", "s3", "s4", "s5", "s1-s3", "s1-s4", "s1-s5", "all"):
        if s_flag == "all":
            monkeypatch.setattr(
                hf_miner_mod,
                "detect_local_hardware",
                lambda: {
                    "gpu_name": "NVIDIA GeForce RTX 2070",
                    "vram_total_gb": 8.0,
                    "vram_gb": 8.0,
                    "backend": "cuda",
                },
            )
            monkeypatch.setattr(hf_miner_mod.torch.cuda, "is_available", lambda: True)
            monkeypatch.setattr(hf_miner_mod.torch.cuda, "reset_peak_memory_stats", lambda *a: None)
            monkeypatch.setattr(hf_miner_mod.torch.cuda, "empty_cache", lambda: None)
            monkeypatch.setattr(hf_miner_mod.torch, "device", lambda *a, **kw: _cpu_dev)
            monkeypatch.setattr(
                hf_miner_mod,
                "measure_vram_footprint",
                lambda dev: {"allocated_mb": 10.0, "peak_allocated_mb": 12.0, "free_vram_gb": 7.5},
            )
        r = hf_miner_mod.run_hf_hack_mining_game(
            model_id="Qwen/Qwen2.5-0.5B",
            lora_rank=2,
            inner_steps=2,
            seq_len=8,
            season=s_flag,
            build_steam_package=False,
        )
        assert len(r["missions"]) >= 2


# ---------------------------------------------------------------------------
# 5. RTX Steam Tool Deployer & 10-Stage Stress Suite Fast Test
# ---------------------------------------------------------------------------


def test_rtx_steam_tool_deployer_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(rtx_deployer_mod, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(rtx_deployer_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    # Mock slow Lean 4 subprocess compiler audit in Stage 10 so suite runs in <1s
    monkeypatch.setattr(
        "paper.autoevolve_peer_review_pipeline.verify_lean4_specification",
        lambda: {
            "theorem_count": 32,
            "sorry_count": 0,
            "user_axiom_count": 0,
            "tier_a_machine_checked": True,
            "theorems": [
                "prefix_trie_shared_nodes_reduce_attention_flops",
                "quic_0rtt_anti_replay_rejects_duplicate_nonce",
                "fused_moe_fp8_vram_fits_16_experts_in_8gb",
                "nat_hole_punch_direct_p2p_avoids_turn_relay_latency",
                "prefix_trie_dedup_node_bound",
                "rfc9001_0rtt_monotonic_nonce_anti_replay",
            ],
        },
    )

    def _tiny_qwen2(seed: int = 42, **kwargs: Any):
        return build_local_qwen2_backbone(
            seed=seed,
            vocab_size=2048,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            device=torch.device("cpu"),
            use_4bit_nf4_if_cuda=False,
        )

    orig_attach = rtx_deployer_mod.attach_qlora_adapters
    monkeypatch.setattr(rtx_deployer_mod, "build_local_qwen2_backbone", _tiny_qwen2)
    monkeypatch.setattr(
        rtx_deployer_mod,
        "attach_qlora_adapters",
        lambda base_model, rank_r=64, lora_alpha=128, gradient_checkpointing=False, seed=42: orig_attach(
            base_model=base_model,
            rank_r=2,
            lora_alpha=4,
            gradient_checkpointing=False,
            seed=seed,
        ),
    )

    stress_rep = rtx_deployer_mod.run_intense_rtx_stress_suite(
        vram_cycles=1,
        udp_burst_packets=2,
        num_hotplug_experts=3,
        steam_concurrent_accounts=50,
    )
    assert stress_rep["stages_executed"] == 10
    assert stress_rep["all_stages_passed"] is True

    monkeypatch.setattr(
        rtx_deployer_mod,
        "run_intense_rtx_stress_suite",
        lambda: stress_rep,
    )
    pkg = rtx_deployer_mod.deploy_free_steam_rtx_tool_v1(
        stress_report=None,
        output_root=tmp_path / "rtx_tool_pkg",
    )
    assert pkg["all_stress_stages_passed"] is True
    assert Path(pkg["steam_tool_vdf"]).exists()
    assert Path(pkg["rtx_deployment_cert_json"]).exists()
