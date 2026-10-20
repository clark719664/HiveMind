"""Comprehensive fast unit and integration tests for CLI entrypoints and configuration (`HM-COV-1`).

Covers:
1. `src/hivemind/config.py`
2. `src/hivemind/cli/main.py`
3. `src/hivemind/cli/simulate_e2e.py`
4. `src/hivemind/cli/poc_rtx2070.py`
5. `src/hivemind/cli/demo_hf_domain_swarm.py`
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import hivemind.config as hm_config
from hivemind.core.quantization import build_local_qwen2_backbone


# ---------------------------------------------------------------------------
# Lightweight Dummy Tokenizer & Tiny Qwen2 Model Helpers
# ---------------------------------------------------------------------------


class _DummyBatch(dict):
    """Dict subclass supporting `.to(device)` and keyword unpacking for `model.generate(**inp)`."""

    def to(self, device: Any) -> "_DummyBatch":
        return _DummyBatch({k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()})


class DummyTokenizer:
    """Fast deterministic tokenizer replacement avoiding Hugging Face Hub network I/O."""

    def __init__(self) -> None:
        self.pad_token: str | None = None
        self.eos_token: str = "<eos>"
        self.pad_token_id: int = 0

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ) -> Any:
        ids = [((ord(c) % 200) + 2) for c in (text or "hivemind")[:32]]
        if not ids:
            ids = [2, 3, 4, 5]
        if return_tensors == "pt":
            return torch.tensor([ids[:8]], dtype=torch.long)
        return ids

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        return " def solved(): return 42"

    def __call__(
        self,
        texts: str | list[str],
        truncation: bool = True,
        padding: str | bool = "max_length",
        max_length: int = 16,
        return_tensors: str = "pt",
    ) -> _DummyBatch:
        if isinstance(texts, str):
            batch_size = 1
            seq_len = min(8, max_length)
        else:
            batch_size = max(1, len(texts))
            seq_len = max_length
        gen = torch.Generator().manual_seed(42)
        input_ids = torch.randint(2, 200, (batch_size, seq_len), generator=gen, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return _DummyBatch({"input_ids": input_ids, "attention_mask": attention_mask})


def _make_tiny_qwen2(device: torch.device | None = None, vocab_size: int = 256):
    """Build a tiny 1-layer Qwen2 model on CPU in <10ms for fast pipeline testing."""
    dev = device or torch.device("cpu")
    model, sha, is_nf4 = build_local_qwen2_backbone(
        seed=42,
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        device=dev,
        use_4bit_nf4_if_cuda=False,
    )
    # Speed up `.generate()` calls in tests
    model.generate = lambda input_ids, max_new_tokens=4, **kwargs: torch.cat(  # type: ignore[method-assign]
        [
            input_ids,
            torch.full(
                (input_ids.shape[0], min(4, int(max_new_tokens))),
                7,
                dtype=torch.long,
                device=input_ids.device,
            ),
        ],
        dim=1,
    )
    return model, sha, is_nf4


# ===========================================================================
# 1. Tests for `src/hivemind/config.py`
# ===========================================================================


def test_config_caches_and_hf_token_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Cover exception handler in ensure_local_drive_caches when dotenv fails
    with patch.dict("sys.modules", {"dotenv": None}):
        caches = hm_config.ensure_local_drive_caches()
        assert "HF_HOME" in caches

    # Direct env token branch (lines 70-72)
    monkeypatch.setenv("HF_TOKEN", "  hf_env_token_abc  ")
    assert hm_config.get_hf_token() == "hf_env_token_abc"

    # Clear env tokens to test Windows registry & huggingface_hub fallbacks
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN"):
        monkeypatch.delenv(k, raising=False)

    if os.name == "nt":
        import winreg

        # Registry hit branch (lines 87-90)
        dummy_ctx = MagicMock()
        dummy_ctx.__enter__.return_value = "key_handle"
        dummy_ctx.__exit__.return_value = False
        with patch.object(winreg, "OpenKey", return_value=dummy_ctx), patch.object(
            winreg, "QueryValueEx", return_value=("  hf_reg_token_xyz  ", 1)
        ):
            assert hm_config.get_hf_token() == "hf_reg_token_xyz"

        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Registry FileNotFoundError + OSError + huggingface_hub.get_token hit (lines 91-94, 101-102)
        def _open_key_side_effect(hive: Any, subkey: str) -> Any:
            if hive == winreg.HKEY_CURRENT_USER:
                return dummy_ctx
            raise OSError("registry locked")

        with patch.object(winreg, "OpenKey", side_effect=_open_key_side_effect), patch.object(
            winreg, "QueryValueEx", side_effect=FileNotFoundError("no value")
        ), patch("huggingface_hub.get_token", return_value="hf_hub_cached_tok"):
            assert hm_config.get_hf_token() == "hf_hub_cached_tok"

        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Registry OSError + huggingface_hub.get_token raising Exception -> None (lines 103-105)
        with patch.object(winreg, "OpenKey", side_effect=OSError("no reg")), patch(
            "huggingface_hub.get_token", side_effect=RuntimeError("hub error")
        ):
            assert hm_config.get_hf_token() is None


def test_hivemind_config_yaml_and_fallback_branches(tmp_path: Path) -> None:
    # Default YAML load
    cfg_default = hm_config.HiveMindConfig.from_yaml()
    assert cfg_default.codename == "Project HiveMind"
    assert "rtx2070_7b_worker" in cfg_default.profiles

    # Non-existent YAML path (line 221)
    cfg_missing = hm_config.HiveMindConfig.from_yaml(tmp_path / "does_not_exist.yaml")
    assert cfg_missing.inner_steps_H == 500

    # Custom YAML without `vram_breakdown_estimate_gb` (lines 242-249)
    custom_yaml = tmp_path / "custom_profile.yaml"
    custom_yaml.write_text(
        "profiles:\n  minimal_worker:\n    description: Minimal\n    target_vram_gb: 8.0\n",
        encoding="utf-8",
    )
    cfg_custom = hm_config.HiveMindConfig.from_yaml(custom_yaml)
    assert "minimal_worker" in cfg_custom.profiles
    assert cfg_custom.profiles["minimal_worker"].vram_budget.total_peak_vram_gb > 0


def test_gpu_backend_rocm_mps_and_hardware_detection_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    # MPS availability & built edge cases (lines 348, 351-352, 361, 364-365)
    with patch.object(torch.backends, "mps", None, create=True):
        assert hm_config.is_mps_available() is False
        assert hm_config.is_mps_built() is False

    broken_mps = SimpleNamespace(
        is_available=MagicMock(side_effect=RuntimeError("mps err")),
        is_built=MagicMock(side_effect=RuntimeError("mps build err")),
    )
    with patch.object(torch.backends, "mps", broken_mps, create=True):
        assert hm_config.is_mps_available() is False
        assert hm_config.is_mps_built() is False

    # resolve_default_device branches (lines 374-376)
    with patch("torch.cuda.is_available", return_value=False), patch(
        "hivemind.config.is_mps_available", return_value=True
    ):
        assert hm_config.resolve_default_device().type == "mps"

    with patch("torch.cuda.is_available", return_value=False), patch(
        "hivemind.config.is_mps_available", return_value=False
    ):
        assert hm_config.resolve_default_device().type == "cpu"

    # detect_gpu_backend & detect_gpu_vendor (lines 393-395, 402-408)
    assert hm_config.detect_gpu_backend("mps") == hm_config.GPU_BACKEND_MPS
    assert hm_config.detect_gpu_backend("cpu") == hm_config.GPU_BACKEND_CPU
    with patch("hivemind.config.is_rocm_available", return_value=True):
        assert hm_config.detect_gpu_backend("cuda:0") == hm_config.GPU_BACKEND_ROCM
    assert hm_config.detect_gpu_vendor(hm_config.GPU_BACKEND_CUDA) == hm_config.GPU_VENDOR_NVIDIA
    assert hm_config.detect_gpu_vendor(hm_config.GPU_BACKEND_ROCM) == hm_config.GPU_VENDOR_AMD
    assert hm_config.detect_gpu_vendor(hm_config.GPU_BACKEND_MPS) == hm_config.GPU_VENDOR_APPLE
    assert hm_config.detect_gpu_vendor(hm_config.GPU_BACKEND_CPU) == hm_config.GPU_VENDOR_CPU
    assert hm_config.detect_gpu_vendor("unknown_backend") == hm_config.GPU_VENDOR_UNKNOWN

    # detect_rocm_gcn_arch branches (lines 413-428)
    with patch("hivemind.config.is_rocm_available", return_value=False):
        assert hm_config.detect_rocm_gcn_arch() is None

    with patch("hivemind.config.is_rocm_available", return_value=True), patch(
        "torch.cuda.is_available", return_value=True
    ):
        with patch("torch.cuda.get_device_properties", side_effect=RuntimeError("bad dev")):
            assert hm_config.detect_rocm_gcn_arch() is None

        with patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(gcnArchName="gfx1100", major=11, minor=0),
        ):
            assert hm_config.detect_rocm_gcn_arch() == "gfx1100"

        with patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(gcnArchName="", major=None, minor=0),
        ):
            assert hm_config.detect_rocm_gcn_arch() is None

        with patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(gcnArchName="", major=11, minor=0),
        ):
            assert hm_config.detect_rocm_gcn_arch() == "gfx0b00"

    # rocm_supports_native_bf16 branches (lines 438-451)
    assert hm_config.rocm_supports_native_bf16(None) is False
    assert hm_config.rocm_supports_native_bf16("sm_75") is False
    assert hm_config.rocm_supports_native_bf16("gfx9") is True
    assert hm_config.rocm_supports_native_bf16("gfx1100") is True
    assert hm_config.rocm_supports_native_bf16("gfx1200") is True
    assert hm_config.rocm_supports_native_bf16("gfx1030") is False
    assert hm_config.rocm_supports_native_bf16("gfx803") is False

    # detect_local_hardware: ROCm branch (lines 531-548)
    fake_props = SimpleNamespace(
        name="AMD Radeon RX 7900 XTX",
        major=11,
        minor=0,
        gcnArchName="gfx1100",
        unified_memory=True,
    )
    with patch("torch.cuda.is_available", return_value=True), patch(
        "hivemind.config.is_rocm_available", return_value=True
    ), patch("torch.cuda.get_device_properties", return_value=fake_props), patch(
        "torch.cuda.mem_get_info", return_value=(20 * 1024**3, 24 * 1024**3)
    ):
        hw_rocm = hm_config.detect_local_hardware()
        assert hw_rocm["gpu_backend"] == hm_config.GPU_BACKEND_ROCM
        assert hw_rocm["gpu_vendor"] == hm_config.GPU_VENDOR_AMD
        assert hw_rocm["supports_native_bf16"] is True

    # detect_local_hardware: Apple Silicon MPS branch (lines 549-576)
    with patch("torch.cuda.is_available", return_value=False), patch(
        "hivemind.config.is_mps_available", return_value=True
    ), patch("hivemind.config.is_mps_built", return_value=True), patch(
        "hivemind.config.detect_gpu_backend", return_value=hm_config.GPU_BACKEND_MPS
    ):
        hw_mps = hm_config.detect_local_hardware()
        assert hw_mps["gpu_backend"] == hm_config.GPU_BACKEND_MPS
        assert hw_mps["unified_memory"] is True

    # detect_local_hardware: CPU fallback + nvidia-smi parsing + exception (lines 578-610)
    fake_smi = SimpleNamespace(
        returncode=0,
        stdout="NVIDIA GeForce RTX 3090, 24576, 20480, 8.6\n",
    )
    with patch("torch.cuda.is_available", return_value=False), patch(
        "hivemind.config.is_mps_available", return_value=False
    ), patch("subprocess.run", return_value=fake_smi):
        hw_smi = hm_config.detect_local_hardware()
        assert "Detected via NVML/nvidia-smi" in hw_smi["gpu_name"]
        assert hw_smi["supports_native_bf16"] is True

    with patch("torch.cuda.is_available", return_value=False), patch(
        "hivemind.config.is_mps_available", return_value=False
    ), patch("subprocess.run", side_effect=OSError("no nvidia-smi")):
        hw_cpu = hm_config.detect_local_hardware()
        assert hw_cpu["cuda_available"] is False


# ===========================================================================
# 2. Tests for `src/hivemind/cli/main.py`
# ===========================================================================


def test_cli_main_commands_and_subcommands() -> None:
    import hivemind.cli.main as cli_main

    # 1. cmd_hardware_check
    assert cli_main.main(["hardware-check"]) == 0

    # 2. cmd_benchmark_vram with tiny mocked model
    class _TinyLossModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(4, 4, requires_grad=True))

        def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> Any:
            return SimpleNamespace(loss=(self.w ** 2).sum())

    tiny_mod = _TinyLossModel()
    fake_bundle = SimpleNamespace(
        model=tiny_mod,
        is_4bit_nf4=True,
        trainable_params=16,
        total_params=64,
    )
    fake_vram = {
        "allocated_mb": 12.0,
        "reserved_mb": 16.0,
        "peak_allocated_mb": 20.0,
        "free_vram_gb": 7.5,
        "total_vram_gb": 8.0,
    }
    with patch.object(cli_main, "build_local_qwen2_backbone", return_value=(tiny_mod, "sha", True)), patch.object(
        cli_main, "attach_qlora_adapters", return_value=fake_bundle
    ), patch.object(cli_main, "measure_vram_footprint", return_value=fake_vram), patch(
        "hivemind.core.diloco_inner.build_inner_optimizer",
        return_value=torch.optim.SGD(tiny_mod.parameters(), lr=1e-3),
    ), patch("torch.randint", return_value=torch.ones((1, 8), dtype=torch.long)):
        assert cli_main.main(["benchmark-vram", "--profile", "rtx2070_7b_worker"]) == 0

    # 3. cmd_serve_coordinator with mocked uvicorn.run and tiny model
    with patch("uvicorn.run") as mock_uvicorn, patch.object(
        cli_main,
        "build_local_qwen2_backbone",
        side_effect=lambda **kw: _make_tiny_qwen2(torch.device("cpu"), vocab_size=128),
    ):
        assert cli_main.main(["serve-coordinator", "--host", "127.0.0.1", "--port", "18080"]) == 0
        mock_uvicorn.assert_called_once()

    # 4. All remaining subcommands dispatched through `main(argv=...)`
    with patch("hivemind.cli.simulate_e2e.run_end_to_end_simulation", return_value={}) as m_sim:
        assert cli_main.main(["simulate-e2e", "--inner-steps", "2", "--spot-check-interval", "1"]) == 0
        m_sim.assert_called_once()

    with patch("hivemind.cli.poc_rtx2070.run_huggingface_rtx2070_poc", return_value={}) as m_poc:
        assert cli_main.main(["run-poc", "--subset-size", "6", "--lora-rank", "4", "--inner-steps", "2"]) == 0
        m_poc.assert_called_once()

    with patch("hivemind.cli.demo_hf_domain_swarm.run_decentralized_hf_domain_demo", return_value={}) as m_swarm:
        assert cli_main.main(["run-seti-swarm", "--lora-rank", "4", "--inner-steps", "2"]) == 0
        m_swarm.assert_called_once()

    with patch("hivemind.cli.demo_hf_domain_swarm.run_three_hf_swarm_experiments", return_value={}) as m_exp3:
        assert cli_main.main(["run-3-experiments", "--lora-rank", "4", "--inner-steps", "2"]) == 0
        m_exp3.assert_called_once()

    with patch("hivemind.cli.demo_hf_domain_swarm.run_astrophysics_hf_swarm_experiments", return_value={}) as m_astro:
        assert cli_main.main(["run-astro-experiments", "--lora-rank", "4", "--inner-steps", "2"]) == 0
        m_astro.assert_called_once()

    with patch("hivemind.game.hf_hack_miner.run_hf_hack_mining_game", return_value={}) as m_game:
        assert cli_main.main(["play-hack-game", "--season", "1", "--lora-rank", "4", "--inner-steps", "2"]) == 0
        m_game.assert_called_once()

    with patch("uvicorn.run") as m_uv_hud, patch(
        "hivemind.game.steam_edition.create_hack_miner_fastapi_app", return_value="fake_app"
    ):
        assert cli_main.main(["serve-hack-hud", "--host", "127.0.0.1", "--port", "18090"]) == 0
        m_uv_hud.assert_called_once()

    with patch(
        "hivemind.game.hf_hack_miner.execute_interactive_hf_shard_scan", return_value={"status": "ok"}
    ), patch(
        "hivemind.game.hf_hack_miner.execute_interactive_bounty_probe", return_value={"bounty": "ok"}
    ):
        assert cli_main.main(["hack-probe", "--gamer", "TestGamer"]) == 0

    with patch(
        "hivemind.game.rtx_steam_tool_deployer.run_intense_rtx_stress_suite", return_value={"passed": True}
    ), patch(
        "hivemind.game.rtx_steam_tool_deployer.deploy_free_steam_rtx_tool_v1", return_value={"app_id": 3133700}
    ):
        assert cli_main.main(["deploy-steam-tool", "--vram-cycles", "1", "--udp-packets", "4", "--hotplug-experts", "2"]) == 0


# ===========================================================================
# 3. Tests for `src/hivemind/cli/simulate_e2e.py`
# ===========================================================================


def test_simulate_e2e_pipeline_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import hivemind.cli.simulate_e2e as sim_e2e

    monkeypatch.setattr(sim_e2e, "ARTIFACTS_ROOT", tmp_path)

    # Use 2 domains and rank=4 so the full Phase A -> B -> C -> Phase 2 pipeline completes in <0.5s
    real_cfg = hm_config.HiveMindConfig.from_yaml()
    real_cfg.expert_domains = ["Code", "Mathematics"]
    real_cfg.router_top_k = 2
    for prof in real_cfg.profiles.values():
        prof.lora_rank_r = 4
        prof.lora_alpha = 8

    orig_train_router = sim_e2e.train_moe_router_gating
    orig_spec_inf = sim_e2e.RegionalDHTSwarmRing.run_speculative_inference

    with patch.object(sim_e2e.HiveMindConfig, "from_yaml", return_value=real_cfg), patch.object(
        sim_e2e,
        "build_local_qwen2_backbone",
        side_effect=lambda **kw: _make_tiny_qwen2(torch.device("cpu"), vocab_size=2048),
    ), patch.object(
        sim_e2e,
        "train_moe_router_gating",
        side_effect=lambda moe_model, domain_token_batches, steps=2, lr=5e-3: orig_train_router(
            moe_model=moe_model, domain_token_batches=domain_token_batches, steps=2, lr=lr
        ),
    ), patch.object(
        sim_e2e.RegionalDHTSwarmRing,
        "run_speculative_inference",
        autospec=True,
        side_effect=lambda self, moe_target_model, draft_model, prompt_ids, max_new_tokens=2, draft_gamma=1, simulated_network_rtt_ms=16.0: orig_spec_inf(
            self,
            moe_target_model=moe_target_model,
            draft_model=draft_model,
            prompt_ids=prompt_ids[:4],
            max_new_tokens=2,
            draft_gamma=1,
            simulated_network_rtt_ms=simulated_network_rtt_ms,
        ),
    ), patch("torch.cuda.is_available", return_value=False):
        summary = sim_e2e.run_end_to_end_simulation(
            profile_name="rtx2070_moe_e2e",
            use_hf_hub=False,
            inner_steps_H=2,
            spot_check_interval_s=1,
        )
        assert "phase_ab_branches" in summary
        assert "Code" in summary["phase_ab_branches"]
        assert summary["spot_check_tamper_test"]["passed"] is False
        assert (tmp_path / "e2e_hivemind_report.json").exists()

        # Also cover `use_hf_hub=True` branch (lines 80-82)
        tiny_base, _, _ = _make_tiny_qwen2(torch.device("cpu"), vocab_size=256)
        tiny_bundle = sim_e2e.attach_qlora_adapters(
            base_model=tiny_base, rank_r=4, lora_alpha=8, gradient_checkpointing=False, seed=42
        )
        with patch.object(sim_e2e, "load_hf_qlora_model", return_value=tiny_bundle):
            summary_hf = sim_e2e.run_end_to_end_simulation(
                profile_name="rtx2070_moe_e2e",
                use_hf_hub=True,
                inner_steps_H=2,
                spot_check_interval_s=1,
            )
            assert summary_hf["phase_c_moe_upcycling"]["num_experts"] == 2


# ===========================================================================
# 4. Tests for `src/hivemind/cli/poc_rtx2070.py`
# ===========================================================================


def test_poc_rtx2070_fetch_subsets_and_full_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hivemind.cli.poc_rtx2070 as poc

    monkeypatch.setattr(poc, "ARTIFACTS_ROOT", tmp_path)

    # 1. Test `fetch_hf_domain_subset` for PubMedQA parquet branch
    pubmed_pq = tmp_path / "pubmed.parquet"
    pq.write_table(
        pa.table(
            {
                "question": ["Does ATP power cells?", "Is DNA double-stranded?"],
                "long_answer": ["Yes, mitochondria synthesize ATP.", "Yes, Watson-Crick helix."],
                "final_decision": ["yes", "yes"],
            }
        ),
        pubmed_pq,
    )
    with patch("huggingface_hub.hf_hub_download", return_value=str(pubmed_pq)):
        rows_pm = poc.fetch_hf_domain_subset("qiaojin/PubMedQA", num_samples=2, hf_token="tok")
        assert len(rows_pm) == 2
        assert "Clinical Question:" in rows_pm[0]

    # 2. Test `fetch_hf_domain_subset` for CodeAlpaca JSON branch (with trailing partial brace)
    alpaca_json = tmp_path / "code_alpaca.json"
    alpaca_json.write_text(
        json.dumps(
            [
                {"instruction": "Write add", "input": "a, b", "output": "return a + b"},
                {"instruction": "Write sub", "input": "", "output": "return a - b"},
            ]
        )
        + "{bad_trailing_json",
        encoding="utf-8",
    )
    with patch("huggingface_hub.hf_hub_download", return_value=str(alpaca_json)):
        rows_ca = poc.fetch_hf_domain_subset("sahil2801/CodeAlpaca-20k", num_samples=4, hf_token="tok")
        assert len(rows_ca) == 2
        assert "### Input:" in rows_ca[0]

    # 3. Test `fetch_hf_domain_subset` fallback exception branch
    with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("offline")):
        rows_fb = poc.fetch_hf_domain_subset("sahil2801/CodeAlpaca-20k", num_samples=3)
        assert len(rows_fb) == 3

    # 4. Run full `run_huggingface_rtx2070_poc` with tiny mocked model & tokenizer
    orig_train_router = poc.train_moe_router_gating
    orig_spec_inf = poc.RegionalDHTSwarmRing.run_speculative_inference
    with patch.object(poc.AutoTokenizer, "from_pretrained", return_value=DummyTokenizer()), patch.object(
        poc,
        "stream_load_hf_causal_lm",
        side_effect=lambda **kw: (_make_tiny_qwen2(torch.device("cpu"), vocab_size=256)[0], True, 1.25),
    ), patch.object(
        poc, "fetch_hf_domain_subset", return_value=list(poc.POC_DOMAIN_CORPORA["Code"]) * 3
    ), patch.object(
        poc,
        "train_moe_router_gating",
        side_effect=lambda moe_model, domain_token_batches, steps=2, lr=3e-3: orig_train_router(
            moe_model=moe_model, domain_token_batches=domain_token_batches, steps=2, lr=lr
        ),
    ), patch.object(
        poc.RegionalDHTSwarmRing,
        "run_speculative_inference",
        autospec=True,
        side_effect=lambda self, moe_target_model, draft_model, prompt_ids, max_new_tokens=2, draft_gamma=1, simulated_network_rtt_ms=16.0: orig_spec_inf(
            self,
            moe_target_model=moe_target_model,
            draft_model=draft_model,
            prompt_ids=prompt_ids,
            max_new_tokens=2,
            draft_gamma=1,
            simulated_network_rtt_ms=simulated_network_rtt_ms,
        ),
    ), patch("torch.cuda.is_available", return_value=False):
        payload = poc.run_huggingface_rtx2070_poc(
            model_id="Qwen/Qwen2.5-0.5B",
            hf_dataset_id="sahil2801/CodeAlpaca-20k",
            hf_subset_size=6,
            lora_rank_r=4,
            lora_alpha=8,
            inner_steps_H=2,
            spot_check_interval_s=1,
        )
        assert payload["poc_model_id"] == "Qwen/Qwen2.5-0.5B"
        assert "Code" in payload["phase_b_diloco_branches"]
        assert (tmp_path / "poc_qwen25_rtx2070_report.json").exists()


# ===========================================================================
# 5. Tests for `src/hivemind/cli/demo_hf_domain_swarm.py`
# ===========================================================================


def test_demo_hf_domain_swarm_fetch_branches_and_runners(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hivemind.cli.demo_hf_domain_swarm as swarm

    monkeypatch.setattr(swarm, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(swarm, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(swarm, "PROJECT_ROOT", tmp_path / "proj")
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "proj").mkdir(parents=True, exist_ok=True)

    # 1. Season 2 & Season 3 curated bank branches (rust, cyber, fr, s3)
    assert len(swarm.fetch_hf_domain_subset("anon/rust-code-instructions", num_samples=4)) == 4
    # Second call hits the JSON subset cache (lines 86-90)
    assert len(swarm.fetch_hf_domain_subset("anon/rust-code-instructions", num_samples=4)) == 4

    # Corrupt cache file to cover lines 91-92 (`except Exception: pass`)
    corrupt_cache = (
        tmp_path / "cache" / "swarm_store" / "hf_subsets" / "cybernative_code_vulnerability_security_dpo_4.json"
    )
    corrupt_cache.parent.mkdir(parents=True, exist_ok=True)
    corrupt_cache.write_text("{corrupt_json", encoding="utf-8")
    assert len(swarm.fetch_hf_domain_subset("CyberNative/Code_Vulnerability_Security_DPO", num_samples=4)) == 4

    assert len(swarm.fetch_hf_domain_subset("CohereForAI/aya_collection", num_samples=4)) == 4
    assert len(swarm.fetch_hf_domain_subset("hivemind/quic-zk-spotcheck", num_samples=4)) == 4

    # 2. PubMedQA Parquet branch (lines 142-168)
    pubmed_pq = tmp_path / "pubmed_s.parquet"
    pq.write_table(
        pa.table(
            {
                "question": ["Q1", "Q2", "Q3"],
                "long_answer": ["A1", "A2", "A3"],
                "final_decision": ["yes", "no", "maybe"],
            }
        ),
        pubmed_pq,
    )
    with patch("huggingface_hub.hf_hub_download", return_value=str(pubmed_pq)):
        res_pm = swarm.fetch_hf_domain_subset("qiaojin/PubMedQA", num_samples=2)
        assert len(res_pm) == 2
        assert "Clinical Question:" in res_pm[0]

    # 3. GSM8K Parquet branch (lines 170-192)
    gsm_pq = tmp_path / "gsm8k_s.parquet"
    pq.write_table(
        pa.table(
            {
                "question": ["Math Q1", "Math Q2"],
                "answer": ["Proof 1", "Proof 2"],
            }
        ),
        gsm_pq,
    )
    with patch("huggingface_hub.hf_hub_download", return_value=str(gsm_pq)):
        res_gsm = swarm.fetch_hf_domain_subset("openai/gsm8k", num_samples=2)
        assert len(res_gsm) == 2
        assert "Quantitative Problem:" in res_gsm[0]

    # 4. CodeAlpaca JSON branch (lines 194-223)
    alpaca_file = tmp_path / "alpaca_s.json"
    alpaca_file.write_text(
        json.dumps(
            [
                {"instruction": "Inst 1", "input": "Inp 1", "output": "Out 1"},
                {"instruction": "Inst 2", "input": "", "output": "Out 2"},
            ]
        )
        + "{bad",
        encoding="utf-8",
    )
    with patch("huggingface_hub.hf_hub_download", return_value=str(alpaca_file)):
        res_ca = swarm.fetch_hf_domain_subset("sahil2801/CodeAlpaca-20k", num_samples=2)
        assert len(res_ca) == 2

    # 5. Datasets Server REST API fallback branch (lines 228-247)
    fake_http_payload = json.dumps(
        {"rows": [{"row": {"instruction": "HTTP Inst", "output": "HTTP Out"}}]}
    ).encode("utf-8")
    fake_resp = MagicMock()
    fake_resp.__enter__.return_value = io.BytesIO(fake_http_payload)
    fake_resp.__exit__.return_value = False
    with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("no hub")), patch(
        "urllib.request.urlopen", return_value=fake_resp
    ), patch.object(swarm, "get_hf_token", return_value="hf_tok"):
        res_http = swarm.fetch_hf_domain_subset("custom/http_dataset", num_samples=1)
        assert len(res_http) == 1
        assert "HTTP Inst" in res_http[0]

    # 6. Final static fallback when both hub and HTTP fail (lines 248-254)
    with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("no hub")), patch(
        "urllib.request.urlopen", side_effect=RuntimeError("no http")
    ):
        res_static = swarm.fetch_hf_domain_subset("custom/offline_dataset", num_samples=3)
        assert len(res_static) == 3

    # 7. Run `run_decentralized_hf_domain_demo` & `run_three_hf_swarm_experiments` with tiny mocked model
    orig_train_router = swarm.train_moe_router_gating
    orig_spec_inf = swarm.RegionalDHTSwarmRing.run_speculative_inference
    with patch.object(swarm, "ensure_local_drive_caches", return_value={}), patch.object(
        swarm.AutoTokenizer, "from_pretrained", return_value=DummyTokenizer()
    ), patch.object(
        swarm,
        "stream_load_hf_causal_lm",
        side_effect=lambda **kw: (_make_tiny_qwen2(torch.device("cpu"), vocab_size=256)[0], True, 1.25),
    ), patch.object(
        swarm,
        "fetch_hf_domain_subset",
        return_value=["### Instruction:\nTest\n### Response:\ndef f(): return 1"] * 9,
    ), patch.object(
        swarm,
        "train_moe_router_gating",
        side_effect=lambda moe_model, domain_token_batches, steps=2, lr=5e-3: orig_train_router(
            moe_model=moe_model, domain_token_batches=domain_token_batches, steps=2, lr=lr
        ),
    ), patch.object(
        swarm.RegionalDHTSwarmRing,
        "run_speculative_inference",
        autospec=True,
        side_effect=lambda self, moe_target_model, draft_model, prompt_ids, max_new_tokens=2, draft_gamma=1, simulated_network_rtt_ms=14.0: orig_spec_inf(
            self,
            moe_target_model=moe_target_model,
            draft_model=draft_model,
            prompt_ids=prompt_ids,
            max_new_tokens=2,
            draft_gamma=1,
            simulated_network_rtt_ms=simulated_network_rtt_ms,
        ),
    ), patch("torch.cuda.is_available", return_value=False):
        single_rep = swarm.run_decentralized_hf_domain_demo(
            model_id="Qwen/Qwen2.5-0.5B",
            dataset_id="sahil2801/CodeAlpaca-20k",
            domain_name="Python_Systems_Code",
            lora_rank=4,
            inner_steps=2,
            seq_len=16,
        )
        assert single_rep["domain_expert"] == "Python_Systems_Code"
        assert single_rep["chain_integrity_verified"] is True
        assert single_rep["inference_settlement"]["rtt_speedup"] is None
        assert single_rep["classification_accuracy"] is None

        three_rep = swarm.run_three_hf_swarm_experiments(
            model_id="Qwen/Qwen2.5-0.5B",
            lora_rank=4,
            inner_steps=2,
            seq_len=16,
        )
        assert len(three_rep["experiments"]) == 3
        assert three_rep["dynamic_btm_moe_v2_hotplug"]["final_num_experts"] == 3
        assert three_rep["final_ledger_state"]["chain_integrity_verified"] is True

        astro_rep = swarm.run_astrophysics_hf_swarm_experiments(
            model_id="Qwen/Qwen2.5-0.5B",
            lora_rank=4,
            inner_steps=2,
            seq_len=16,
        )
        assert len(astro_rep["experiments"]) == 4
        assert astro_rep["dynamic_btm_moe_v2_hotplug"]["final_num_experts"] == 4
        assert astro_rep["final_ledger_state"]["chain_integrity_verified"] is True
        assert astro_rep["evidence_scope"]["classification_accuracy_measured"] is False
        for exp in astro_rep["experiments"]:
            assert exp["inference_settlement"]["rtt_speedup"] is None
            assert exp["classification_accuracy"] is None
            assert exp["data_provenance"]["rows"] > 0
            assert all(
                isinstance(w["spot_check_passed"], bool)
                for w in exp["worker_loss_trajectories"].values()
            )
        for gen in astro_rep["regional_swarm_speculative_generations"].values():
            assert gen["latency_reduction_factor"] is None

    # 8. Astrophysics dataset formatters (offline fallback + live Parquet/CSV branches)
    with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("offline")):
        for astro_ds in (
            "mzrk36/exoplanets-unified-dataset",
            "juliensimon/nasa-exoplanets",
            "BrunoBarreto/sdss_dr12_stars_regression",
            "Shoriful025/exoplanet_atmospheric_spectroscopy_results",
        ):
            rows = swarm.fetch_hf_domain_subset(astro_ds, num_samples=3)
            assert len(rows) == 3

    from hivemind.data.hf_real_trainer import train_and_publish_astrophysics_domain_experts
    from hivemind.swarm.dht_storage import ContentAddressedSwarmStore

    astro_store = ContentAddressedSwarmStore(tmp_path / "astro_cas")
    ref_lora = {
        "layers.0.mlp.gate_proj.lora_A": torch.randn(4, 16),
        "layers.0.mlp.gate_proj.lora_B": torch.randn(32, 4),
    }
    reg_astro, summary_astro = train_and_publish_astrophysics_domain_experts(
        store=astro_store, reference_lora_state=ref_lora, inner_steps=2
    )
    assert summary_astro["num_domains_trained"] == 4
    assert len(reg_astro.experts) == 4
