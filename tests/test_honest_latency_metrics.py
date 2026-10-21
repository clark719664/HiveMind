"""No typed-in speed figures are reported as results; perplexity ignores padding."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from hivemind.cli import demo_hf_domain_swarm, poc_rtx2070, simulate_e2e
from hivemind.core.quantization import build_local_qwen2_backbone
from hivemind.game import hf_hack_miner, steam_edition
from hivemind.security import guardrail_attack_simulator
from hivemind.swarm import inference_ring, swarm_fabric
from hivemind.swarm.swarm_fabric import TokenTreeSpeculativeVerifier

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_HUD = REPO_ROOT / "steam_dist" / "HiveMind_Steam_FreeEdition" / "web_ui" / "index.html"


def _src(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def test_no_hardcoded_speedups_remain_in_source() -> None:
    assert "rtt_speedup=5.0" not in _src(hf_hack_miner)
    assert '"rtt_speedup": 5.6' not in _src(hf_hack_miner)
    assert "rtt_speedup=6.0" not in _src(swarm_fabric)
    assert "rtt_speedup=4.0" not in _src(guardrail_attack_simulator)
    assert "round(float(len(best_path)), 2)" not in _src(swarm_fabric)
    assert "x RTT" not in _src(poc_rtx2070) and "x RTT" not in _src(simulate_e2e)
    for text in (_src(steam_edition), SHIPPED_HUD.read_text(encoding="utf-8")):
        assert "latency_reduction_factor || 5.6" not in text
        assert "0.0 ms Exposed Swap Latency" not in text
        assert "simulated RTT" in text and "simulated swap latency" in text


def test_token_tree_report_names_a_token_count_not_a_speedup() -> None:
    report = TokenTreeSpeculativeVerifier(gamma=6).verify_token_tree([[1, 2, 3, 9]], [1, 2, 3, 4, 5])
    assert report["effective_rtt_speedup"] is None
    assert report["rtt_speedup_status"].startswith("not_measured")
    assert report["accepted_tokens_per_verification_pass"] == report["accepted_length"] == 4


def test_speculative_result_carries_its_latency_basis() -> None:
    res = inference_ring.SpeculativeSwarmResult(
        region_ring="r", active_ring_peers=1, mean_ring_ping_ms=1.0, prompt_length=1,
        generated_tokens=[1], draft_gamma=5, swarm_verification_passes=1,
        mean_acceptance_rate=1.0, effective_tokens_per_sec=1.0, latency_reduction_factor=1.0,
    )
    assert "simulated" in res.latency_basis and "Neither is a measured network speedup" in res.latency_basis


def test_batch_perplexity_masks_padding() -> None:
    model, _, _ = build_local_qwen2_backbone(
        seed=7, vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
        device=torch.device("cpu"), use_4bit_nf4_if_cuda=False,
    )
    pad = 0
    ids = torch.randint(1, 128, (2, 12))
    ids[:, 7:] = pad
    masked = demo_hf_domain_swarm._evaluate_batch_perplexity(model, ids, pad_token_id=pad)
    unmasked = demo_hf_domain_swarm._evaluate_batch_perplexity(model, ids)
    assert masked != unmasked

    keep = ids != pad
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=keep.long(), labels=ids.masked_fill(~keep, -100)).loss
    assert math.isclose(masked, round(math.exp(float(ref)), 4), rel_tol=1e-6)
    assert "masked" in demo_hf_domain_swarm.PERPLEXITY_EVAL_ROWS
