"""Unit tests for the Mathematical, Social, and Economic International English Launch Model (HM-S19-INTL)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.economy.international_launch_model import (
    SwarmLiveBaseline,
    compute_congestion_price,
    compute_erlang_c_metrics,
    compute_progressive_starter_grant,
    evaluate_international_launch_readiness,
    simulate_international_launch_wave,
)
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app


def test_erlang_c_detects_saturation_under_unthrottled_english_spike() -> None:
    # Current baseline: 8 producers, 380 tok/s -> service rate ~2.2579 req/s
    stable = compute_erlang_c_metrics(active_producers_c=8, aggregate_tok_per_s=380.0, arrival_rate_req_per_s=0.40)
    assert stable.is_stable is True
    assert stable.utilization_rho < 0.25
    assert stable.p95_total_latency_s < 1.5

    # Unthrottled Hacker News / r/LocalLLaMA spike: 4.5 req/s exceeds 2.26 req/s capacity
    saturated = compute_erlang_c_metrics(active_producers_c=8, aggregate_tok_per_s=380.0, arrival_rate_req_per_s=4.5)
    assert saturated.is_stable is False
    assert saturated.utilization_rho > 1.0
    assert saturated.erlang_c_queue_prob == 1.0


def test_progressive_starter_grant_prevents_unbacked_inflation() -> None:
    quote = compute_progressive_starter_grant(epoch_supply_1k_tokens=200.0, epoch_consumer_demand_1k_tokens=400.0)
    assert quote.instant_grant_hmt == 15.0
    assert quote.prosumer_gpu_unlock_bonus_hmt == 85.0
    assert quote.effective_unbacked_inflation_reduction_pct == 85.0
    assert quote.daily_consumer_drip_hmt == 2.5  # 5.0 * (200 / 400)


def test_congestion_pricing_boosts_producer_rewards_during_peak_demand() -> None:
    calm = compute_congestion_price(base_posted_price_per_1k=0.40, utilization_rho=0.45)
    assert calm["surge_multiplier"] == 1.0
    assert calm["clearing_price_per_1k"] == 0.40

    busy = compute_congestion_price(base_posted_price_per_1k=0.40, utilization_rho=0.90)
    assert busy["surge_multiplier"] > 2.5
    assert busy["producer_reward_per_1k"] > calm["producer_reward_per_1k"] * 2.5


def test_staged_international_wave_simulations_prove_controls_prevent_collapse() -> None:
    # Stage 2 (r/LocalLLaMA, 8,000 visitors) after Stage 1 producer seed (37 producers, 2,150 tok/s)
    proj = simulate_international_launch_wave(
        wave_name="r/LocalLLaMA Technical Wave",
        new_visitors=8000,
        signup_rate=0.15,
        direct_producer_fraction_alpha_p=0.04,
        consumer_to_prosumer_conversion_kappa=0.09,
        baseline=SwarmLiveBaseline(
            total_accounts=287,
            producers=37,
            prosumers=52,
            consumers=195,
            admins=3,
            peak_tflops_fp16=1820.0,
            daytime_aggregate_tok_per_s=2150.0,
        ),
    )
    assert proj.actual_conversion_rate_kappa >= proj.critical_conversion_rate_kappa_star
    assert proj.utilization_rho_with_controls < 0.78
    assert proj.utilization_rho_without_controls > proj.utilization_rho_with_controls
    assert proj.is_self_sustaining is True


def test_gateway_international_launch_readiness_endpoint(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "router.db", b"k" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    resp = client.get("/v1/hivemind/international-launch-readiness")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "hivemind.international_launch_readiness"
    assert data["ready_for_unthrottled_mass_english_launch_today"] is False
    assert data["ready_for_staged_producer_first_rollout"] is True
    assert len(data["four_stage_rollout_plan"]) == 4
    assert len(data["wave_simulations"]) == 3
