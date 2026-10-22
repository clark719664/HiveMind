"""Tests for Next-Generation Outer-Loop DiLoCo Optimizers & Aggregators (`SI-06`..`SI-09`)."""

from __future__ import annotations

import math
import pytest
import torch

from hivemind.swarm.diloco_nextgen import (
    AutoLoCoIntervalCorrector,
    ChunkedOrthonormalDCTTransform,
    ComputeVisibleSparseLoCoCompressor,
    CosineGatedAdamDecayOuterOptimizer,
    CPUOffloadedFlatBucketOuterState,
    FoMoEPartialReplicaPlanner,
    HeLoCoDirectionCorrector,
    LEVERAGED_GITHUB_REPOSITORIES,
    MuonNewtonSchulzLoRAPreconditioner,
    ReputationWeightedIsoCTSMerger,
    SkipAttentionSpeculativeRouter,
    TSVDecorrelatedMerger,
)


def test_si06_cgad_zero_staleness_and_cosine_cutoff_scale_insurance() -> None:
    """Verify `SI-06` (`arXiv:2605.09126`):
    - At `tau = 0`, `gamma = 1.0` and `sigma = 1.0`.
    - At `tau >= tau_cut`, `sigma = 0.0` and stale updates do not pollute Adam's `m` or `v` buffers.
    - Across all `tau >= 0`, `tau * sigma(tau) <= 1 / (e * alpha)`.
    """
    opt = CosineGatedAdamDecayOuterOptimizer(lr=1e-2, alpha=0.2, tau_cut=16.0)
    gamma0, sigma0 = opt.compute_gate(0.0)
    assert math.isclose(gamma0, 1.0, rel_tol=1e-6)
    assert math.isclose(sigma0, 1.0, rel_tol=1e-6)

    bound = opt.theoretical_staleness_bias_bound()
    for tau in (0.5, 1.0, 2.5, 5.0, 8.0, 12.0, 15.9, 16.0, 24.0):
        _, sig = opt.compute_gate(tau)
        assert tau * sig <= bound + 1e-6

    params = {
        "expert_0.lora_A": torch.ones(4, 4, dtype=torch.float32),
        "expert_1.lora_A": torch.ones(4, 4, dtype=torch.float32),
    }
    # Step 1: fresh update on expert_0 (tau=0), extreme stale update on expert_1 (tau=20 >= tau_cut=16)
    grads = {
        "expert_0.lora_A": torch.full((4, 4), 0.5, dtype=torch.float32),
        "expert_1.lora_A": torch.full((4, 4), 500.0, dtype=torch.float32),
    }
    delays = {"expert_0.lora_A": 0.0, "expert_1.lora_A": 20.0}
    updated, telemetry = opt.step_fragments(params, grads, worker_delays=delays)

    t_map = {t.fragment_name: t for t in telemetry}
    assert not t_map["expert_0.lora_A"].dropped_by_cutoff
    assert t_map["expert_1.lora_A"].dropped_by_cutoff
    # Stale update past tau_cut must not allocate or pollute v_buffers for expert_1
    assert "expert_1.lora_A" not in opt.v_buffers
    assert torch.allclose(updated["expert_1.lora_A"], params["expert_1.lora_A"])
    assert not torch.allclose(updated["expert_0.lora_A"], params["expert_0.lora_A"])


def test_si06_pa_cgad_tracks_per_fragment_age_under_streaming_diloco() -> None:
    """Verify `PA-CGAD` increments fragment age when a fragment is skipped in Streaming DiLoCo."""
    opt = CosineGatedAdamDecayOuterOptimizer(lr=1e-2, alpha=0.2, tau_cut=16.0)
    params = {
        "frag_a": torch.zeros(2, 2),
        "frag_b": torch.zeros(2, 2),
    }
    grads = {
        "frag_a": torch.ones(2, 2),
        "frag_b": torch.ones(2, 2),
    }
    # Round 1: only sync frag_a
    params, _ = opt.step_fragments(params, grads, synced_fragments=["frag_a"])
    assert opt.fragment_ages["frag_a"] == 0
    assert opt.fragment_ages["frag_b"] == 1

    # Round 2: only sync frag_a again -> frag_b age becomes 2
    params, _ = opt.step_fragments(params, grads, synced_fragments=["frag_a"])
    assert opt.fragment_ages["frag_b"] == 2

    # Round 3: sync frag_b with worker_delay=0 -> effective_tau uses fragment_age=2
    _, tel = opt.step_fragments(params, grads, synced_fragments=["frag_b"])
    assert len(tel) == 1
    assert tel[0].fragment_name == "frag_b"
    assert math.isclose(tel[0].effective_tau, 2.0, rel_tol=1e-6)
    assert opt.fragment_ages["frag_b"] == 0


def test_si07_reputation_weighted_iso_cts_flattens_spectrum_and_rejects_poisoner() -> None:
    """Verify `SI-07` (`arXiv:2607.03011` `IsoLoCo` + `SI-04`) flattens singular spectrum and resists sign-flip poisoners."""
    torch.manual_seed(7)
    base = {"lora_B": torch.zeros(8, 8)}
    # Construct a highly anisotropic true update (rank-1 dominant spike + smaller modes)
    u_true = torch.eye(8)
    s_true = torch.tensor([20.0, 2.0, 1.5, 1.0, 0.8, 0.5, 0.3, 0.2])
    delta_honest = u_true @ torch.diag(s_true)

    w1 = {"lora_B": delta_honest.clone()}
    w2 = {"lora_B": delta_honest * 0.95}
    w_poison = {"lora_B": -10.0 * delta_honest}

    merger = ReputationWeightedIsoCTSMerger(common_space_fraction=0.75, trim_density=1.0, scaling_lambda=1.0)
    merged, report = merger.merge_worker_deltas(
        base_lora_state=base,
        worker_lora_states=[w1, w2, w_poison],
        worker_reputation_weights=[0.95, 0.95, 0.01],
    )
    assert report["mean_condition_number_before"] > 10.0
    assert math.isclose(report["mean_condition_number_after"], 1.0, rel_tol=1e-4)
    # Cosine alignment with the isotropic honest target (diag(mean(s_true))) must be ~1.0 despite -10x poisoner
    iso_honest_target = u_true @ torch.diag(torch.ones_like(s_true) * s_true.mean())
    cos_sim = torch.nn.functional.cosine_similarity(
        merged["lora_B"].reshape(1, -1), iso_honest_target.reshape(1, -1)
    ).item()
    assert cos_sim > 0.99


def test_si08_heloco_direction_corrector_and_autoloco_interval_scaler() -> None:
    """Verify `SI-08` (`HeLoCo` `arXiv:2606.00271` & `AutoLoCo` `arXiv:2609.36662`)."""
    corrector = HeLoCoDirectionCorrector(conflict_damping=0.2)
    m_ref = torch.tensor([1.0, 0.0, 0.0, 0.0])
    stale_grad = torch.tensor([-2.0, 1.5, 0.5, 0.0])

    corrected, stats = corrector.correct_pseudo_gradient(stale_grad, m_ref)
    assert stats["conflict_detected"] == 1.0
    assert stats["cosine_before"] < 0.0
    assert stats["cosine_after"] > stats["cosine_before"]
    # Orthogonal components (indices 1, 2) must be preserved intact
    assert math.isclose(float(corrected[1].item()), 1.5, rel_tol=1e-6)
    assert math.isclose(float(corrected[2].item()), 0.5, rel_tol=1e-6)

    autoloco = AutoLoCoIntervalCorrector(
        reference_inner_steps_h=32,
        reference_inner_lr=1e-3,
        base_outer_lr=0.7,
        base_outer_momentum=0.9,
    )
    # When inner steps double (H=64), outer LR scales down and momentum decays appropriately
    res = autoloco.correct_outer_hyperparams(actual_inner_steps_h=64, actual_inner_lr=1e-3)
    assert math.isclose(res["interval_scale_ratio"], 2.0, rel_tol=1e-6)
    assert res["corrected_outer_lr"] < 0.7
    assert res["corrected_outer_momentum"] < 0.9


def test_si09_fomoe_partial_replica_and_pulse_sparseloco_compressor() -> None:
    """Verify `SI-09` (`FoMoE` `arXiv:2606.19025` & `PULSELoCo`/`SparseLoCo` `arXiv:2602.03839`, `arXiv:2508.15706`)."""
    planner = FoMoEPartialReplicaPlanner(total_experts=128, active_top_k=2)
    sweet_spot = planner.evaluate_overlap_configuration(
        num_workers_m=16, overlap_factor_oe=4, reshuffle_frequency_gamma=0.0
    )
    assert sweet_spot["local_experts_per_worker"] == 32
    assert sweet_spot["ghost_experts_per_worker"] == 96
    assert math.isclose(sweet_spot["expert_vram_fraction"], 0.25, rel_tol=1e-6)
    assert sweet_spot["wan_payload_reduction_x"] > 2.5

    torch.manual_seed(42)
    ref_param = torch.ones(64, 64)
    # Create a delta where 90% is tiny sub-BF16-ulp noise and 10% is strong signal
    delta = torch.randn(64, 64) * 1e-5
    delta[:4, :16] = 0.25

    comp = ComputeVisibleSparseLoCoCompressor(topk_ratio=0.02, n_bins=4)
    recon, stats = comp.compress_and_reconstruct(delta, reference_param=ref_param)
    assert recon.shape == delta.shape
    assert stats["compute_invisible_fraction"] > 0.85
    assert stats["wire_compression_ratio_x"] > 30.0
    # Deterministic check: running a second time produces bitwise identical reconstruction
    recon2, _ = comp.compress_and_reconstruct(delta, reference_param=ref_param)
    assert torch.equal(recon, recon2)


def test_si06b_cpu_offloaded_flat_bucket_outer_state_zero_vram() -> None:
    """Verify `SI-06B` (`prime-diloco` / `OpenDiLoCo` `arXiv:2407.07852`) flat CPU buffer state."""
    params = {
        "expert_0.lora_A": torch.ones(8, 8, dtype=torch.float32),
        "expert_0.lora_B": torch.full((8, 8), 2.0, dtype=torch.float32),
    }
    outer_state = CPUOffloadedFlatBucketOuterState(params)
    mem = outer_state.memory_telemetry()
    assert mem["extra_gpu_vram_bytes"] == 0
    assert mem["is_contiguous"] is True
    assert mem["total_parameters"] == 128

    inner_after = {
        "expert_0.lora_A": torch.full((8, 8), 0.75, dtype=torch.float32),
        "expert_0.lora_B": torch.full((8, 8), 1.50, dtype=torch.float32),
    }
    pgrads = outer_state.compute_pseudo_gradients(inner_after)
    assert torch.allclose(pgrads["expert_0.lora_A"], torch.full((8, 8), 0.25))
    assert torch.allclose(pgrads["expert_0.lora_B"], torch.full((8, 8), 0.50))

    opt = CosineGatedAdamDecayOuterOptimizer(lr=1e-2, alpha=0.2, tau_cut=16.0)
    updated_master, _ = opt.step_fragments(params, pgrads)
    synced_inner = outer_state.apply_outer_update_and_sync_inner(updated_master)
    assert torch.allclose(synced_inner["expert_0.lora_A"], updated_master["expert_0.lora_A"])


def test_si07b_tsv_decorrelated_merger_and_si09c_chunked_dct() -> None:
    """Verify `SI-07B` (`TSVLoCo` `isoloco` `arXiv:2607.03011`) and `SI-09C` (`DeMo`/`SparseLoCo` chunked DCT)."""
    torch.manual_seed(11)
    d1 = torch.randn(16, 16)
    d2 = torch.randn(16, 16)
    merged_tsv, tsv_report = TSVDecorrelatedMerger.merge_2d_deltas(
        [d1, d2], worker_reputation_weights=[0.8, 0.2]
    )
    assert merged_tsv.shape == (16, 16)
    assert tsv_report["subspace_orthogonality_error"] < 1e-4

    dct = ChunkedOrthonormalDCTTransform(chunk_size=8)
    # Full k=64 must reconstruct the input matrix up to floating-point precision
    full_recon, dct_full_stats = dct.encode_and_decode_topk(d1, topk_per_chunk=64)
    assert torch.allclose(full_recon, d1, atol=1e-5)
    assert math.isclose(dct_full_stats["dct_energy_retained_fraction"], 1.0, rel_tol=1e-5)

    # Smooth low-frequency matrix retains >99.9% energy with only topk_per_chunk=4 (6.25% density)
    grid = torch.linspace(0.0, 1.0, 16)
    smooth = torch.outer(torch.cos(grid), torch.cos(grid))
    sparse_recon, dct_sparse_stats = dct.encode_and_decode_topk(smooth, topk_per_chunk=4)
    assert dct_sparse_stats["dct_energy_retained_fraction"] > 0.999
    assert torch.allclose(sparse_recon, smooth, atol=0.05)


def test_si10_skip_attention_speculative_router_and_car_fallback() -> None:
    """Verify `SI-10` (`pmerolla/fomoe` Skip-Attention Speculative Prefetch + Cache-Aware Routing `CAR`)."""
    torch.manual_seed(19)
    router = SkipAttentionSpeculativeRouter(num_experts=16, top_k=2, car_threshold=0.50)
    norm_w = torch.ones(32)
    router_w = torch.randn(16, 32) * 0.25

    post_ffn = torch.randn(8, 32)
    # Post-attention residual is a small perturbation of post-FFN residual (residual stream persistence)
    post_attn = post_ffn + 0.05 * torch.randn(8, 32)

    out = router.route_with_cache_aware_fallback(
        post_ffn_residual=post_ffn,
        post_attn_residual=post_attn,
        next_layer_norm_weight=norm_w,
        next_layer_router_weight=router_w,
        baseline_cached_experts=[0, 1, 2, 3],
    )
    assert out["routed_experts"].shape == (8, 2)
    assert out["speculative_topk_accuracy"] >= 0.75
    assert out["effective_vram_cache_hit_rate"] >= out["speculative_topk_accuracy"]


def test_si11_muon_newton_schulz_lora_preconditioner_and_repo_registry() -> None:
    """Verify `SI-11` (`arXiv:2610.19064` Newton-Schulz LoRA preconditioner) & GitHub citation registry."""
    torch.manual_seed(23)
    # Create an ill-conditioned 2D LoRA momentum matrix
    u = torch.eye(16, 8)
    s = torch.tensor([50.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1])
    v = torch.eye(8, 16)
    ill_grad = u @ torch.diag(s) @ v

    precond = MuonNewtonSchulzLoRAPreconditioner(steps=5)
    orth_grad, stats = precond.orthogonalize_2d(ill_grad)
    assert orth_grad.shape == ill_grad.shape
    assert stats["condition_number_before"] > 100.0
    assert stats["condition_number_after"] < 3.0

    # Verify all 8 cloned open-source reference repositories are attributed in LEVERAGED_GITHUB_REPOSITORIES
    repo_names = {entry["name"] for entry in LEVERAGED_GITHUB_REPOSITORIES}
    assert {
        "SparseLoCo",
        "isoloco",
        "OpenDiLoCo",
        "prime-diloco",
        "DeMo",
        "asyncdiloco",
        "fomoe",
        "hivemind",
    }.issubset(repo_names)


def test_hm_s8_04_07_08_end_to_end_bridges_in_strategy_and_dma_streamer() -> None:
    """Verify `HM-S8-04` (`EF21` + `PULSELoCo`/`SparseLoCo` 2-bit), `HM-S8-07` (`PinnedRAMExpertDMAStreamer`
    Skip-Attention + `CAR`), and `HM-S8-08` (`AdaptiveDiLoCoCadenceController` + `AutoLoCo`/`HeLoCo`).
    """
    from hivemind.swarm.strategy_optimizer import (
        AdaptiveDiLoCoCadenceController,
        EF21DiLoCoErrorFeedbackBuffer,
    )
    from hivemind.swarm.swarm_fabric import PinnedRAMExpertDMAStreamer

    torch.manual_seed(31)
    # 1. HM-S8-04: EF21 + PULSELoCo/SparseLoCo 2-bit statistical centroid compression
    ef21 = EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.05)
    deltas = {"lora_A": torch.randn(32, 32) * 0.1}
    refs = {"lora_A": torch.ones(32, 32)}
    recon, stats = ef21.compress_with_pulse_sparseloco_2bit(deltas, reference_params=refs, outer_step=1)
    assert "lora_A" in recon
    assert stats["conservation_error_l2"] <= 1e-6
    assert stats["wire_compression_ratio_x"] > 10.0

    # 2. HM-S8-07: PinnedRAMExpertDMAStreamer Skip-Attention + Cache-Aware Routing (CAR)
    streamer = PinnedRAMExpertDMAStreamer(device=torch.device("cpu"))
    expert_names = [f"expert_{i}" for i in range(8)]
    for name in expert_names:
        streamer.stage_expert_in_pinned_ram(name, {"w": torch.ones(4, 4)})
    post_ffn = torch.randn(6, 16)
    post_attn = post_ffn + 0.03 * torch.randn(6, 16)
    norm_w = torch.ones(16)
    router_w = torch.randn(8, 16) * 0.25
    car_report = streamer.prefetch_and_route_skip_attention_car(
        post_ffn_residual=post_ffn,
        post_attn_residual=post_attn,
        next_layer_norm_weight=norm_w,
        next_layer_router_weight=router_w,
        expert_names=expert_names,
        layer_idx=2,
        top_k=2,
        car_threshold=0.40,
        max_prefetch_experts=3,
    )
    assert car_report["milestone"] == "HM-S8-07"
    assert len(car_report["prefetched_experts"]) == 3
    assert car_report["effective_vram_cache_hit_rate"] >= car_report["speculative_topk_accuracy"]

    # 3. HM-S8-08: AdaptiveDiLoCoCadenceController + AutoLoCo + HeLoCo
    cadence_ctrl = AdaptiveDiLoCoCadenceController()
    m_ref = torch.tensor([1.0, 2.0, 0.0, 0.0])
    stale_g = torch.tensor([-1.5, -1.0, 0.5, 0.25])
    bridge = cadence_ctrl.recommend_cadence_with_autoloco_and_heloco(
        rtt_ms=110.0,
        bbr_pacing_mbps=25.0,
        packet_loss_rate=0.02,
        turn_relay_active=True,
        actual_inner_lr=1e-3,
        reference_inner_steps_h=16,
        stale_pseudo_grad=stale_g,
        outer_momentum_ref=m_ref,
    )
    assert bridge["milestone"] == "HM-S8-08"
    assert bridge["cadence"]["sla_8pct_satisfied"] is True
    assert bridge["heloco_direction_telemetry"]["conflict_detected"] == 1.0
    assert (
        bridge["heloco_direction_telemetry"]["cosine_after"]
        > bridge["heloco_direction_telemetry"]["cosine_before"]
    )


