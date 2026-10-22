"""Tests for Next-Generation Outer-Loop DiLoCo Optimizers & Aggregators (`SI-06`..`SI-09`)."""

from __future__ import annotations

import math
import pytest
import torch

from hivemind.swarm.diloco_nextgen import (
    AutoLoCoIntervalCorrector,
    ComputeVisibleSparseLoCoCompressor,
    CosineGatedAdamDecayOuterOptimizer,
    FoMoEPartialReplicaPlanner,
    HeLoCoDirectionCorrector,
    ReputationWeightedIsoCTSMerger,
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
