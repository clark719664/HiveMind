# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit & integration tests for the Flagship BYOB Scientific Bee (``hivemind/bee-rusty-sundials``)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.catalog import normalize_model_alias, tier_by_id
from hivemind.router.gateway import create_router_app
from hivemind.swarm.rusty_sundials_scientific_bee import (
    BEE_RUSTY_SUNDIALS_ID,
    BEE_RUSTY_SUNDIALS_PUBLIC_ID,
    BeeRetentionStore,
    analyze_physical_mathematical_problem,
    build_rusty_sundials_rescue_dataset,
    rescue_numeric_calculation,
    rusty_sundials_bee_descriptor,
)


def test_analyze_physical_mathematical_problem_across_four_domains() -> None:
    astro = analyze_physical_mathematical_problem(
        "Kepler post-newtonian binary orbit secular energy drift in explicit RK4"
    )
    assert astro.domain == "astronomy"
    assert "Dual-RRK" in astro.recommended_rusty_sundials_crate or "arkode" in astro.recommended_rusty_sundials_crate

    fluid = analyze_physical_mathematical_problem(
        "3D dyadic Navier-Stokes turbulence enstrophy bound S_N^2 <= 2 Omega_N^3 and Obstruction D4"
    )
    assert fluid.domain == "fluid_mechanics"
    assert "EnstrophyProductionBound" in fluid.lean4_reference_theorem

    qf = analyze_physical_mathematical_problem(
        "Bose-Einstein condensate Projected Gross-Pitaevskii quantized vortex circulation"
    )
    assert qf.domain == "quantum_fluid"
    assert "qf-pgpe" in qf.recommended_rusty_sundials_crate

    gen = analyze_physical_mathematical_problem(
        "Stiff Hopfield-Ninio kinetic proofreading and DNA linking number Lk = Tw + Wr"
    )
    assert gen.domain == "genetics"
    assert "cvode" in gen.recommended_rusty_sundials_crate


def test_rescue_numeric_calculation_all_four_domains() -> None:
    astro_res = rescue_numeric_calculation("Rescue Kepler orbit", domain="astronomy", steps=80, dt=0.02)
    assert astro_res.rescued_succeeded is True
    assert astro_res.naive_failed is True
    assert astro_res.rescued_invariant_drift < 1e-12
    assert astro_res.improvement_factor > 1e9

    fluid_res = rescue_numeric_calculation("Rescue dyadic Navier-Stokes", domain="fluid_mechanics", steps=80, dt=0.01)
    assert fluid_res.rescued_succeeded is True
    assert fluid_res.naive_failed is True
    assert fluid_res.invariants_verified["mechanica_fluidorum_enstrophy_bound_satisfied"] is True
    assert fluid_res.invariants_verified["inviscid_flux_telescoping_residual"] < 1e-14

    qf_res = rescue_numeric_calculation("Rescue PGPE superfluid", domain="quantum_fluid", steps=80, dt=0.01)
    assert qf_res.rescued_succeeded is True
    assert qf_res.invariants_verified["k1_rel_norm_drift"] < 1e-10
    assert qf_res.invariants_verified["k4_plane_wave_max_err"] < 1e-10
    assert qf_res.invariants_verified["vortex_winding_quantization_residual"] < 1e-12

    gen_res = rescue_numeric_calculation("Rescue Hopfield-Ninio and DNA Lk", domain="genetics", steps=80, dt=0.01)
    assert gen_res.rescued_succeeded is True
    assert gen_res.naive_failed is True
    assert gen_res.invariants_verified["strict_positivity_preserved"] is True
    assert gen_res.invariants_verified["max_dna_linking_number_drift"] < 1e-12


def test_build_dataset_and_bee_retention_store(tmp_path: Path) -> None:
    ds_path = tmp_path / "rusty_sundials_dataset.jsonl"
    ds = build_rusty_sundials_rescue_dataset(ds_path)
    assert ds["bee_id"] == BEE_RUSTY_SUNDIALS_ID
    assert ds["num_records"] == 12
    assert ds_path.exists()
    lines = ds_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 12

    store = BeeRetentionStore(base_dir=tmp_path / "retention")
    promoted = store.retain_verified_checkpoint(
        node_id="xavkal-test-node",
        backbone_model="Qwen/Qwen3.5-Coder-7B-Instruct-AWQ",
        lora_rank=32,
        adapter_weights_bytes=b"verified-sundials-weights",
        heldout_energy_e=4.5,
        parent_energy_e=25.0,
    )
    assert promoted["all_four_physics_domains_verified"] is True
    assert promoted["promoted_and_retained"] is True
    assert (tmp_path / "retention" / "latest_promoted.json").exists()

    rejected = store.retain_verified_checkpoint(
        node_id="xavkal-test-node",
        backbone_model="Qwen/Qwen3.5-Coder-7B-Instruct-AWQ",
        lora_rank=32,
        adapter_weights_bytes=b"regressed-sundials-weights",
        heldout_energy_e=30.0,
        parent_energy_e=25.0,
    )
    assert rejected["promoted_and_retained"] is False
    assert rejected["retained_dir"] is None


def test_catalog_and_gateway_endpoints_for_rusty_sundials_bee(tmp_path: Path) -> None:
    assert normalize_model_alias("hivemind/bee-rusty-sundials") == "bee-rusty-sundials"
    assert normalize_model_alias("rusty-sundials/numeric-7b") == "bee-rusty-sundials"
    tier = tier_by_id("bee-rusty-sundials")
    assert tier is not None
    assert tier.min_vram_gb == 6.0

    desc = rusty_sundials_bee_descriptor()
    assert desc["public_id"] == BEE_RUSTY_SUNDIALS_PUBLIC_ID

    store = RouterStore(tmp_path / "router.db", b"k" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    resp_info = client.get("/v1/hivemind/bees/rusty-sundials")
    assert resp_info.status_code == 200
    info_json = resp_info.json()
    assert info_json["bee_id"] == "bee-rusty-sundials"
    assert set(info_json["supported_domains"]) == {
        "astronomy",
        "fluid_mechanics",
        "quantum_fluid",
        "genetics",
    }
    plan = info_json["season20_solver_bee_and_agent_plan"]
    assert plan["season"] == "HM-S20-SOLVER"
    assert len(plan["waves"]["wave_1_solver_bee_v0"]) == 9
    assert "rtx_2070_local_8gb" in plan["owner_agent_fleet"]
    assert "gcp_tesla_t4_16gb_fleet" in plan["owner_agent_fleet"]

    resp_rescue = client.post(
        "/v1/hivemind/bees/rusty-sundials/rescue",
        json={
            "prompt": "Rescue dyadic Navier-Stokes turbulence simulation from explicit blowup",
            "domain": "fluid_mechanics",
            "steps": 40,
            "dt": 0.01,
        },
    )
    assert resp_rescue.status_code == 200
    rescue_json = resp_rescue.json()
    assert rescue_json["domain"] == "fluid_mechanics"
    assert rescue_json["rescued_succeeded"] is True
    assert rescue_json["invariants_verified"]["mechanica_fluidorum_enstrophy_bound_satisfied"] is True
