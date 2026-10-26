"""Tests for the Unity + Steamworks.NET + Google Antigravity Scientific Discovery Game."""

from __future__ import annotations

from fastapi.testclient import TestClient

from hivemind.game.scientific_discovery_game import (
    HiveMindScientificGameEngine,
    build_antigravity_game_copilot_config,
    fold_exoplanet_transit_lightcurve,
    inspect_unity_steam_stack,
    integrate_orbital_symplectic_comparison,
    render_scientific_discovery_game_html,
    run_anse_energy_reactor_mission,
    run_e8_lattice_multikrum_mission,
)
from hivemind.game.steam_edition import create_hack_miner_fastapi_app
from hivemind.router.portal_page import render_portal_page


def test_unity_steamworks_and_solar_system_stack_present() -> None:
    stack = inspect_unity_steam_stack()
    assert stack["all_files_present"] is True
    assert stack["stitch_mcp_design"]["design_system_asset"] == "assets/a3653ca2e0a1411081b38c8c351992f7"
    assert stack["stitch_mcp_design"]["design_system_name"] == "Cyber-Honeycomb Supercluster"
    paths = {f["path"]: f for f in stack["files"]}
    assert paths["Assets/Scripts/SolarSystem/NBodySimulation.cs"]["present"] is True
    assert paths["Assets/Scripts/HiveMind/HiveMindScientificGameBridge.cs"]["present"] is True
    assert paths["Packages/com.rlabrecque.steamworks.net/package.json"]["present"] is True
    assert paths["WebUI/stitch_cyber_honeycomb_dashboard.html"]["present"] is True
    assert paths["WebUI/stitch_e8_honeycomb_animated_svg.html"]["present"] is True


def test_antigravity_copilot_config_and_portal_snippet() -> None:
    cfg = build_antigravity_game_copilot_config()
    assert cfg["sdk_package"] == "google-antigravity"
    assert cfg["config_class_openrouter"] == "LocalOpenAIAgentConfig"
    assert "LocalOpenAIAgentConfig" in cfg["python_sdk_snippet"]

    portal_html = render_portal_page(client_id=None, nonce="testnonce123")
    assert "Google Antigravity SDK" in portal_html
    assert "LocalOpenAIAgentConfig" in portal_html


def test_mission1_symplectic_verlet_and_exoplanet_bls_folding() -> None:
    orbit = integrate_orbital_symplectic_comparison(steps=400, dt=0.015)
    assert orbit["symplectic_gate_passed"] is True
    assert orbit["verlet_relative_drift"] <= 1e-4
    assert orbit["verlet_relative_drift"] < orbit["euler_relative_drift"]
    assert orbit["drift_reduction_factor"] > 10.0

    locked = fold_exoplanet_transit_lightcurve(true_period_days=11.23, trial_period_days=11.23)
    assert locked["transit_locked"] is True
    assert locked["bls_transit_snr"] >= 12.0

    off_period = fold_exoplanet_transit_lightcurve(true_period_days=11.23, trial_period_days=7.19)
    assert off_period["bls_transit_snr"] < locked["bls_transit_snr"]


def test_mission2_e8_lattice_and_multikrum_poisoner_defense() -> None:
    res = run_e8_lattice_multikrum_mission(num_honest=5, num_byzantine=2)
    assert res["all_poisoners_intercepted"] is True
    assert res["mean_e8_sqnr_db"] >= 20.0
    assert res["aggregated_cosine_similarity"] >= 0.99


def test_mission3_anse_energy_reactor_and_steam_preemption() -> None:
    res = run_anse_energy_reactor_mission()
    assert res["d1_within_200_chars"] is True
    assert res["d3_passed_all_tests"] is True
    assert res["selected_best_candidate_id"] == "attempt-3-symplectic-verlet"
    assert res["d6_dpo_pair_harvested"] is True
    assert res["d6_delta_energy"] >= 10.0
    assert res["autopoietic_promotion"]["promoted"] is True
    assert res["preemption_within_1500ms_sla"] is True


def test_full_scientific_odyssey_and_fastapi_endpoints() -> None:
    engine = HiveMindScientificGameEngine()
    odyssey = engine.run_full_scientific_odyssey()
    assert odyssey["all_missions_verified"] is True
    assert odyssey["total_scientific_pts"] == 1200 + 1350 + 1500
    assert "0.00 EUR / 0.00 USD" in odyssey["non_monetary_charter"]

    html = render_scientific_discovery_game_html(odyssey)
    assert "HIVEMIND: DYSON &amp; EXOPLANET ODYSSEY" in html
    assert "SebLague/Solar-System" in html
    assert "Cyber-Honeycomb Supercluster" in html
    assert "assets/a3653ca2e0a1411081b38c8c351992f7" in html
    assert "animated-svg-ANIMATION_1" in html

    client = TestClient(create_hack_miner_fastapi_app())
    r_page = client.get("/scientific-game")
    assert r_page.status_code == 200
    assert "HIVEMIND: DYSON" in r_page.text

    r_status = client.get("/api/scientific-game/status")
    assert r_status.status_code == 200
    assert r_status.json()["all_missions_verified"] is True

    r_solve = client.post(
        "/api/scientific-game/solve",
        json={"mission_id": "MISSION-SCI-02-E8-LATTICE-QUORUM", "gamer_tag": "CyberValkyrie_FR"},
    )
    assert r_solve.status_code == 200
    solve_data = r_solve.json()
    assert solve_data["verified"] is True
    assert solve_data["scientific_pts_awarded"] == 1350

    r_preempt = client.post(
        "/api/scientific-game/preempt",
        json={"gamer_tag": "CyberValkyrie_FR", "game_exe": "HiveMind_Unity_SolarSystem.exe", "foreground_active": True},
    )
    assert r_preempt.status_code == 200
    preempt_data = r_preempt.json()
    assert preempt_data["background_worker_paused"] is True
    assert preempt_data["preemption_within_1500ms_sla"] is True
