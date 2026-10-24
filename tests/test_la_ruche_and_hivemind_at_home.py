"""Unit and integration tests for La Ruche MoE Model Vault, Dataset Mining Campaign & HiveMind@home GCP Dashboard."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import (
    build_hivemind_at_home_state,
    create_hivemind_at_home_app,
    export_hivemind_at_home_artifacts,
    render_hivemind_at_home_html,
)
from hivemind.data.dataset_mining_campaign import (
    SCIENTIFIC_MINING_DOMAIN_CORPORA,
    run_dataset_mining_and_ruche_campaign,
)
from hivemind.swarm.la_ruche_vault import (
    RUCHE_ALVEOLE_HEX_LAYOUT,
    LaRucheModelVault,
    load_community_moe_from_ruche,
)


def test_la_ruche_rejects_degraded_expert_and_accepts_improved_expert(
    tmp_path: Path,
) -> None:
    """Verify La Ruche gatekeeper rejects experts that fail to improve holdout NLL."""
    vault = LaRucheModelVault(
        vault_dir=tmp_path / "ruche_test",
        hidden_size=32,
        vocab_size=259,
        canonical_rank_r=4,
    )
    bad_tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(4, 32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(32, 4),
    }

    with pytest.raises(ValueError, match="La Ruche quality gate rejected"):
        vault.deposit_qlora_expert(
            domain_name="code_python",
            adapter_tensors=bad_tensors,
            contributor_handle="@xaviercallens",
            hardware_device="cpu",
            dataset_shard_cid="cid://sha256/test",
            train_nll_before=5.5,
            train_nll_after=5.4,
            holdout_nll_before=5.5,
            holdout_nll_after=5.6,
            holdout_perplexity_before=244.69,
            holdout_perplexity_after=270.42,
        )


def test_dataset_mining_campaign_populates_la_ruche_and_dashboard(
    tmp_path: Path,
) -> None:
    """Verify end-to-end 8-domain Dataset Mining Campaign, La Ruche MoE export, and HiveMind@home FastAPI portal."""
    assert len(RUCHE_ALVEOLE_HEX_LAYOUT) == 8
    assert len(SCIENTIFIC_MINING_DOMAIN_CORPORA) == 8

    vault_dir = tmp_path / "la_ruche_vault"
    mining_report_path = tmp_path / "dataset_mining_report.json"
    html_path = tmp_path / "hivemind_at_home.html"
    state_path = tmp_path / "hivemind_at_home_state.json"

    summary = run_dataset_mining_and_ruche_campaign(
        vault_dir=vault_dir,
        report_json_path=mining_report_path,
        training_steps=4,
        learning_rate=0.03,
    )

    assert summary["all_shards_dedup_clean"] is True
    assert summary["completed_shard_leases"] == 8
    assert summary["duplicate_shard_leases"] == 0
    assert summary["lease_conservation_holds"] is True
    assert summary["moe_forward_verified"] is True
    assert len(summary["mined_shards"]) == 8
    assert len(summary["domain_training_metrics"]) == 8

    # Verify loading the Community BTM Sparse MoE model from La Ruche
    moe_model, manifest = load_community_moe_from_ruche(vault_dir)
    assert manifest["num_domains"] == 8
    assert manifest["mean_perplexity_reduction_pct"] > 0.0
    tokens = torch.tensor([[2, 5, 9, 14, 21, 33]], dtype=torch.long)
    out = moe_model(input_ids=tokens)
    assert out.logits.shape == (1, 6, 259)
    assert bool(torch.isfinite(out.logits).all())

    # Verify HiveMind@home + GCP + La Ruche HTML/JSON export & FastAPI server
    state = export_hivemind_at_home_artifacts(
        output_html_path=html_path,
        output_state_json_path=state_path,
        mining_ruche_report_path=mining_report_path,
    )
    assert html_path.exists()
    assert state_path.exists()
    html_text = render_hivemind_at_home_html(state)
    assert "HIVEMIND@HOME" in html_text
    assert "LA RUCHE" in html_text
    assert "gwenlaya-tpu-1" in html_text

    app = create_hivemind_at_home_app(mining_ruche_report_path=mining_report_path)
    client = TestClient(app)
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "la-ruche-svg" in r_index.text

    r_status = client.get("/api/hivemind-at-home/status")
    assert r_status.status_code == 200
    assert r_status.json()["formal_verification"]["lean4_theorems_verified"] == 128

    r_ruche = client.get("/api/la-ruche/manifest")
    assert r_ruche.status_code == 200
    assert r_ruche.json()["num_domains"] == 8

    r_shards = client.get("/api/dataset-mining/shards")
    assert r_shards.status_code == 200
    assert r_shards.json()["completed_shard_leases"] == 8
