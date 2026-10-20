import os
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from hivemind.game.hf_hack_miner import (
    ByteTokenizer,
    create_interactive_lora_micro_model,
    execute_interactive_lora_micro_forge,
)
from hivemind.game.steam_edition import create_hack_miner_fastapi_app


def test_user_facing_real_training_executes_and_verifies_heldout():
    """Verify that real_training=True executes actual causal LM LoRA passes,
    saves the adapter, reloads it, and improves held-out results."""
    res = execute_interactive_lora_micro_forge(
        gamer_tag="CyberValkyrie_FR",
        contract_id="CONTRACT-HF-04-RUST-SYSTEMS",
        real_training=True,
        steps=8,
        learning_rate=3e-3,
    )
    assert res["training_executed"] is True
    assert res["runtime_verified"] is True
    assert res["loss_is_simulated"] is False
    assert res["rewards_simulated"] is False
    assert res["source"] == "causal_lm_lora_training"
    assert res["spot_check_passed"] is True

    # Validate empirical held-out loss progression
    assert res["heldout_evaluated"] is True
    assert res["heldout_initial_loss"] > 0.0
    assert res["heldout_final_loss"] > 0.0
    assert res["heldout_final_loss"] < res["heldout_initial_loss"]
    assert res["loss_reduction_pct"] > 0.0
    assert res["heldout_improved"] is True

    # Validate adapter persistence and reloadability
    adapter_path = Path(res["adapter_saved_path"])
    assert adapter_path.exists()
    assert adapter_path.stat().st_size > 0
    assert res["adapter_reloaded_and_verified"] is True
    assert res["steps_completed"] == 8
    assert len(res["training_losses"]) == 8


def test_saved_adapter_reloads_into_new_model_and_reproduces_heldout_loss():
    """Verify that the saved adapter file can be loaded into an independent
    model instance and reproduces the reported held-out loss improvement."""
    res = execute_interactive_lora_micro_forge(
        gamer_tag="NeuralGhost_DE",
        contract_id="CONTRACT-HF-04-RUST-SYSTEMS",
        real_training=True,
        steps=6,
        learning_rate=3e-3,
    )
    saved_adapter_path = Path(res["adapter_saved_path"])
    assert saved_adapter_path.exists()

    # Load adapter weights into a fresh model
    fresh_model = create_interactive_lora_micro_model(seed=42)
    saved_state = torch.load(saved_adapter_path, weights_only=True)
    with torch.no_grad():
        for name, param in fresh_model.named_parameters():
            if name in saved_state:
                param.copy_(saved_state[name])

    # Evaluate on the exact held-out samples with ByteTokenizer
    from hivemind.cli.demo_hf_domain_swarm import fetch_hf_domain_subset
    from hivemind.game.hf_hack_miner import fetch_season4_curated_corpus

    rows = fetch_season4_curated_corpus("anon/rust-code-instructions", num_samples=16)
    if not rows:
        rows = fetch_hf_domain_subset(dataset_id="anon/rust-code-instructions", num_samples=16)
    heldout_texts = rows[-2:]

    fresh_model.eval()
    tokenizer = ByteTokenizer()
    eval_losses = []
    with torch.no_grad():
        for text in heldout_texts:
            enc = tokenizer(text, max_length=128)
            ids = enc["input_ids"]
            out = fresh_model(input_ids=ids, labels=ids)
            eval_losses.append(float(out.loss))

    avg_reloaded_loss = round(sum(eval_losses) / len(eval_losses), 4)
    assert pytest.approx(avg_reloaded_loss, rel=1e-3) == res["heldout_final_loss"]
    assert avg_reloaded_loss < res["heldout_initial_loss"]



def test_fastapi_hud_forge_endpoint_real_training():
    """Verify that the user-facing HUD API endpoint `/api/forge-lora-mission`
    properly accepts real_training=True and returns actual training telemetry."""
    app = create_hack_miner_fastapi_app()
    client = TestClient(app)

    response = client.post(
        "/api/forge-lora-mission",
        json={
            "gamer_tag": "CyberValkyrie_FR",
            "contract_id": "CONTRACT-HF-04-RUST-SYSTEMS",
            "real_training": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["training_executed"] is True
    assert data["loss_is_simulated"] is False
    assert data["heldout_evaluated"] is True
    assert data["heldout_improved"] is True
    assert Path(data["adapter_saved_path"]).exists()


def test_fastapi_hud_forge_endpoint_default_simulation_intact():
    """Verify that the default HUD API endpoint remains 100% backward-compatible
    with existing demo fixture expectations."""
    app = create_hack_miner_fastapi_app()
    client = TestClient(app)

    response = client.post(
        "/api/forge-lora-mission",
        json={
            "gamer_tag": "CyberValkyrie_FR",
            "contract_id": "CONTRACT-HF-04-RUST-SYSTEMS",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "demo_fixture"
    assert data["training_executed"] is False
    assert data["loss_is_simulated"] is True
