"""Real local training must never imply validator or settlement certification."""

import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

import hivemind.cli.demo_hf_domain_swarm as datasets
import hivemind.game.hf_hack_miner as miner
import hivemind.game.steam_edition as steam
from hivemind.data.hf_real_trainer import (
    load_tokenized_adapter_state,
    split_tokenized_corpus,
)


@pytest.fixture(autouse=True)
def isolated_artifacts(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(miner, "ARTIFACTS_ROOT", artifacts)
    monkeypatch.setattr(steam, "ARTIFACTS_ROOT", artifacts)
    monkeypatch.setattr(steam, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(datasets, "CACHE_ROOT", tmp_path / "cache")
    return artifacts


def assert_uncertified(result):
    assert result["training_executed"] is True
    assert result["runtime_verified"] is False
    assert result["spot_check_passed"] is False
    assert result["spot_check_executed"] is False
    assert result["validation_status"] == "not_run"
    assert result["loss_is_simulated"] is False
    assert result["rewards_simulated"] is True
    assert result["settlement_executed"] is False
    assert result["reward_status"] == "not_awarded"
    for key in ("lora_pts_awarded", "bounty_pts_awarded", "total_pts_awarded", "hive_minted"):
        assert result[key] == 0
    assert result["steam_rich_presence"] is None
    assert result["rich_presence_hmac"] is None


def test_user_facing_real_training_uses_disjoint_data_without_certification(isolated_artifacts):
    result = miner.execute_interactive_lora_micro_forge(real_training=True, steps=8, learning_rate=3e-3)
    assert_uncertified(result)
    assert result["source"] == "causal_lm_lora_training"
    assert result["model_kind"] == "randomly_initialized_tiny_gpt2"
    assert result["pretrained_model_verified"] is False
    assert result["training_scope"] == "local_text_training_smoke_test"
    assert result["heldout_evaluated"] is True
    assert result["heldout_disjoint_verified"] is True
    # The fixture has three unique examples, repeated to return 16 rows.
    assert result["training_sample_count"] == result["training_unique_token_count"] == 2
    assert result["heldout_sample_count"] == result["heldout_unique_token_count"] == 1
    assert result["heldout_final_loss"] < result["heldout_initial_loss"]
    assert result["heldout_improved"] is True
    assert result["training_samples_sha256"] != result["heldout_samples_sha256"]
    path = Path(result["adapter_saved_path"])
    assert path.parent == (isolated_artifacts / "lora_adapters").resolve()
    assert path.is_file() and path.stat().st_size > 0
    assert result["adapter_reloaded_and_verified"] is True
    assert result["adapter_verification_method"] == "exact_state_dict_round_trip"
    assert result["adapter_fresh_model_reload_verified"] is False
    assert result["steps_completed"] == len(result["training_losses"]) == 8


def test_saved_adapter_reloads_into_fresh_model_with_disjoint_holdout():
    result = miner.execute_interactive_lora_micro_forge(real_training=True, steps=6, learning_rate=3e-3)
    fresh = miner.create_interactive_lora_micro_model(seed=42).eval()
    load_tokenized_adapter_state(fresh, result["adapter_saved_path"])
    rows = datasets.fetch_hf_domain_subset(dataset_id="anon/rust-code-instructions", num_samples=16)
    tokenizer = miner.ByteTokenizer()
    train_texts, heldout_texts = split_tokenized_corpus(tokenizer, rows, max_length=128)
    assert set(train_texts).isdisjoint(heldout_texts)
    train_tokens = {tuple(tokenizer(text, max_length=128)["input_ids"][0].tolist()) for text in train_texts}
    heldout_tokens = {tuple(tokenizer(text, max_length=128)["input_ids"][0].tolist()) for text in heldout_texts}
    assert train_tokens.isdisjoint(heldout_tokens)
    with torch.no_grad():
        losses = []
        for text in heldout_texts:
            encoded = tokenizer(text, max_length=128)
            losses.append(float(fresh(**encoded, labels=encoded["input_ids"]).loss))
    assert sum(losses) / len(losses) == pytest.approx(result["heldout_final_loss"], rel=1e-6)


@pytest.mark.parametrize("gamer_tag", ["../../escape", "/tmp/escape", r"C:\\temp\\escape", r"..\\..\\escape", "a/b", "\x00", "x" * 4096])
def test_untrusted_gamer_tag_cannot_control_adapter_path(gamer_tag, isolated_artifacts):
    result = miner.execute_interactive_lora_micro_forge(gamer_tag=gamer_tag, real_training=True, steps=1)
    path = Path(result["adapter_saved_path"])
    assert path.resolve().parent == (isolated_artifacts / "lora_adapters").resolve()
    assert path.name.startswith("adapter-") and path.suffix == ".pt"
    assert path.is_file()


def test_repeated_requests_do_not_overwrite_adapter():
    first = miner.execute_interactive_lora_micro_forge(real_training=True, steps=1)
    content = Path(first["adapter_saved_path"]).read_bytes()
    second = miner.execute_interactive_lora_micro_forge(real_training=True, steps=1)
    assert first["adapter_saved_path"] != second["adapter_saved_path"]
    assert Path(first["adapter_saved_path"]).read_bytes() == content


@pytest.mark.parametrize("rows", [[], ["one repeated example"] * 16, ["a" * 128 + str(i) for i in range(16)], ["x", "valid independent text"]])
def test_insufficient_independent_corpus_fails_before_training(rows, monkeypatch, isolated_artifacts):
    monkeypatch.setattr(miner, "fetch_season4_curated_corpus", lambda *args, **kwargs: rows)
    monkeypatch.setattr(miner, "fetch_hf_domain_subset", lambda *args, **kwargs: rows)
    def forbidden_model():
        pytest.fail("invalid corpus must fail before model creation or training")
    monkeypatch.setattr(miner, "create_interactive_lora_micro_model", forbidden_model)
    with pytest.raises(ValueError):
        miner.execute_interactive_lora_micro_forge(real_training=True)
    assert not list(isolated_artifacts.rglob("*.pt"))


def test_failed_training_removes_reserved_artifact(monkeypatch, isolated_artifacts):
    import hivemind.data.hf_real_trainer as trainer
    def fail_training(**kwargs):
        raise ValueError("training failed")
    monkeypatch.setattr(trainer, "train_tokenized_domain_adapter", fail_training)
    with pytest.raises(ValueError, match="training failed"):
        miner.execute_interactive_lora_micro_forge(real_training=True)
    assert not list(isolated_artifacts.rglob("*.pt"))


def test_training_does_not_modify_leaderboard_or_call_steam(monkeypatch, isolated_artifacts):
    path = isolated_artifacts / "hf_hack_miner_game_report.json"
    content = json.dumps({"leaderboard": [{"gamer_tag": "CyberValkyrie_FR", "total_points": 4, "hive_balance": 5, "bounties_solved": 0}]})
    path.write_text(content)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    def forbidden_bridge():
        pytest.fail("local training must not claim Steam rewards or presence")
    monkeypatch.setattr(miner, "SteamworksSDKBridge", forbidden_bridge)
    result = miner.execute_interactive_lora_micro_forge(real_training=True, steps=2)
    assert_uncertified(result)
    assert path.read_text() == content


def test_fastapi_real_training_and_hud_expose_limitations():
    client = TestClient(steam.create_hack_miner_fastapi_app())
    response = client.post("/api/forge-lora-mission", json={"real_training": True, "gamer_tag": "../api-escape"})
    assert response.status_code == 200
    assert_uncertified(response.json())
    assert response.json()["heldout_disjoint_verified"] is True
    html = client.get("/").text
    assert "Run Local LoRA Smoke Test (Random-Init Model)" in html
    block = html.split("async function runRealLoraTraining()", 1)[1].split("async function runInteractiveBounty()", 1)[0]
    assert "DISJOINT HOLDOUT" in block and "Settlement executed" in block
    assert "awarded to" not in block
    assert "rp-text" not in block


@pytest.mark.parametrize("value", ["false", "true", 0, 1, [], None])
def test_fastapi_rejects_non_boolean_training_flag(value):
    client = TestClient(steam.create_hack_miner_fastapi_app())
    response = client.post("/api/forge-lora-mission", json={"real_training": value})
    assert response.status_code == 422


def test_fastapi_insufficient_corpus_returns_validation_error(monkeypatch):
    monkeypatch.setattr(miner, "fetch_season4_curated_corpus", lambda *args, **kwargs: ["same text"] * 16)
    client = TestClient(steam.create_hack_miner_fastapi_app())
    response = client.post("/api/forge-lora-mission", json={"real_training": True})
    assert response.status_code == 422
    assert "distinct" in response.json()["detail"]


def test_fastapi_default_simulation_stays_labeled():
    client = TestClient(steam.create_hack_miner_fastapi_app())
    response = client.post("/api/forge-lora-mission", json={})
    assert response.status_code == 200
    result = response.json()
    assert result["source"] == "demo_fixture"
    assert result["training_executed"] is False
    assert result["loss_is_simulated"] is True
    assert result["runtime_verified"] is False
    assert result["rewards_simulated"] is True


@pytest.mark.parametrize("text", ["", "x"])
def test_byte_tokenizer_does_not_invent_short_corpus_tokens(text):
    encoded = miner.ByteTokenizer()(text, max_length=128)
    assert encoded["input_ids"][0].tolist() == list(text.encode("utf-8"))


def test_no_measurable_improvement_remains_uncertified_and_unrewarded():
    result = miner.execute_interactive_lora_micro_forge(
        real_training=True, steps=1, learning_rate=1e-12,
    )
    assert result["heldout_final_loss"] == result["heldout_initial_loss"]
    assert result["heldout_improved"] is False
    assert result["loss_reduction_pct"] == 0.0
    assert_uncertified(result)
