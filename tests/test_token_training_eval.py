"""Real-token smoke tests are deliberately separate from synthetic regression tests."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from hivemind.data.hf_real_trainer import (
    _encode_synthetic_regression_batch, train_single_domain_lora_adapter,
)
from hivemind.data.token_training_eval import (
    FIXTURE_PATH, FIXTURE_SHA256, adapter_state, build_model, evaluate_tokens,
    import_adapter, load_fixture, run_token_training_evaluation, sha256,
    tensor_bytes, token_nll, tokenize_documents, validate_splits,
)
from hivemind.swarm.scientific_swarm_eval import (
    SwarmKnowledgeTransferEngine, _evaluate_lora_on_scientific_holdout,
)
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore


def test_actual_tokenizer_uses_text_while_legacy_simulation_does_not() -> None:
    first = tokenize_documents(["rust", "é"])
    second = tokenize_documents(["ruby", "é"])
    assert first["input_ids"][0].tolist() == [256, 114, 117, 115, 116, 257]
    assert first["input_ids"][1].tolist() == [256, 195, 169, 257, 258, 258]
    assert first["labels"][1, -2:].tolist() == [-100, -100]
    assert not torch.equal(first["input_ids"], second["input_ids"])
    a = _encode_synthetic_regression_batch(["rust"], 8, 8, seed=42)
    b = _encode_synthetic_regression_batch(["not rust at all"], 8, 8, seed=42)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_token_windows_score_each_real_target_once() -> None:
    samples = ["a" * 500, "é" * 77]
    batch = tokenize_documents(samples, sequence_length=16)
    targets = batch["labels"][:, 1:]
    assert int((targets != -100).sum()) == sum(len(s.encode("utf-8")) + 1 for s in samples)
    assert int((targets == 257).sum()) == len(samples)
    assert not bool((targets == 256).any())


@pytest.mark.parametrize("samples,length", [([], 128), ([""], 128), ([" "], 128),
                                             (["ok"], 1), (["ok"], 129)])
def test_tokenizer_rejects_invalid_inputs(samples, length) -> None:
    with pytest.raises(ValueError):
        tokenize_documents(samples, sequence_length=length)


def test_cross_entropy_uses_shift_and_ignores_padding() -> None:
    # True probabilities are 0.75 for target 1, then 0.5 for target 0.
    logits = torch.tensor([[[0.25, 0.75], [0.5, 0.5], [0.99, 0.01], [0.5, 0.5]]]).log()
    labels = torch.tensor([[0, 1, 0, -100]])
    loss, count = token_nll(logits, labels)
    assert count == 2
    assert float(loss) == pytest.approx(-math.log(0.75) - math.log(0.5))
    with pytest.raises(ValueError, match="no next-token"):
        token_nll(logits, torch.full_like(labels, -100))
    with pytest.raises(ValueError, match="non-finite"):
        token_nll(logits * float("nan"), labels)


def test_fixture_has_pinned_separate_documents(tmp_path: Path) -> None:
    fixture, digest = load_fixture()
    assert digest == FIXTURE_SHA256
    assert {r["id"] for r in fixture["train"]}.isdisjoint(r["id"] for r in fixture["holdout"])
    for key in ("id", "text"):
        corrupted = copy.deepcopy(fixture)
        corrupted["holdout"][0][key] = corrupted["train"][0][key]
        with pytest.raises(ValueError, match="duplicate"):
            validate_splits(corrupted)
    for split in ("train", "holdout"):
        corrupted = copy.deepcopy(fixture)
        corrupted[split] = []
        with pytest.raises(ValueError, match="empty"):
            validate_splits(corrupted)
    custom = tmp_path / "custom.json"
    custom.write_text(FIXTURE_PATH.read_text())
    assert load_fixture(custom)[1] == digest


def test_synthetic_mse_never_masquerades_as_token_perplexity() -> None:
    state = {"x.lora_A": torch.zeros(4, 8), "x.lora_B": torch.zeros(8, 4)}
    result = _evaluate_lora_on_scientific_holdout("Rust_Systems_Security", state)
    assert result["loss_metric"] == "mean_squared_error"
    assert result["text_content_used"] is False
    assert result["measurement_kind"] == "synthetic_regression"
    assert result["holdout_perplexity"] is None


def test_synthetic_transfer_without_peers_is_not_success(tmp_path: Path) -> None:
    engine = SwarmKnowledgeTransferEngine(ContentAddressedSwarmStore(tmp_path), ["trainer"])
    state = {"x.lora_A": torch.zeros(4, 8), "x.lora_B": torch.zeros(8, 4)}
    audit = engine.verify_swarm_peer_knowledge_transfer("Rust_Systems_Security", "trainer", state)
    assert audit.all_peers_leveraged_successfully is False
    assert audit.knowledge_transfer_efficiency_ratio is None
    assert audit.peer_measurement_status == "not_run"
    assert audit.mean_peer_pre_transfer_loss is None
    assert audit.mean_peer_post_transfer_loss is None
    assert audit.mean_peer_loss_reduction_pct is None
    assert audit.mean_hotswap_latency_ms is None
    with pytest.raises(ValueError, match="participating"):
        engine.verify_swarm_peer_knowledge_transfer("Rust_Systems_Security", "stranger", state)


def test_import_verifies_bytes_names_shapes_dtypes_and_finite_values() -> None:
    model, _ = build_model(seed=42)
    original = adapter_state(model)
    payload = tensor_bytes(original)
    with pytest.raises(ValueError, match="SHA-256"):
        import_adapter(model, payload[:-1] + bytes([payload[-1] ^ 1]), sha256(payload))
    for mutation, message in (("name", "names"), ("shape", "shape/dtype"),
                              ("dtype", "shape/dtype"), ("nan", "non-finite")):
        modified = {k: v.clone() for k, v in original.items()}
        name = next(iter(modified))
        if mutation == "name":
            modified["not_a_model_parameter"] = modified.pop(name)
        elif mutation == "shape":
            modified[name] = modified[name].flatten()
        elif mutation == "dtype":
            modified[name] = modified[name].double()
        else:
            modified[name].fill_(float("nan"))
        bad = tensor_bytes(modified)
        with pytest.raises(ValueError, match=message):
            import_adapter(model, bad, sha256(bad))
    assert sha256(tensor_bytes(adapter_state(model))) == sha256(payload)


def test_clean_runs_reproduce_real_training_and_local_transfer(tmp_path: Path) -> None:
    first = run_token_training_evaluation(tmp_path / "first", steps=4)
    torch.manual_seed(9999)  # Ambient RNG does not change the pinned experiment.
    second = run_token_training_evaluation(tmp_path / "second", steps=4)
    assert first["measurement_kind"] == "measured_token_training_smoke"
    assert first["pretrained"] is False
    assert first["base_model_sha256"] == second["base_model_sha256"]
    assert first["frozen_parameter_sha256_before"] == first["frozen_parameter_sha256_after"]
    assert first["holdout"] == second["holdout"]
    assert first["transfer"] == second["transfer"]
    transfer = first["transfer"]
    assert transfer["initial_adapter_sha256"] != transfer["adapter_sha256"]
    assert transfer["exact_adapter_match"] and transfer["holdout_matches_within_1e_7"]
    assert transfer["recipient_training_steps"] == 0
    assert transfer["actual_serialized_bytes"] == (tmp_path / "first/adapter.safetensors").stat().st_size
    holdout = first["holdout"]
    assert holdout["baseline"] == holdout["recipient_baseline"]
    assert holdout["trained_adapter"] == holdout["imported_adapter"]
    for name in ("baseline", "trained_adapter", "imported_adapter"):
        metric = holdout[name]
        assert metric["status"] == "measured"
        assert metric["scored_tokens"] == 436
        assert metric["perplexity"] == pytest.approx(math.exp(metric["mean_nll_nats"]))
    assert first["unmeasured"]["parallel_speedup"]["value"] is None
    assert first["unmeasured"]["parallel_speedup"]["status"] == "unavailable"
    assert first["unmeasured"]["gpu_validation"]["status"] == "not_run"
    assert first["unmeasured"]["wan_transfer"]["status"] == "not_run"
    assert json.loads((tmp_path / "first/report.json").read_text()) == first
    with pytest.raises(ValueError, match="empty"):
        run_token_training_evaluation(tmp_path / "first")


def test_zero_steps_honestly_reports_no_improvement(tmp_path: Path) -> None:
    report = run_token_training_evaluation(tmp_path, steps=0)
    assert report["training"]["steps_completed"] == 0
    assert report["training"]["step_loss_nats"] == []
    assert report["holdout"]["improved"] is False
    assert report["holdout"]["nll_improvement_nats"] == 0.0
    assert report["holdout"]["baseline"] == report["holdout"]["trained_adapter"]


@pytest.mark.parametrize("steps,lr", [(-1, 0.01), (1, 0.0), (1, float("nan"))])
def test_invalid_training_arguments(tmp_path: Path, steps: int, lr: float) -> None:
    with pytest.raises(ValueError):
        run_token_training_evaluation(tmp_path, steps=steps, learning_rate=lr)


def test_generated_campaign_ui_does_not_repeat_unsupported_claims() -> None:
    # Check the shipped template, including the offline catch branch.
    from hivemind.game import steam_edition

    template = Path(steam_edition.__file__).read_text()
    assert "Parallel speedup: not measured (sequential simulation)" in template
    assert "mastered 4/4 Domains" not in template
    assert "(3.85x Speedup)" not in template
    assert "(4.0x Leverage)" not in template
    assert "(5.4x RTT)" not in template
    assert "inherited >= 84% Holdout Loss Reduction" not in template


def test_synthetic_trainer_reports_what_it_actually_uses_and_runs() -> None:
    state = {"x.lora_A": torch.zeros(4, 8), "x.lora_B": torch.zeros(8, 4)}
    _, metrics = train_single_domain_lora_adapter("Rust_Systems_Security", state, inner_steps=0)
    assert metrics["measurement_kind"] == "synthetic_regression"
    assert metrics["loss_metric"] == "mean_squared_error"
    assert metrics["text_content_used"] is False
    assert metrics["dataset_id"].startswith("synthetic/")
    assert metrics["requested_inner_steps"] == 0
    assert metrics["inner_steps"] == 2
    assert "not compression-only" in metrics["returned_state_provenance"]
    assert "not serialized wire bytes" in metrics["byte_count_basis"]
