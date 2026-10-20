"""Fast/offline regressions for the opt-in pretrained acceptance tool.

The end-to-end unit test substitutes a random tiny model; it is never evidence of
pretrained quality. The separate evidence JSON is produced with the pinned model.
"""

import json
from pathlib import Path

import pytest
import torch

from tools.training_acceptance import pretrained_cpu as validation


def test_snapshot_defaults_to_pinned_offline_unauthenticated_assets(tmp_path, monkeypatch):
    import huggingface_hub

    for name in validation.ASSETS:
        (tmp_path / name).write_bytes(b"fixture")
    monkeypatch.setattr(validation, "MODEL_WEIGHTS_SHA256", validation.file_sha256(tmp_path / "model.safetensors"))
    calls = []

    def download(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert validation.snapshot(tmp_path, False) == tmp_path
    assert calls[0][0] == validation.MODEL_ID
    assert calls[0][1]["revision"] == validation.MODEL_REVISION
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["token"] is False
    assert calls[0][1]["allow_patterns"] == validation.ASSETS
    validation.snapshot(tmp_path, True)
    assert calls[1][1]["local_files_only"] is False


def test_incomplete_cache_fails_without_random_model_fallback(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Incomplete pinned model cache"):
        validation.snapshot(tmp_path, False)


def test_modified_pretrained_weights_fail_closed(tmp_path, monkeypatch):
    import huggingface_hub

    for name in validation.ASSETS:
        (tmp_path / name).write_bytes(b"not the official model")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: str(tmp_path))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validation.snapshot(tmp_path, False)


def test_tensor_digest_is_order_independent_and_sensitive_to_weight_changes():
    first = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    assert validation.tensor_digest(first.items()) == validation.tensor_digest(reversed(list(first.items())))
    original = validation.tensor_digest(first.items())
    first["a"] += 1
    assert original != validation.tensor_digest(first.items())


@pytest.mark.parametrize("steps,threads", [(0, 2), (65, 2), (16, 0), (16, 9)])
def test_run_bounds_fail_before_loading_model(tmp_path, monkeypatch, steps, threads):
    monkeypatch.setattr(validation, "snapshot", lambda *a: pytest.fail("must not load a model"))
    with pytest.raises(ValueError, match="bounded validation"):
        validation.run_validation(cache_dir=tmp_path, allow_download=False,
                                  output=tmp_path / "report.json", steps=steps, threads=threads)


def test_failed_run_replaces_old_pass_report_and_returns_failure(tmp_path, monkeypatch, capsys):
    output = tmp_path / "report.json"
    output.write_text('{"status": "PASS"}')

    def missing(**kwargs):
        assert kwargs["allow_download"] is False
        raise FileNotFoundError("No cached pinned model")

    monkeypatch.setattr(validation, "run_validation", missing)
    assert validation.main(["--out", str(output), "--cache-dir", str(tmp_path)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "ERROR"
    assert report["production_quality_verified"] is False
    assert "No cached pinned model" in report["error"]
    assert json.loads(capsys.readouterr().out) == report


def test_validation_checks_fresh_reload_and_replay_without_network(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from hivemind.game.hf_hack_miner import ByteTokenizer, create_interactive_lora_micro_model

    for name in validation.ASSETS:
        (tmp_path / name).write_bytes(b"unit test model fixture")
    monkeypatch.setattr(validation, "snapshot", lambda *a: tmp_path)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: ByteTokenizer())
    monkeypatch.setattr(validation, "build_model", lambda path, seed: create_interactive_lora_micro_model(seed=seed))
    report = validation.run_validation(cache_dir=tmp_path, allow_download=False,
                                       output=tmp_path / "report.json", steps=2)
    assert report["status"] == "PASS"
    assert all(report["checks"].values())
    assert report["training"]["sample_count"] == 9
    assert report["training"]["heldout_sample_count"] == 3
    assert report["fresh_reload_max_abs_logit_difference"] == 0
    assert report["training"]["adapter_fresh_model_reload_verified"] is False
    assert report["checks"]["fresh_model_reload_verified"] is True
    assert report["production_quality_verified"] is False
    assert report["runtime_validator_certified"] is False
    assert report["settlement_executed"] is False
    assert report["rewards_awarded"] == 0
    assert Path(report["training"]["adapter_saved_path"]).is_file()
