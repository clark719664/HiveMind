"""A rejected community import must not partially change a live model or ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hivemind.core.btm_upcycle import SparseMoEMLPBlock, UpcycledSparseMoEModel, UpcycledSwiGLUExpert
from hivemind.swarm.hf_community_hub import CommunityAdapterBundleManager, _compute_state_dict_sha256


def _bundle(tmp_path: Path):
    manager = CommunityAdapterBundleManager(store_dir=tmp_path / "store")
    directory = tmp_path / "bundle"
    directory.mkdir()
    state = {"layer0.lora_A": torch.ones(2, 4), "layer0.lora_B": torch.ones(4, 2)}
    entries = [
        {"domain_name": name, "tensor_sha256": _compute_state_dict_sha256(state),
         "author_gamer_tag": "author", "author_contributor_handle": "@author",
         "canonical_rank_r": 2, "lora_alpha": 4}
        for name in ("first", "second")
    ]
    manifest = {"bundle_id": "test-bundle", "hf_repo_id": "test/bundle", "entries": entries}
    manifest["bundle_sha256"] = hashlib.sha256(
        "|".join(f"{e['domain_name']}:{e['tensor_sha256']}" for e in entries).encode()
    ).hexdigest()
    (directory / "hivemind_swarm_manifest.json").write_text(json.dumps(manifest))
    torch.save({name: state for name in ("first", "second")}, directory / "community_experts_canonical_r16.pt")
    return manager, directory, manifest


def _model():
    backbone = torch.nn.Module()
    backbone.config = SimpleNamespace(hidden_size=4, intermediate_size=8)
    experts = [UpcycledSwiGLUExpert(4, 8, torch.ones(8, 4), torch.ones(8, 4), torch.ones(4, 8))
               for _ in range(2)]
    backbone.mlp = SparseMoEMLPBlock(4, experts, ["original-a", "original-b"])
    return UpcycledSparseMoEModel(backbone, ["original-a", "original-b"])


@pytest.mark.parametrize("corruption", ["bundle_hash", "second_tensor_hash", "second_metadata", "duplicate_domain"])
def test_invalid_bundle_leaves_everything_unchanged(tmp_path, corruption):
    manager, directory, manifest = _bundle(tmp_path)
    model = _model()
    if corruption == "bundle_hash":
        manifest["bundle_sha256"] = "0" * 64
    elif corruption == "second_tensor_hash":
        manifest["entries"][1]["tensor_sha256"] = "0" * 64
    elif corruption == "second_metadata":
        del manifest["entries"][1]["author_gamer_tag"]
    else:
        manifest["entries"][1]["domain_name"] = "first"
    (directory / "hivemind_swarm_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        manager.import_community_bundle(directory, "consumer", model)
    assert model.num_experts == 2
    assert len(model.backbone.mlp.experts) == 2
    assert manager.ledger.balances == {}
    assert manager.ledger.reputation_scores == {}
    assert manager.miner_royalty_pts == {}
    assert manager.import_history == []
    assert not manager.ledger.ledger_path.exists()


def test_late_hotplug_failure_rolls_back_model_and_preserves_ledger(tmp_path, monkeypatch):
    manager, directory, _ = _bundle(tmp_path)
    manager.ledger.credit_reward("existing", 7.0)
    original_disk = manager.ledger.ledger_path.read_bytes()
    model = _model()
    original_router = model.backbone.mlp.router.wg
    original_counts = model.backbone.mlp.expert_token_counts
    original_hotplug = model.hotplug_domain_expert

    def fail_on_second(**kwargs):
        original_hotplug(**kwargs)
        if kwargs["domain_name"] == "second":
            raise RuntimeError("second expert failed after insertion")

    monkeypatch.setattr(model, "hotplug_domain_expert", fail_on_second)
    with pytest.raises(RuntimeError, match="second expert"):
        manager.import_community_bundle(directory, "consumer", model)
    assert model.num_experts == 2
    assert model.expert_domains == ["original-a", "original-b"]
    block = model.backbone.mlp
    assert len(block.experts) == block.num_experts == block.router.num_experts == 2
    assert block.expert_domains == model.expert_domains
    assert block.router.wg is original_router
    assert block.expert_token_counts is original_counts
    assert manager.ledger.balances == {"existing": 7.0}
    assert manager.ledger.ledger_path.read_bytes() == original_disk
    assert manager.miner_royalty_pts == {}
    assert manager.import_history == []
    assert not list(manager.ledger.ledger_path.parent.glob(".community-import-*"))


def test_persistence_failure_does_not_change_model_or_ledger(tmp_path, monkeypatch):
    manager, directory, _ = _bundle(tmp_path)
    model = _model()

    def fail_replace(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr("hivemind.swarm.hf_community_hub.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk unavailable"):
        manager.import_community_bundle(directory, "consumer", model)
    assert model.num_experts == len(model.backbone.mlp.experts) == 2
    assert manager.ledger.balances == {}
    assert manager.miner_royalty_pts == {}
    assert manager.import_history == []
    assert not manager.ledger.ledger_path.exists()


def test_valid_bundle_commits_all_rewards_and_experts(tmp_path):
    manager, directory, _ = _bundle(tmp_path)
    model = _model()
    receipt = manager.import_community_bundle(directory, "consumer", model)
    assert receipt["verified_domains"] == ["first", "second"]
    assert receipt["royalties_settled_hive"] == 30.0
    assert model.num_experts == len(model.backbone.mlp.experts) == 4
    assert manager.ledger.get_balance("author") == 30.0
    assert manager.miner_royalty_pts == {"author": 500}
    assert manager.import_history == [receipt]
    assert json.loads(manager.ledger.ledger_path.read_text())["balances"]["author"] == 30.0
