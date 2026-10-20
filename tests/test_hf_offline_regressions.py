"""Runtime offline flags must prevent every model/dataset network fallback."""

import io
import json
import socket
import urllib.request
from types import SimpleNamespace

import huggingface_hub
import huggingface_hub.constants as hub_constants
import huggingface_hub.utils._headers as hub_headers
import httpx
import pytest
import requests
import torch
import transformers
import transformers.utils.hub as transformers_hub
from safetensors.torch import save_file

import hivemind.config as config
import hivemind.cli.demo_hf_domain_swarm as datasets
import hivemind.cli.poc_rtx2070 as poc
import hivemind.core.quantization as quantization
import hivemind.game.hf_hack_miner as miner


OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
MODEL_ID = "hivemind-tests/tiny-offline-qwen2"
COMMIT = "a" * 40


@pytest.fixture(autouse=True)
def isolated_no_network(monkeypatch, tmp_path):
    # Libraries are deliberately already imported while their offline constants
    # remain false. Only the environment changes at the call site.
    for key in OFFLINE_FLAGS:
        monkeypatch.delenv(key, raising=False)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(transformers_hub, "_is_offline_mode", False)
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(transformers_hub, "TRANSFORMERS_CACHE", str(tmp_path / "hub"))
    for module in (datasets, miner):
        monkeypatch.setattr(module, "CACHE_ROOT", tmp_path / "cache")
        monkeypatch.setattr(module, "ensure_local_drive_caches", lambda: {})
    for module in (datasets, poc, miner):
        monkeypatch.setattr(module, "ARTIFACTS_ROOT", tmp_path / "artifacts")

    def forbidden(*args, **kwargs):
        # pytest.fail is a BaseException, so production fallback handlers cannot
        # hide an attempted network request or credential lookup.
        pytest.fail("offline code attempted network access or credential lookup")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setattr(huggingface_hub, "get_token", forbidden)
    monkeypatch.setattr(hub_headers, "get_token", forbidden)
    return tmp_path


@pytest.mark.parametrize("flag", OFFLINE_FLAGS)
@pytest.mark.parametrize("value", ["1", "ON", "yes", " TrUe "])
def test_offline_flags_are_evaluated_dynamically(flag, value, monkeypatch):
    assert config.hf_download_kwargs("explicit-token") == {"local_files_only": False, "token": "explicit-token"}
    monkeypatch.setenv(flag, value)
    assert config.hf_download_kwargs("explicit-token") == {"local_files_only": True, "token": False}
    monkeypatch.setenv("HF_API_TOKEN", "must-not-be-read-or-promoted")
    assert config.get_hf_token() is None
    assert "HF_TOKEN" not in config.os.environ
    monkeypatch.setenv(flag, "0")
    assert config.hf_offline_enabled() is False


@pytest.mark.parametrize("flag", OFFLINE_FLAGS)
@pytest.mark.parametrize("dataset_id", ["sahil2801/CodeAlpaca-20k", "qiaojin/PubMedQA", "openai/gsm8k"])
def test_offline_cold_dataset_cache_never_reaches_rest_or_hub_network(flag, dataset_id, monkeypatch):
    monkeypatch.setenv(flag, "1")
    rows = datasets.fetch_hf_domain_subset(dataset_id, num_samples=3)
    assert len(rows) == 3
    assert all("binary_search" in row for row in rows)


@pytest.mark.parametrize("flag", OFFLINE_FLAGS)
@pytest.mark.parametrize("dataset_id", ["sahil2801/CodeAlpaca-20k", "qiaojin/PubMedQA"])
def test_poc_offline_cold_cache_uses_curated_fallback(flag, dataset_id, monkeypatch):
    monkeypatch.setenv(flag, "1")
    rows = poc.fetch_hf_domain_subset(dataset_id, num_samples=3, hf_token="must-not-be-used")
    assert len(rows) == 3
    assert rows[0] == poc.POC_DOMAIN_CORPORA["Code"][0]


@pytest.mark.parametrize("dataset_id,marker", [
    ("anon/rust-code-instructions", "Rust Systems"),
    ("CyberNative/Code_Vulnerability_Security_DPO", "CyberSecurity Audit"),
    ("CohereForAI/aya_collection", "Question Scientifique"),
    ("hivemind/lean-theorems", "Season 3 Swarm"),
])
def test_offline_curated_banks_remain_available(dataset_id, marker, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    rows = datasets.fetch_hf_domain_subset(dataset_id, num_samples=3)
    assert len(rows) == 3
    assert marker in rows[0]


def test_offline_subset_cache_is_read_before_credentials_or_hub(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cache = tmp_path / "cache/swarm_store/hf_subsets/sahil2801_codealpaca_20k_2.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(["local row one", "local row two"]))

    def forbidden(*args, **kwargs):
        pytest.fail("a populated subset cache must not consult credentials or the Hub")

    monkeypatch.setattr(datasets, "get_hf_token", forbidden)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    assert datasets.fetch_hf_domain_subset(num_samples=2) == ["local row one", "local row two"]


def hub_snapshot(tmp_path, repo_id, repo_type="models"):
    repo_cache = tmp_path / "hub" / (repo_type + "--" + repo_id.replace("/", "--"))
    snapshot = repo_cache / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    refs = repo_cache / "refs"
    refs.mkdir()
    (refs / "main").write_text(COMMIT)
    return snapshot


@pytest.mark.parametrize("module", [datasets, poc])
def test_offline_hub_cached_dataset_survives_partial_subset(module, monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    snapshot = hub_snapshot(tmp_path, "sahil2801/CodeAlpaca-20k", "datasets")
    (snapshot / "code_alpaca_20k.json").write_text(json.dumps([
        {"instruction": "Use cached data", "input": "only", "output": "Local answer"},
    ]))
    rows = module.fetch_hf_domain_subset(num_samples=3)
    assert len(rows) == 1
    assert "Local answer" in rows[0]


@pytest.mark.parametrize("flag", OFFLINE_FLAGS)
def test_offline_cold_model_cache_fails_without_network(flag, monkeypatch):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(OSError):
        quantization.stream_load_hf_causal_lm(
            MODEL_ID, torch.device("cpu"), torch.float32, hf_token="must-not-be-used"
        )


def test_offline_hub_cached_model_loads_without_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    snapshot = hub_snapshot(tmp_path, MODEL_ID)
    cfg = transformers.Qwen2Config(
        vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        tie_word_embeddings=False,
    )
    model = transformers.Qwen2ForCausalLM(cfg)
    cfg.to_json_file(snapshot / "config.json")
    save_file(model.state_dict(), snapshot / "model.safetensors")
    loaded, _, _ = quantization.stream_load_hf_causal_lm(
        MODEL_ID, torch.device("cpu"), torch.float32,
        hf_token="must-not-be-used", apply_nf4=False,
    )
    for name, weight in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], weight)


@pytest.mark.parametrize("module,function_name", [
    (datasets, "run_decentralized_hf_domain_demo"),
    (datasets, "run_three_hf_swarm_experiments"),
    (poc, "run_huggingface_rtx2070_poc"),
    (miner, "run_hf_hack_mining_game"),
])
def test_tokenizer_entry_points_explicitly_use_local_only(module, function_name, monkeypatch):
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module, "detect_local_hardware", lambda: {
        "gpu_name": "test CPU", "vram_free_gb": 0, "vram_total_gb": 0,
    })
    monkeypatch.setattr(module, "fetch_hf_domain_subset", lambda **kwargs: ["local corpus"])
    if module is miner:
        monkeypatch.setattr(miner, "HackMiningGameEngine", lambda: SimpleNamespace(
            verify_all_steam_accounts=lambda: {}, gamers={},
        ))

    original_load = module.AutoTokenizer.from_pretrained

    def check_options(model_id, **kwargs):
        assert model_id == MODEL_ID
        assert kwargs == {"local_files_only": True, "token": False}
        return original_load(model_id, **kwargs)

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", check_options)
    with pytest.raises(OSError):
        getattr(module, function_name)(model_id=MODEL_ID)


def test_online_dataset_fallback_retains_authenticated_rest_path(monkeypatch):
    calls = []
    monkeypatch.setenv("HF_TOKEN", "test-online-token")

    def missing_hub_file(**kwargs):
        calls.append(("hub", kwargs))
        raise OSError("mocked unavailable dataset file")

    def mock_rest(request, timeout):
        calls.append(("rest", request, timeout))
        return io.BytesIO(json.dumps({"rows": [{"row": {
            "instruction": "Online question", "output": "Online answer",
        }}]}).encode())

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing_hub_file)
    monkeypatch.setattr(urllib.request, "urlopen", mock_rest)
    rows = datasets.fetch_hf_domain_subset(num_samples=1)
    assert "Online answer" in rows[0]
    assert calls[0][1]["local_files_only"] is False
    assert calls[0][1]["token"] == "test-online-token"
    assert calls[1][1].get_header("Authorization") == "Bearer test-online-token"
    assert calls[1][1].full_url.startswith("https://datasets-server.huggingface.co/rows?")
    assert calls[1][2] == 15


def test_offline_flag_set_during_hub_failure_blocks_rest_fallback(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-online-token")

    def switch_offline(**kwargs):
        assert kwargs["local_files_only"] is False
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        raise OSError("remote download became unavailable")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", switch_offline)
    rows = datasets.fetch_hf_domain_subset(num_samples=2)
    assert len(rows) == 2 and "binary_search" in rows[0]


def test_online_model_calls_remain_enabled_with_explicit_token(monkeypatch):
    calls = []

    class DownloadReached(Exception):
        pass

    def mock_config(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return SimpleNamespace()

    def mock_download(**kwargs):
        calls.append(kwargs)
        raise DownloadReached

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", mock_config)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", mock_download)
    with pytest.raises(DownloadReached):
        quantization.stream_load_hf_causal_lm(
            MODEL_ID, torch.device("cpu"), torch.float32, hf_token="test-online-token"
        )
    assert calls[0] == (MODEL_ID, {"local_files_only": False, "token": "test-online-token"})
    assert calls[1]["local_files_only"] is False
    assert calls[1]["token"] == "test-online-token"
