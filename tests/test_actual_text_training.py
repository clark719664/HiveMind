"""Offline CPU tests using a real, tiny causal LM and real PEFT LoRA weights."""

import pickle
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from hivemind.data.hf_real_trainer import (
    load_tokenized_adapter_state,
    split_tokenized_corpus,
    train_tokenized_domain_adapter,
)


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[ord(c) % 32 for c in text[:kwargs["max_length"]]]])}


def model():
    torch.manual_seed(7)
    base = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2))
    return get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"))


def snapshot(net):
    return {name: value.detach().clone() for name, value in net.state_dict().items()}


def assert_unchanged(net, before):
    assert all(torch.equal(value, before[name]) for name, value in net.state_dict().items())


def test_actual_causal_language_loss_trains_lora_on_text():
    net = model().eval()
    before = snapshot(net)
    state, report = train_tokenized_domain_adapter(net, Tokenizer(), ["actual corpus tokens"], steps=2, max_length=20)
    assert report["training_backend"] == "causal_lm_lora"
    assert report["steps_completed"] == 2
    assert report["adapter_parameters_changed"]
    assert not report["heldout_evaluated"]
    assert not report["heldout_disjoint_verified"]
    assert not report["adapter_reloaded_and_verified"]
    assert not net.training
    assert all("lora_" in key for key in state)
    assert all(torch.equal(value, before[name]) for name, value in net.state_dict().items() if "lora_" not in name)


def test_reject_non_adapter_trainable_model():
    with pytest.raises(ValueError, match="LoRA"):
        train_tokenized_domain_adapter(torch.nn.Linear(2, 2), Tokenizer(), ["ab"])


@pytest.mark.parametrize("samples", [[], [""], ["x"], "abc", [123]])
def test_invalid_corpus_cannot_train(samples):
    net = model()
    before = snapshot(net)
    with pytest.raises(ValueError):
        train_tokenized_domain_adapter(net, Tokenizer(), samples)
    assert_unchanged(net, before)


@pytest.mark.parametrize(
    "train, heldout, max_length, message",
    [
        (["same exact text"], ["same exact text"], 20, "exact text"),
        (["rust safe memory buffer allocation"], ["rust safe memory buffer pointer"], 20, "model-visible"),
        (["abc"], ["ABC"], 20, "model-visible"),
        (["complete seen prefix plus suffix"], ["complete seen prefix"], 32, "prefix"),
        (["complete seen prefix"], ["complete seen prefix plus suffix"], 32, "prefix"),
        (["abcde"], ["ABC"], 20, "prefix"),
    ],
)
def test_leakage_fails_before_evaluation_or_optimizer(monkeypatch, train, heldout, max_length, message):
    net = model().eval()
    before = snapshot(net)

    def forbidden(*args, **kwargs):
        pytest.fail("leakage must be rejected before forward evaluation or optimizer construction")

    monkeypatch.setattr(net, "forward", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    with pytest.raises(ValueError, match=message):
        train_tokenized_domain_adapter(net, Tokenizer(), train, heldout_samples=heldout, max_length=max_length)
    assert_unchanged(net, before)
    assert not net.training


def test_masked_token_differences_do_not_evade_leakage_check():
    class PaddedTokenizer:
        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.tensor([[1 if text == "train" else 9, 2, 3]]),
                "attention_mask": torch.tensor([[0, 1, 1]]),
            }

    with pytest.raises(ValueError, match="model-visible"):
        train_tokenized_domain_adapter(model(), PaddedTokenizer(), ["train"], heldout_samples=["heldout"])


@pytest.mark.parametrize("heldout", [[], [""], ["valid holdout", "x"], ["valid holdout", None]])
def test_every_heldout_example_is_validated_without_mutation(heldout):
    net = model().eval()
    before = snapshot(net)
    with pytest.raises(ValueError, match="heldout_samples"):
        train_tokenized_domain_adapter(net, Tokenizer(), ["training tokens"], heldout_samples=heldout)
    assert_unchanged(net, before)
    assert not net.training


@pytest.mark.parametrize(
    "mask",
    [torch.tensor([[1]]), torch.tensor([[0, 0, 0]]), torch.tensor([[1, 0, 0]]),
     torch.tensor([[1, 2, 1]]), torch.tensor([[1.0, float("nan"), 1.0]])],
)
def test_invalid_heldout_attention_is_rejected(mask):
    class BadMaskTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            encoded = super().__call__(text, **kwargs)
            if text == "bad":
                encoded["attention_mask"] = mask
            return encoded

    net = model()
    before = snapshot(net)
    with pytest.raises(ValueError, match="heldout_samples"):
        train_tokenized_domain_adapter(net, BadMaskTokenizer(), ["train"], heldout_samples=["good", "bad"])
    assert_unchanged(net, before)


@pytest.mark.parametrize("bad_loss", [None, torch.tensor(float("nan")), torch.tensor(float("inf")), torch.tensor([1.0, 2.0])])
def test_nonfinite_or_invalid_heldout_loss_cannot_be_skipped(monkeypatch, bad_loss):
    net = model().eval()
    before = snapshot(net)
    real_forward = net.forward
    calls = 0

    def bad_second_example(**kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(loss=bad_loss) if calls == 2 else real_forward(**kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid baseline evaluation must fail before optimizer construction")

    monkeypatch.setattr(net, "forward", bad_second_example)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    with pytest.raises(ValueError, match="heldout_samples\\[1\\]"):
        train_tokenized_domain_adapter(net, Tokenizer(), ["train"], heldout_samples=["holdout one", "holdout two"])
    assert calls == 2
    assert_unchanged(net, before)
    assert not net.training


def test_split_is_deterministic_and_deduplicates_exact_and_effective_tokens():
    samples = ["abc", "abc", "ABC", "def", "ghi", "jkl", "mno"]
    expected = (["abc", "def", "ghi", "jkl"], ["mno"])
    assert split_tokenized_corpus(Tokenizer(), samples, max_length=20) == expected
    assert split_tokenized_corpus(Tokenizer(), samples, max_length=20) == expected
    assert split_tokenized_corpus(Tokenizer(), ["train prefix", "train suffix", "other"], max_length=5) == (["train prefix"], ["other"])


@pytest.mark.parametrize("samples", [[], ["only"], ["same", "same"], ["abc", "ABC"], ["valid", "x"]])
def test_split_rejects_insufficient_or_invalid_corpus(samples):
    with pytest.raises(ValueError):
        split_tokenized_corpus(Tokenizer(), samples, max_length=20)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan"), float("inf")])
def test_split_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="heldout_fraction"):
        split_tokenized_corpus(Tokenizer(), ["abc", "def"], heldout_fraction=fraction)


def test_split_always_preserves_a_distinct_training_example():
    assert split_tokenized_corpus(Tokenizer(), ["abc", "def"], heldout_fraction=0.99) == (["abc"], ["def"])


def test_disjoint_lora_training_and_reload_into_fresh_model(tmp_path):
    save_path = tmp_path / "test_adapter.pt"
    # Shared task structure is legitimate; these inputs differ *within* the
    # truncation window, unlike the old identical-prefix leakage fixture.
    train_samples = ["rust safe memory", "rust safe buffer"] * 4
    heldout_samples = ["rust safe pointer", "rust safe borrow"]
    net = model().eval()
    state, report = train_tokenized_domain_adapter(
        net, Tokenizer(), train_samples, steps=12, max_length=20, learning_rate=2e-3,
        heldout_samples=heldout_samples, adapter_save_path=save_path,
    )
    assert report["heldout_evaluated"] is True
    assert report["heldout_disjoint_verified"] is True
    assert report["training_unique_token_count"] == 2
    assert report["heldout_unique_token_count"] == 2
    assert report["heldout_sample_count"] == 2
    assert report["heldout_samples_sha256"] != report["samples_sha256"]
    assert report["adapter_saved_path"] == str(save_path)
    assert save_path.exists()
    assert report["adapter_reloaded_and_verified"] is True
    assert report["adapter_verification_method"] == "exact_state_dict_round_trip"
    assert report["adapter_fresh_model_reload_verified"] is False
    assert report["heldout_final_loss"] < report["heldout_initial_loss"]
    assert report["heldout_loss_reduction_pct"] > 0.0
    assert report["heldout_improved"] is True

    fresh = model().eval()
    loaded = load_tokenized_adapter_state(fresh, save_path)
    assert set(loaded) == set(state)
    assert all(torch.equal(loaded[name], state[name]) for name in state)
    losses = []
    with torch.no_grad():
        for text in heldout_samples:
            ids = Tokenizer()(text, max_length=20)["input_ids"]
            losses.append(float(fresh(input_ids=ids, labels=ids).loss))
    assert sum(losses) / len(losses) == pytest.approx(report["heldout_final_loss"], abs=1e-7)
    assert not fresh.training


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype", "nonfinite", "not_tensor", "not_dict"])
def test_corrupted_adapter_rejected_before_any_weight_copy(tmp_path, corruption):
    net = model().eval()
    before = snapshot(net)
    state = {name: parameter.detach().clone() + 1 for name, parameter in net.named_parameters() if parameter.requires_grad}
    first = next(iter(state))
    if corruption == "missing":
        del state[first]
    elif corruption == "extra":
        state["unknown.lora_A"] = torch.ones(1)
    elif corruption == "shape":
        state[first] = torch.ones(1)
    elif corruption == "dtype":
        state[first] = state[first].double()
    elif corruption == "nonfinite":
        state[first].fill_(float("nan"))
    elif corruption == "not_tensor":
        state[first] = "invalid"
    elif corruption == "not_dict":
        state = []
    path = tmp_path / "broken.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="checkpoint"):
        load_tokenized_adapter_state(net, path)
    assert_unchanged(net, before)


def test_missing_adapter_file_leaves_model_unchanged(tmp_path):
    net = model()
    before = snapshot(net)
    with pytest.raises(FileNotFoundError):
        load_tokenized_adapter_state(net, tmp_path / "absent.pt")
    assert_unchanged(net, before)


@pytest.mark.parametrize("corruption", ["missing", "changed"])
def test_saved_roundtrip_verifies_all_weights_and_rolls_back(tmp_path, monkeypatch, corruption):
    net = model().eval()
    before = snapshot(net)
    real_save = torch.save
    path = tmp_path / "adapter.pt"
    path.write_bytes(b"previous checkpoint remains untouched")

    def corrupt_save(state, destination, *args, **kwargs):
        state = {name: value.clone() for name, value in state.items()}
        key = next(iter(state))
        if corruption == "missing":
            del state[key]
        else:
            state[key].add_(1)
        real_save(state, destination, *args, **kwargs)

    monkeypatch.setattr(torch, "save", corrupt_save)
    with pytest.raises(ValueError, match="checkpoint"):
        train_tokenized_domain_adapter(net, Tokenizer(), ["train"], steps=2, adapter_save_path=path)
    assert_unchanged(net, before)
    assert not net.training
    assert path.read_bytes() == b"previous checkpoint remains untouched"
    assert list(tmp_path.iterdir()) == [path]


def test_failed_final_evaluation_rolls_back_training(monkeypatch):
    net = model().eval()
    before = snapshot(net)
    real_forward = net.forward
    eval_calls = 0

    def bad_final_evaluation(**kwargs):
        nonlocal eval_calls
        if not net.training:
            eval_calls += 1
            if eval_calls == 2:
                return SimpleNamespace(loss=torch.tensor(float("nan")))
        return real_forward(**kwargs)

    monkeypatch.setattr(net, "forward", bad_final_evaluation)
    with pytest.raises(ValueError, match="heldout_samples"):
        train_tokenized_domain_adapter(net, Tokenizer(), ["train"], steps=2, heldout_samples=["heldout"])
    assert_unchanged(net, before)
    assert not net.training


@pytest.mark.parametrize("bad_encoding", [None, {}, {"input_ids": torch.tensor([[1.0, 2.0]])},
                                           {"input_ids": torch.tensor([[1, -2]])}])
def test_invalid_tokenizer_output_fails_before_training(bad_encoding):
    net = model()
    before = snapshot(net)
    with pytest.raises(ValueError, match="samples"):
        train_tokenized_domain_adapter(net, lambda *args, **kwargs: bad_encoding, ["valid text"])
    assert_unchanged(net, before)


def test_unreadable_checkpoint_leaves_model_unchanged(tmp_path):
    net = model()
    before = snapshot(net)
    path = tmp_path / "unreadable.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises((pickle.UnpicklingError, RuntimeError, EOFError)):
        load_tokenized_adapter_state(net, path)
    assert_unchanged(net, before)


@pytest.mark.parametrize("baseline, final, reduction", [(1.0, 2.0, -100.0), (0.0, 0.0, None)])
def test_holdout_improvement_is_measured_not_assumed(monkeypatch, baseline, final, reduction):
    net = model().eval()
    real_forward = net.forward
    eval_losses = iter([baseline, final])

    def measured_evaluation(**kwargs):
        if not net.training:
            return SimpleNamespace(loss=torch.tensor(next(eval_losses)))
        return real_forward(**kwargs)

    monkeypatch.setattr(net, "forward", measured_evaluation)
    _, report = train_tokenized_domain_adapter(net, Tokenizer(), ["train"], steps=1, heldout_samples=["heldout"])
    assert report["heldout_improved"] is False
    assert report["heldout_initial_loss"] == baseline
    assert report["heldout_final_loss"] == final
    assert report["heldout_loss_reduction_pct"] == reduction


def test_split_retains_longest_prefix_representatives_in_original_order():
    samples = ["ab", "different", "abc", "xy", "abcd", "xyz", "abcdef", "XYZ"]
    # Prefixes and token-equivalent duplicates collapse to maximal sequences.
    assert split_tokenized_corpus(Tokenizer(), samples, max_length=20) == (
        ["different", "xyz"], ["abcdef"]
    )


def test_split_retains_diverging_longest_prefix_branches():
    assert split_tokenized_corpus(Tokenizer(), ["ab", "abc", "abd"], max_length=20) == (["abc"], ["abd"])


@pytest.mark.parametrize("samples", [["ab", "abc", "abcd"], ["AB", "abcdef"], ["abcdef", "abc", "ab"]])
def test_split_rejects_corpus_with_only_one_prefix_family(samples):
    with pytest.raises(ValueError, match="at least two distinct non-prefix"):
        split_tokenized_corpus(Tokenizer(), samples, max_length=20)
