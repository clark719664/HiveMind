import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel
from hivemind.data.hf_real_trainer import train_tokenized_domain_adapter

class Tokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids':torch.tensor([[ord(c)%32 for c in text[:kwargs['max_length']]]])}

def model():
    torch.manual_seed(7)
    base = GPT2LMHeadModel(GPT2Config(vocab_size=32,n_positions=32,n_embd=16,n_layer=1,n_head=2))
    return get_peft_model(base, LoraConfig(r=2,lora_alpha=4,target_modules=['c_attn'],task_type='CAUSAL_LM'))

def test_actual_causal_language_loss_trains_lora_on_text():
    net = model().eval()
    state, report = train_tokenized_domain_adapter(net, Tokenizer(), ['actual corpus tokens'], steps=2, max_length=20)
    assert report['training_backend'] == 'causal_lm_lora'
    assert report['steps_completed'] == 2
    assert report['adapter_parameters_changed']
    assert not report['heldout_evaluated']
    assert not net.training
    assert all('lora_' in key for key in state)

def test_reject_non_adapter_trainable_model():
    net = torch.nn.Linear(2,2)
    with pytest.raises(ValueError, match='LoRA'): train_tokenized_domain_adapter(net, Tokenizer(), ['ab'])

@pytest.mark.parametrize('samples', [[], [''], ['x']])
def test_invalid_corpus_cannot_train(samples):
    with pytest.raises(ValueError): train_tokenized_domain_adapter(model(), Tokenizer(), samples)

def test_adapter_reload_and_heldout_evaluation_improves_loss(tmp_path):
    save_path = tmp_path / "test_adapter.pt"
    train_samples = ["rust safe memory buffer allocation", "rust safe memory buffer pointer"] * 4
    heldout_samples = ["rust safe memory buffer allocation", "rust safe memory buffer pointer"]
    net = model().eval()
    state, report = train_tokenized_domain_adapter(
        net,
        Tokenizer(),
        train_samples,
        steps=8,
        max_length=20,
        learning_rate=2e-3,
        heldout_samples=heldout_samples,
        adapter_save_path=save_path,
    )
    assert report['heldout_evaluated'] is True
    assert report['heldout_sample_count'] == 2
    assert report['adapter_saved_path'] == str(save_path)
    assert save_path.exists()
    assert report['adapter_reloaded_and_verified'] is True
    assert report['heldout_final_loss'] < report['heldout_initial_loss']
    assert report['heldout_loss_reduction_pct'] > 0.0
    assert report['heldout_improved'] is True

    # Verify that a fresh model instance loaded with the saved adapter achieves the identical final held-out loss
    fresh = model().eval()
    loaded_state = torch.load(save_path, weights_only=True)
    with torch.no_grad():
        for name, param in fresh.named_parameters():
            if name in loaded_state:
                param.copy_(loaded_state[name])
    tok = Tokenizer()
    h_losses = []
    with torch.no_grad():
        for t in heldout_samples:
            ids = tok(t, max_length=20)['input_ids']
            h_losses.append(float(fresh(input_ids=ids, labels=ids).loss))
    reloaded_loss = sum(h_losses) / len(h_losses)
    assert pytest.approx(reloaded_loss, rel=1e-4) == report['heldout_final_loss']

def test_invalid_heldout_corpus_raises():
    with pytest.raises(ValueError, match="heldout_samples"):
        train_tokenized_domain_adapter(
            model(),
            Tokenizer(),
            ["valid training tokens here"],
            heldout_samples=[""],
        )

