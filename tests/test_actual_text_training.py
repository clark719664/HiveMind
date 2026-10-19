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
