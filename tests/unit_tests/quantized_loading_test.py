from unittest.mock import Mock

import pytest
import torch
from transformers import BitsAndBytesConfig



@pytest.mark.parametrize('bits', [4, 8])
@pytest.mark.parametrize('legacy', [False, True])
def test_quantization_configuration_is_converted_at_model_loading_boundary(bits, legacy):
    from providers.huggingface import HuggingFaceProvider
    config = {f'load_in_{bits}bit': True}
    kwargs = config if legacy else {'quantization_config': config}
    model_class = Mock()
    HuggingFaceProvider().load_model('org/model', model_class, **kwargs)
    passed = model_class.from_pretrained.call_args.kwargs
    assert 'load_in_4bit' not in passed and 'load_in_8bit' not in passed
    assert isinstance(passed['quantization_config'], BitsAndBytesConfig)
    assert getattr(passed['quantization_config'], f'load_in_{bits}bit') is True


@pytest.mark.parametrize('bits', [4, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real bitsandbytes training requires CUDA')
def test_actual_quantized_qwen2_dpo_step(tmp_path, monkeypatch, caplog, bits):
    from providers.huggingface import HuggingFaceProvider
    from datasets import Dataset
    from peft import LoraConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from finetuning_strategies.strategies.dpo import DPO

    source = tmp_path / 'base'
    backend = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'hello': 3, 'yes': 4, 'no': 5}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    tokenizer.save_pretrained(source)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, eos_token_id=2, pad_token_id=1)).save_pretrained(source)
    strategy = DPO()
    monkeypatch.setattr(strategy, 'launch_tensorboard', lambda *a, **k: None)
    output = tmp_path / 'trained'
    strategy.set_config({'output_dir': str(output), 'bf16': False, 'fp16': False, 'max_length': 32,
        'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'gradient_checkpointing': False,
        'report_to': 'none', 'model_init_kwargs': {f'load_in_{bits}bit': True}})
    trainer = strategy.finetune(str(source), Dataset.from_dict({'prompt': ['hello'], 'chosen': ['yes'], 'rejected': ['no']}),
        'causal_language_modeling', HuggingFaceProvider(), f'qlora_{bits}bit', str(output),
        peft_config=LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    assert trainer.state.global_step == 1
    assert trainer.args.model_init_kwargs is None
    assert strategy.config.model_init_kwargs == {f'load_in_{bits}bit': True}
    assert 'model_init_kwargs` will be ignored' not in caplog.text
    assert getattr(trainer.model, f'is_loaded_in_{bits}bit')
    assert (output / 'adapter_model.safetensors').exists()
    del trainer
    torch.cuda.empty_cache()
