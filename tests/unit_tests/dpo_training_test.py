from types import SimpleNamespace
import pytest

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from finetuning_strategies.strategies.dpo import DPO


@pytest.mark.parametrize("explicit_prompt", [True, False])
def test_real_preference_training_saves_model(tmp_path, explicit_prompt):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3, "world": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    config = GPT2Config(vocab_size=5, n_positions=64, n_embd=8, n_layer=1, n_head=1, eos_token_id=2, pad_token_id=1)
    provider = SimpleNamespace(load_model=lambda *args, **kwargs: GPT2LMHeadModel(config), load_tokenizer=lambda *args: fast)
    strategy = DPO()
    strategy.set_config({"output_dir": str(tmp_path), "bf16": False, "fp16": False, "max_length": 16, "gradient_accumulation_steps": 1, "num_train_epochs": 1, "gradient_checkpointing": False, "report_to": "none"})
    rows = {"prompt": ["hello"], "chosen": ["world"], "rejected": ["hello"]} if explicit_prompt else {"chosen": ["hello world"], "rejected": ["hello hello"]}
    trainer = strategy.finetune("tiny", Dataset.from_dict(rows), "causal_language_modeling", provider, "full_finetune", str(tmp_path))
    assert trainer.state.global_step == 1
    assert (tmp_path / "config.json").is_file()
