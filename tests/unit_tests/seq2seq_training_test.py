from types import SimpleNamespace

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

from finetuning_strategies.strategies.sft import SFT


def test_native_seq2seq_training_saves_model(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3, "world": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    model = T5ForConditionalGeneration(T5Config(vocab_size=5, d_model=8, d_ff=16, num_layers=1, num_decoder_layers=1, num_heads=1, decoder_start_token_id=1, pad_token_id=1, eos_token_id=2))
    provider = SimpleNamespace(load_model=lambda *args, **kwargs: model, load_tokenizer=lambda *args: fast)
    strategy = SFT()
    strategy.set_config({"output_dir": str(tmp_path), "bf16": False, "fp16": False, "max_length": 16, "num_train_epochs": 1, "per_device_train_batch_size": 1, "gradient_checkpointing": False, "report_to": "none"})
    trainer = strategy.finetune("tiny", Dataset.from_dict({"prompt": ["hello world"], "completion": ["hello"]}), "seq2seq_modeling", provider, output_dir=str(tmp_path))
    assert trainer.state.global_step == 1
    assert (tmp_path / "config.json").is_file()
