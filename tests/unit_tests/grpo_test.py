from datasets import Dataset, IterableDataset
import pytest

from finetuning_strategies.strategies.grpo import GRPO, exact_match_reward
from recommendation_engine.language_use_cases.use_case import LanguageUseCase
from recommendation_engine.validation_models.recommendation import RecommendedModel
from utilities.hardware_detector import HardwareDetector
from utilities.recommendation_engine import RecommendationEngine
from utilities.training_objective import TrainingObjective


def hardware(vram=24.0):
    return HardwareDetector.HardwareSpecs(
        physical_core_count=8,
        disk_storage_gb=500,
        available_disk_storage_gb=200,
        ram_gb=64,
        gpus=[{"name": "GPU", "total_memory_gb": vram}],
    )


def test_exact_match_reward_is_deterministic_and_safe():
    rewards = exact_match_reward(
        completions=[" answer ", [{"role": "assistant", "content": "other"}], {"bad": "shape"}],
        answer=["answer", "answer", "answer"],
        unused_user_value="never evaluated",
    )
    assert rewards == [1.0, 0.0, 0.0]


def test_grpo_dataset_contract_requires_prompt_and_answer():
    GRPO.validate_data_format(Dataset.from_dict({"prompt": ["2 + 2"], "answer": ["4"]}))
    with pytest.raises(ValueError, match="'prompt' and 'answer'"):
        GRPO.validate_data_format(Dataset.from_dict({"prompt": ["2 + 2"]}))
    with pytest.raises(ValueError, match="does not support streaming"):
        GRPO.validate_data_format(IterableDataset.from_generator(lambda: iter([])))


def test_grpo_recommendation_config_has_valid_group_batching():
    config, adapter = GRPO.build_recommended_config(
        "lora_bf16", hardware(), RecommendedModel(model_id="org/model", num_parameters=500_000_000, downloads=1)
    )
    assert config.num_generations >= 2
    assert config.gradient_accumulation_steps % config.num_generations == 0
    assert config.beta == 0.0
    assert adapter is not None and adapter.target_modules == "all-linear"


def test_instruction_following_reinforcement_resolves_grpo():
    use_case = LanguageUseCase.get_use_case_by_slug("instruction_following")
    assert TrainingObjective.REINFORCEMENT in use_case.compatible_objectives
    _, strategy, task_slug, strategy_slug, data_format = RecommendationEngine._resolve_use_case_and_strategy(
        "instruction_following", TrainingObjective.REINFORCEMENT
    )
    assert strategy is GRPO
    assert task_slug == "causal_language_modeling"
    assert strategy_slug == "grpo"
    assert data_format.column_aliases["ideal_answer"] == "answer"


def test_real_grpo_training_saves_model(tmp_path):
    from types import SimpleNamespace
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3, "world": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=5, n_positions=64, n_embd=8, n_layer=1, n_head=1, eos_token_id=2, pad_token_id=1))
    provider = SimpleNamespace(load_model=lambda *args, **kwargs: model, load_tokenizer=lambda *args: fast)
    strategy = GRPO()
    strategy.set_config({"output_dir": str(tmp_path), "bf16": False, "fp16": False, "max_completion_length": 2, "num_generations": 2, "gradient_accumulation_steps": 2, "num_train_epochs": 1, "gradient_checkpointing": False, "report_to": "none", "save_steps": 100})
    trainer = strategy.finetune("tiny", Dataset.from_dict({"prompt": ["hello", "world"], "answer": ["world", "hello"]}), "causal_language_modeling", provider, "full_finetune", str(tmp_path))
    assert trainer.state.global_step >= 1
    assert (tmp_path / "config.json").is_file()
