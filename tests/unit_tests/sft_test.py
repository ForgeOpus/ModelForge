import pytest
from unittest.mock import patch, MagicMock
from finetuning_strategies.strategies.sft import SFT


def make_dataset(columns):
    """Create a mock dataset with the given column names as features."""
    ds = MagicMock()
    ds.features = {col: None for col in columns}
    return ds


# ---------------------------------------------------------------------------
# validate_data_format
# ---------------------------------------------------------------------------

class TestValidateDataFormat:
    def test_messages_column_valid(self):
        ds = make_dataset(["messages"])
        assert SFT.validate_data_format(ds) is True

    def test_text_column_valid(self):
        ds = make_dataset(["text"])
        assert SFT.validate_data_format(ds) is True

    def test_prompt_and_completion_valid(self):
        ds = make_dataset(["prompt", "completion"])
        assert SFT.validate_data_format(ds) is True

    def test_missing_all_required_columns_raises(self):
        ds = make_dataset(["input"])
        with pytest.raises(ValueError):
            SFT.validate_data_format(ds)

    def test_error_mentions_all_valid_formats(self):
        ds = make_dataset(["irrelevant"])
        with pytest.raises(ValueError) as excinfo:
            SFT.validate_data_format(ds)
        msg = str(excinfo.value)
        assert "messages" in msg
        assert "text" in msg
        assert "prompt" in msg


# ---------------------------------------------------------------------------
# apply_data_format
# ---------------------------------------------------------------------------

class TestApplyDataFormat:
    def test_conversational_returns_messages_key(self):
        result = SFT.apply_data_format("Hello", "Hi there", type_="conversational")
        assert "messages" in result
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_language_modeling_returns_text_key(self):
        result = SFT.apply_data_format("Hello", "ignored", type_="language_modeling")
        assert result == {"text": "Hello"}

    def test_prompt_returns_prompt_completion_keys(self):
        result = SFT.apply_data_format("Hello", "Hi", type_="prompt")
        assert result == {"prompt": "Hello", "completion": "Hi"}

    def test_conversational_prompt_returns_list_values(self):
        result = SFT.apply_data_format("Hello", "Hi", type_="conversational_prompt")
        assert isinstance(result["prompt"], list)
        assert isinstance(result["completion"], list)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            SFT.apply_data_format("Hello", "Hi", type_="instruction")

    def test_non_string_prompt_raises(self):
        with pytest.raises(ValueError, match="string"):
            SFT.apply_data_format(123, "Hi")

    def test_non_string_completion_raises(self):
        with pytest.raises(ValueError, match="string"):
            SFT.apply_data_format("Hello", None)


# ---------------------------------------------------------------------------
# build_sft_config
# ---------------------------------------------------------------------------

_CPU_CONFIG = {"bf16": False, "fp16": False}


class TestBuildSFTConfig:
    def test_build_sft_config_creates_config(self):
        sft = SFT()
        sft.config = None
        sft.build_sft_config(**_CPU_CONFIG)
        assert sft.config is not None

    def test_build_sft_config_default_output_dir(self):
        sft = SFT()
        sft.config = None
        sft.build_sft_config(**_CPU_CONFIG)
        assert sft.config.output_dir == "sft_output"

    def test_build_sft_config_custom_learning_rate(self):
        sft = SFT()
        sft.config = None
        sft.build_sft_config(learning_rate=1e-4, **_CPU_CONFIG)
        assert sft.config.learning_rate == 1e-4

    def test_build_sft_config_custom_epochs(self):
        sft = SFT()
        sft.config = None
        sft.build_sft_config(num_train_epochs=5.0, **_CPU_CONFIG)
        assert sft.config.num_train_epochs == 5.0


# ---------------------------------------------------------------------------
# set_config
# ---------------------------------------------------------------------------

class TestSetConfig:
    def test_set_config_when_none_builds_new(self):
        sft = SFT()
        sft.config = None
        sft.set_config({"output_dir": "my_output", **_CPU_CONFIG})
        assert sft.config is not None
        assert sft.config.output_dir == "my_output"

    def test_set_config_existing_updates_attributes(self):
        sft = SFT()
        sft.config = None
        sft.build_sft_config(**_CPU_CONFIG)
        sft.set_config({"seed": 99})
        assert sft.config.seed == 99


# ---------------------------------------------------------------------------
# finetune (fully mocked)
# ---------------------------------------------------------------------------

class TestFinetune:
    @pytest.fixture
    def mock_task_class(self):
        mock_cls = MagicMock()
        mock_cls.model_class.from_pretrained.return_value = MagicMock()
        return mock_cls

    @pytest.fixture
    def mock_tokenizer(self):
        tok = MagicMock()
        tok.pad_token = "pad"
        tok.pad_token_id = 0
        return tok

    @pytest.fixture
    def valid_dataset(self):
        return make_dataset(["text"])

    def test_finetune_validates_data_format_first(self):
        sft = SFT()
        invalid_ds = make_dataset(["irrelevant"])
        with pytest.raises(ValueError):
            sft.finetune("gpt2", invalid_ds, "causal_language_modeling")

    def test_finetune_calls_trainer_train(self, mock_task_class, mock_tokenizer, valid_dataset):
        sft = SFT()
        sft.build_sft_config(**_CPU_CONFIG)
        mock_trainer_instance = MagicMock()

        with patch("finetuning_strategies.strategies.sft.TaskModels.get_task_class", return_value=mock_task_class):
            with patch("finetuning_strategies.strategies.sft.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                with patch("finetuning_strategies.strategies.sft.SFTTrainer", return_value=mock_trainer_instance) as MockTrainer:
                    sft.finetune("gpt2", valid_dataset, "causal_language_modeling")

        mock_trainer_instance.train.assert_called_once()

    def test_finetune_returns_trainer(self, mock_task_class, mock_tokenizer, valid_dataset):
        sft = SFT()
        sft.build_sft_config(**_CPU_CONFIG)
        mock_trainer_instance = MagicMock()

        with patch("finetuning_strategies.strategies.sft.TaskModels.get_task_class", return_value=mock_task_class):
            with patch("finetuning_strategies.strategies.sft.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                with patch("finetuning_strategies.strategies.sft.SFTTrainer", return_value=mock_trainer_instance):
                    result = sft.finetune("gpt2", valid_dataset, "causal_language_modeling")

        assert result is mock_trainer_instance

    def test_finetune_pads_tokenizer_when_no_pad_token(self, mock_task_class, valid_dataset):
        sft = SFT()
        sft.build_sft_config(**_CPU_CONFIG)
        tok = MagicMock()
        tok.pad_token = None
        tok.eos_token = "<eos>"
        tok.eos_token_id = 1
        mock_trainer_instance = MagicMock()

        with patch("finetuning_strategies.strategies.sft.TaskModels.get_task_class", return_value=mock_task_class):
            with patch("finetuning_strategies.strategies.sft.AutoTokenizer.from_pretrained", return_value=tok):
                with patch("finetuning_strategies.strategies.sft.SFTTrainer", return_value=mock_trainer_instance):
                    sft.finetune("gpt2", valid_dataset, "causal_language_modeling")

        assert tok.pad_token == "<eos>"
        assert tok.pad_token_id == 1

    def test_finetune_uses_existing_config(self, mock_task_class, mock_tokenizer, valid_dataset):
        sft = SFT()
        sft.build_sft_config(output_dir="custom_output", seed=123, **_CPU_CONFIG)
        mock_trainer_instance = MagicMock()

        with patch("finetuning_strategies.strategies.sft.TaskModels.get_task_class", return_value=mock_task_class):
            with patch("finetuning_strategies.strategies.sft.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                with patch("finetuning_strategies.strategies.sft.SFTTrainer", return_value=mock_trainer_instance) as MockTrainer:
                    sft.finetune("gpt2", valid_dataset, "causal_language_modeling")

        # The config passed to SFTTrainer should be the one we built
        call_kwargs = MockTrainer.call_args[1]
        assert call_kwargs["args"].output_dir == "custom_output"
        assert call_kwargs["args"].seed == 123
