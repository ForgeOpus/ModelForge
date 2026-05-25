import pytest
from unittest.mock import MagicMock, patch
from finetuning_strategies.strategies.strategy import FinetuningStrategy
from finetuning_strategies.strategies.sft import SFT


@pytest.fixture(autouse=True)
def clean_strategy_registry():
    original = dict(FinetuningStrategy._registry)
    yield
    FinetuningStrategy._registry.clear()
    FinetuningStrategy._registry.update(original)


class TestFinetuningStrategyRegistration:
    def test_valid_subclass_auto_registers(self):
        class TestStrategy(FinetuningStrategy):
            name = "Test Strategy"
            slug = "test_strategy_unique_1"
            description = "A test strategy"
            supported_providers = ["huggingface"]
            supported_tasks = ["causal_language_modeling"]
            extra_model_copies = 0
            batch_multiplier = 1.0

            def set_config(self, config):
                pass

            def finetune(self, model, data):
                pass

            @classmethod
            def build_recommended_config(cls, peft_tier, hardware, chosen_model):
                return None, None

        assert "test_strategy_unique_1" in FinetuningStrategy._registry
        assert FinetuningStrategy._registry["test_strategy_unique_1"] is TestStrategy

    def test_slug_none_raises(self):
        with pytest.raises(ValueError, match="slug"):
            class BadStrategy(FinetuningStrategy):
                name = "Bad Strategy"
                slug = None
                description = "Missing slug"
                supported_providers = ["huggingface"]

                def set_config(self, config):
                    pass

                def finetune(self, model, data):
                    pass

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            class BadStrategy(FinetuningStrategy):
                name = None
                slug = "bad_strategy_no_name_1"
                description = "Missing name"
                supported_providers = ["huggingface"]

                def set_config(self, config):
                    pass

                def finetune(self, model, data):
                    pass

    def test_name_non_string_raises(self):
        with pytest.raises(ValueError, match="name"):
            class BadStrategy(FinetuningStrategy):
                name = 42
                slug = "bad_strategy_non_str_name_1"
                description = "Non-string name"
                supported_providers = ["huggingface"]

                def set_config(self, config):
                    pass

                def finetune(self, model, data):
                    pass

    def test_missing_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            class BadStrategy(FinetuningStrategy):
                name = "Bad Strategy"
                slug = "bad_strategy_no_desc_1"
                description = None
                supported_providers = ["huggingface"]

                def set_config(self, config):
                    pass

                def finetune(self, model, data):
                    pass

    def test_missing_supported_providers_raises(self):
        with pytest.raises(ValueError, match="supported_providers"):
            class BadStrategy(FinetuningStrategy):
                name = "Bad Strategy"
                slug = "bad_strategy_no_prov_1"
                description = "No providers"
                supported_providers = None

                def set_config(self, config):
                    pass

                def finetune(self, model, data):
                    pass

    def test_supported_providers_not_list_raises(self):
        with pytest.raises(ValueError, match="supported_providers"):
            class BadStrategy(FinetuningStrategy):
                name = "Bad Strategy"
                slug = "bad_strategy_str_prov_1"
                description = "String providers"
                supported_providers = "huggingface"

                def set_config(self, config):
                    pass

                def finetune(self, model, data):
                    pass

    def test_duplicate_slug_raises(self):
        class FirstStrategy(FinetuningStrategy):
            name = "First"
            slug = "dup_slug_test_1"
            description = "First"
            supported_providers = ["huggingface"]
            supported_tasks = ["causal_language_modeling"]
            extra_model_copies = 0
            batch_multiplier = 1.0

            def set_config(self, config):
                pass

            def finetune(self, model, data):
                pass

            @classmethod
            def build_recommended_config(cls, peft_tier, hardware, chosen_model):
                return None, None

        with pytest.raises(ValueError, match="Duplicate slug"):
            class SecondStrategy(FinetuningStrategy):
                name = "Second"
                slug = "dup_slug_test_1"
                description = "Second"
                supported_providers = ["huggingface"]
                supported_tasks = ["causal_language_modeling"]
                extra_model_copies = 0
                batch_multiplier = 1.0

                def set_config(self, config):
                    pass

                def finetune(self, model, data):
                    pass

                @classmethod
                def build_recommended_config(cls, peft_tier, hardware, chosen_model):
                    return None, None

    def test_sft_in_registry_after_import(self):
        assert "sft" in FinetuningStrategy._registry
        assert FinetuningStrategy._registry["sft"] is SFT

    def test_get_name_returns_strategy_name(self):
        sft = SFT()
        assert sft.get_name() == "Supervised Fine-Tuning"

    def test_get_supported_providers_returns_list_of_strings(self):
        sft = SFT()
        providers = sft.get_supported_providers()
        assert isinstance(providers, list)
        assert all(isinstance(p, str) for p in providers)

    def test_device_attribute_is_valid(self):
        assert SFT.device in ("cpu", "cuda", "mps")

    def test_missing_supported_tasks_raises(self):
        with pytest.raises(ValueError, match="supported_tasks"):
            class BadStrategy(FinetuningStrategy):
                name = "Bad Strategy"
                slug = "bad_strategy_no_tasks_1"
                description = "No tasks"
                supported_providers = ["huggingface"]
                supported_tasks = None
                extra_model_copies = 0
                batch_multiplier = 1.0

                def set_config(self, config):
                    pass

                def finetune(self, model, data):
                    pass

                @classmethod
                def build_recommended_config(cls, peft_tier, hardware, chosen_model):
                    return None, None

    def test_empty_supported_tasks_raises(self):
        with pytest.raises(ValueError, match="supported_tasks"):
            class BadStrategy(FinetuningStrategy):
                name = "Bad Strategy"
                slug = "bad_strategy_empty_tasks_1"
                description = "Empty tasks"
                supported_providers = ["huggingface"]
                supported_tasks = []
                extra_model_copies = 0
                batch_multiplier = 1.0

                def set_config(self, config):
                    pass

                def finetune(self, model, data):
                    pass

                @classmethod
                def build_recommended_config(cls, peft_tier, hardware, chosen_model):
                    return None, None

    def test_negative_extra_model_copies_raises(self):
        with pytest.raises(ValueError, match="extra_model_copies"):
            class BadStrategy(FinetuningStrategy):
                name = "Bad Strategy"
                slug = "bad_strategy_neg_copies_1"
                description = "Negative copies"
                supported_providers = ["huggingface"]
                supported_tasks = ["causal_language_modeling"]
                extra_model_copies = -1
                batch_multiplier = 1.0

                def set_config(self, config):
                    pass

                def finetune(self, model, data):
                    pass

                @classmethod
                def build_recommended_config(cls, peft_tier, hardware, chosen_model):
                    return None, None

    def test_zero_batch_multiplier_raises(self):
        with pytest.raises(ValueError, match="batch_multiplier"):
            class BadStrategy(FinetuningStrategy):
                name = "Bad Strategy"
                slug = "bad_strategy_zero_mult_1"
                description = "Zero multiplier"
                supported_providers = ["huggingface"]
                supported_tasks = ["causal_language_modeling"]
                extra_model_copies = 0
                batch_multiplier = 0.0

                def set_config(self, config):
                    pass

                def finetune(self, model, data):
                    pass

                @classmethod
                def build_recommended_config(cls, peft_tier, hardware, chosen_model):
                    return None, None

    def test_sft_has_required_engine_attributes(self):
        assert isinstance(SFT.supported_tasks, list)
        assert len(SFT.supported_tasks) > 0
        assert all(isinstance(t, str) for t in SFT.supported_tasks)
        assert isinstance(SFT.extra_model_copies, int)
        assert SFT.extra_model_copies >= 0
        assert isinstance(SFT.batch_multiplier, float)
        assert SFT.batch_multiplier > 0.0
        assert "causal_language_modeling" in SFT.supported_tasks
        assert "seq2seq_modeling" in SFT.supported_tasks

    def test_launch_tensorboard_starts_when_enabled(self, tmp_path):
        args = MagicMock()
        args.report_to = ["tensorboard"]
        args.logging_dir = str(tmp_path / "tb")
        tensorboard = MagicMock()
        tensorboard.launch.return_value = "http://127.0.0.1:6006/"

        SFT._tensorboard_urls.clear()
        with patch.object(SFT, "_create_tensorboard_program", return_value=tensorboard):
            with patch("finetuning_strategies.strategies.strategy.webbrowser.open") as browser_open:
                result = SFT().launch_tensorboard(args)

        assert result == "http://127.0.0.1:6006/"
        tensorboard.configure.assert_called_once()
        tensorboard.launch.assert_called_once()
        browser_open.assert_called_once_with("http://127.0.0.1:6006/")

    def test_launch_tensorboard_skips_when_not_enabled(self, tmp_path):
        args = MagicMock()
        args.report_to = "none"
        args.logging_dir = str(tmp_path / "tb")

        with patch.object(SFT, "_create_tensorboard_program") as tensorboard:
            result = SFT().launch_tensorboard(args)

        assert result is None
        tensorboard.assert_not_called()
