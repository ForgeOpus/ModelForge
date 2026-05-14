import pytest
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

            def set_config(self, config):
                pass

            def finetune(self, model, data):
                pass

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

            def set_config(self, config):
                pass

            def finetune(self, model, data):
                pass

        with pytest.raises(ValueError, match="Duplicate slug"):
            class SecondStrategy(FinetuningStrategy):
                name = "Second"
                slug = "dup_slug_test_1"
                description = "Second"
                supported_providers = ["huggingface"]

                def set_config(self, config):
                    pass

                def finetune(self, model, data):
                    pass

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
