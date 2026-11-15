import pytest
from finetuning_strategies.strategy_factory import FinetuningStrategies, StrategyResponse
from finetuning_strategies.strategies.sft import SFT
from finetuning_strategies.strategies.dpo import DPO


class TestFinetuningStrategies:
    def test_get_strategy_class_sft(self):
        result = FinetuningStrategies.get_strategy_class("sft")
        assert result is SFT

    def test_get_strategy_class_dpo(self):
        assert FinetuningStrategies.get_strategy_class("dpo") is DPO

    def test_get_strategy_class_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            FinetuningStrategies.get_strategy_class("unknown")

    def test_list_strategies_non_empty(self):
        strategies = FinetuningStrategies.list_strategies()
        assert len(strategies) >= 1

    def test_list_strategies_contains_sft(self):
        strategies = FinetuningStrategies.list_strategies()
        slugs = [s.slug for s in strategies]
        assert "sft" in slugs

    def test_strategy_response_has_all_fields(self):
        strategies = FinetuningStrategies.list_strategies()
        for s in strategies:
            assert isinstance(s, StrategyResponse)
            assert s.name
            assert s.slug
            assert s.description
            assert isinstance(s.supported_providers, list)

    def test_singleton_same_instance(self):
        instance1 = FinetuningStrategies()
        instance2 = FinetuningStrategies()
        assert instance1 is instance2
