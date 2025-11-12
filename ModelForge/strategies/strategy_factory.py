"""
Strategy factory for creating training strategy instances.
"""
from .sft_strategy import SFTStrategy
from .rlhf_strategy import RLHFStrategy
from .dpo_strategy import DPOStrategy
from .qlora_strategy import QLoRAStrategy
from ..exceptions import ConfigurationError
from ..logging_config import logger


class StrategyFactory:
    """Factory for creating training strategy instances."""

    _strategies = {
        "sft": SFTStrategy,
        "rlhf": RLHFStrategy,
        "dpo": DPOStrategy,
        "qlora": QLoRAStrategy,
    }

    @classmethod
    def create_strategy(cls, strategy_name: str = "sft"):
        """
        Create a strategy instance by name.

        Args:
            strategy_name: Name of the strategy ("sft", "rlhf", "dpo", "qlora")

        Returns:
            Strategy instance

        Raises:
            ConfigurationError: If strategy name is not recognized
        """
        logger.info(f"Creating training strategy: {strategy_name}")

        strategy_name = strategy_name.lower()

        if strategy_name not in cls._strategies:
            raise ConfigurationError(
                f"Unknown training strategy: {strategy_name}. "
                f"Available strategies: {list(cls._strategies.keys())}"
            )

        strategy_class = cls._strategies[strategy_name]
        return strategy_class()

    @classmethod
    def get_available_strategies(cls) -> list:
        """
        Get list of available strategy names.

        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys())

    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        """
        Register a new strategy.

        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        logger.info(f"Registering strategy: {name}")
        cls._strategies[name.lower()] = strategy_class
