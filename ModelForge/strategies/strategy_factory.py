"""
Strategy factory for creating training strategy instances.

Strategy imports are deferred to avoid pulling in heavy dependencies (trl, wandb)
at module load time — which would break lightweight entry points like the notebook
CLI wizard that only need HardwareService.
"""
from ..exceptions import ConfigurationError
from ..logging_config import logger


def _import_sft():
    from .sft_strategy import SFTStrategy
    return SFTStrategy


def _import_rlhf():
    from .rlhf_strategy import RLHFStrategy
    return RLHFStrategy


def _import_dpo():
    from .dpo_strategy import DPOStrategy
    return DPOStrategy


def _import_qlora():
    from .qlora_strategy import QLoRAStrategy
    return QLoRAStrategy


class StrategyFactory:
    """Factory for creating training strategy instances."""

    _strategy_loaders = {
        "sft": _import_sft,
        "rlhf": _import_rlhf,
        "dpo": _import_dpo,
        "qlora": _import_qlora,
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

        if strategy_name not in cls._strategy_loaders:
            raise ConfigurationError(
                f"Unknown training strategy: {strategy_name}. "
                f"Available strategies: {list(cls._strategy_loaders.keys())}"
            )

        strategy_class = cls._strategy_loaders[strategy_name]()
        return strategy_class()

    @classmethod
    def get_available_strategies(cls) -> list:
        """
        Get list of available strategy names.

        Returns:
            List of strategy names
        """
        return list(cls._strategy_loaders.keys())

    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        """
        Register a new strategy.

        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        logger.info(f"Registering strategy: {name}")
        cls._strategy_loaders[name.lower()] = lambda _cls=strategy_class: _cls
