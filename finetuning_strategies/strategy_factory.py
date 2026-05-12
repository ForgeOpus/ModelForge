from finetuning_strategies.strategies import FinetuningStrategy
from typing import Dict, Type, List, Any

class FinetuningStrategies:
    """
    This module provides a collection of fine-tuning strategies for various types of models,
    including Large Language Models (LLMs). Each strategy is implemented as a class that inherits
    from the FinetuningStrategy base class, allowing for a consistent interface and easy integration
    with different fine-tuning providers. The strategies are designed to be flexible and adaptable
    to different use cases and model architectures, enabling users to select the most appropriate
    fine-tuning approach for their specific needs. By organizing the strategies in a modular way,
    ModelForge can easily extend its capabilities in the future by adding new strategies or
    enhancing existing ones without disrupting the overall architecture of the system.
    """

    _registry: Dict[str, FinetuningStrategy] = {}

    @classmethod
    def register(cls, strategy_class: Type[FinetuningStrategy]):
        def decorator(strategy_class: Type[FinetuningStrategy]):
            if not issubclass(strategy_class, FinetuningStrategy):
                raise ValueError(f"Strategy class '{strategy_class.__name__}' must inherit from FinetuningStrategy")
            if strategy_class.slug in cls._registry:
                raise ValueError(f"Strategy with slug '{strategy_class.slug}' is already registered")
            cls._registry[strategy_class.slug] = strategy_class
        return strategy_class

    @classmethod
    def get_strategy(cls, slug: str) -> Type[FinetuningStrategy]:
        if slug not in cls._registry:
            raise ValueError(f"Strategy with slug '{slug}' is not registered")
        return cls._registry[slug]
    
    @classmethod
    def list_strategies_slugs(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def list_strategies_display_info(cls) -> List[Dict[str, Any]]:
        return [
            {"name": strategy_class.name, 
            "supported_providers": strategy_class.supported_providers} 
            for strategy_class in cls._registry.values()
        ]

    @classmethod
    def get_strategy_display_info(cls, slug: str) -> Dict[str, Any]:
        strategy_class: type[Any] = cls.get_strategy(slug)
        return {
            "name": strategy_class.name,
            "supported_providers": strategy_class.supported_providers,
            "description": strategy_class.description
        }

    
    @classmethod
    def clear_registry(cls):
        cls._registry.clear()
        


