from finetuning_strategies.strategies import FinetuningStrategy
from pydantic import BaseModel, Field
from typing import Type, List

class StrategyResponse(BaseModel):
    name: str = Field(..., description="The display name of the fine-tuning strategy.")
    slug: str = Field(..., description="The unique slug identifier for the fine-tuning strategy.")
    description: str = Field(..., description="A detailed description of the fine-tuning strategy, including its use cases and requirements.")
    supported_providers: List[str] = Field(..., description="A list of fine-tuning providers that are compatible with this strategy.")

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

    instance = None

    def __new__(cls):
        if cls.instance is not None:
            cls.instance = super(FinetuningStrategies, cls).__new__(cls)
        return cls.instance

    
    @staticmethod
    def get_strategy_class(slug: str) -> Type[FinetuningStrategy]:
        """Get a fine-tuning strategy class by its slug identifier.

        Args:
            slug (str): The unique slug identifier for the desired fine-tuning strategy.

        Returns:
            Type[FinetuningStrategy]: The fine-tuning strategy class associated with the provided slug.

        Raises:
            ValueError: If no strategy is found for the given slug.
        """
        if slug in FinetuningStrategy._registry:
            return FinetuningStrategy._registry[slug]
        else:
            raise ValueError(f"No strategy found for slug '{slug}'")
    
    @staticmethod
    def list_strategies() -> List[StrategyResponse]:
        """List all available fine-tuning strategies with their details.

        Returns:
            List[StrategyResponse]: A list of strategy responses, each containing the name, slug, description,
            and supported providers for a fine-tuning strategy.
        """
        return [
            StrategyResponse(
                name=strategy_cls.name,
                slug=strategy_cls.slug,
                description=strategy_cls.description,
                supported_providers=strategy_cls.supported_providers
            )
            for strategy_cls in FinetuningStrategy._registry.values()
        ]


