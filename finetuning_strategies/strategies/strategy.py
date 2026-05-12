from accelerate.commands.config.default import description
from abc import ABC, abstractmethod
from providers import SUPPORTED_PROVIDERS
from typing import List, Optional, Dict, Literal, Any
import torch

class FinetuningStrategy(ABC):
    
    name: str = None
    slug: str = None
    description: str = None


    supported_providers: Literal[tuple(SUPPORTED_PROVIDERS)] = None

    device: Literal['cpu', 'cuda', 'mps'] = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    @property
    @abstractmethod
    def name(self) -> str:
        """Define the display name fo rthe strategy."""
        raise NotImplementedError("Please add a name")

    @property
    @abstractmethod
    def slug(self) -> str:
        """Define a unique slug identifier for the strategy."""
        raise NotImplementedError("Please add a slug")

    @property
    @abstractmethod
    def supported_providers(self) -> List[str]:
        """Define a list of supported fine-tuning providers for this strategy.
        This should return a list of provider names that are compatible with the strategy.
        """
        raise NotImplementedError("Please add the supported providers for this strategy")

    @property
    @abstractmethod
    def description(self) -> str:
        """A description of the strategy that can be displayed to users when they are selecting a strategy for fine-tuning. 
        This should provide a clear and concise overview of the strategy, its use cases, and any important considerations or 
        requirements for using it effectively.
        """
        raise NotImplementedError("Please add a description")

    @property
    @abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        # Implement logic to set configuration parameters for the fine-tuning strategy
        raise NotImplementedError("Please add a ")

    @property
    @abstractmethod
    def finetune(self, model, data):
        raise NotImplementedError("Please add a ")

    def get_supported_providers(self) -> List[str]:
        return self.supported_providers

    def get_name(self) -> str:
        return self.name