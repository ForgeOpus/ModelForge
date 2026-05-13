from abc import ABC, abstractmethod
from providers import SUPPORTED_PROVIDERS
from typing import List, Dict, Literal, Any
import torch

class FinetuningStrategy(ABC):
    
    name: str = None
    slug: str = None
    description: str = None

    _registry = {}

    supported_providers: Literal[tuple(SUPPORTED_PROVIDERS)] = None

    device: Literal['cpu', 'cuda', 'mps'] = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'slug') and cls.slug is not None and isinstance(cls.slug, str):
            if cls.slug in FinetuningStrategy._registry:
                raise ValueError(f"Duplicate slug '{cls.slug}' found for strategy class '{cls.__name__}'. Please ensure that each strategy class has a unique slug.")
            FinetuningStrategy._registry[cls.slug] = cls
        else:
            raise ValueError(f"Strategy class '{cls.__name__}' must define a 'slug' attribute")
        
        if cls.name is None or not isinstance(cls.name, str):
            raise ValueError(f"Strategy class '{cls.__name__}' must define a 'name' attribute of type string")
        if cls.description is None or not isinstance(cls.description, str):
            raise ValueError(f"Strategy class '{cls.__name__}' must define a 'description' attribute of type string")
        if cls.supported_providers is None:
            raise ValueError(f"Strategy class '{cls.__name__}' must define a 'supported_providers' attribute")
        if not isinstance(cls.supported_providers, list) or not all(isinstance(provider, str) for provider in cls.supported_providers):
            raise ValueError(f"Strategy class '{cls.__name__}' must have 'supported_providers' as a list of strings")

    @abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError("Please add a method to configure trainer parameters for this strategy.")

    @abstractmethod
    def finetune(self, model, data):
        raise NotImplementedError("Please add an entry point method to trigger the fine-tuning process for this strategy.")

    def get_supported_providers(self) -> List[str]:
        return self.supported_providers

    def get_name(self) -> str:
        return self.name