from typing import Dict, Any, Type, List
from providers.provider import Provider

class ProviderFactory:
    """
    Factory for generating model providers
    """

    _registry: Dict[str, Type[Provider]] = {}

    @classmethod
    def register(cls, provider: Type[{Provider}]):
        def decorator(provider: Type[{Provider}]):
            if not issubclass(provider, {Provider}):
                raise ValueError(f"Task type '{provider.__name__}' must be inherited from the base FinetuningTask")
            if provider.slug in cls._registry:
                raise ValueError(f"Task of type '{provider.slug}' already registered. Please choose a unique slug")
        return provider

    @classmethod
    def get_available_providers(cls) -> List[str, Type[Provider]]:
        return list(cls._registry.keys())
    
    @classmethod
    def get_provider(cls, slug) -> Type[Provider]:
        if not slug in cls._registry:
            raise ValueError(f"Unsupported provider type: {slug}. Please choose one of {cls.get_available_providers()}")
        return cls._registry(slug)
        
    def clear_registry(cls) -> None:
        cls._registry.clear()

