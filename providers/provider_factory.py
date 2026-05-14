from typing import Dict, Any, Type, List
from providers.provider import Provider

class ProviderFactory:
    """
    Factory for generating model providers
    """

    @classmethod
    def get_available_providers(cls) -> List[str]:
        return list(Provider._registry.keys())
    
    @classmethod
    def get_provider(cls, slug) -> Type[Provider]:
        if not slug in Provider._registry:
            raise ValueError(f"Unsupported provider type: {slug}. Please choose one of {cls.get_available_providers()}")
        return Provider._registry[slug]

    @classmethod
    def clear_registry(cls) -> None:
        Provider._registry.clear()

