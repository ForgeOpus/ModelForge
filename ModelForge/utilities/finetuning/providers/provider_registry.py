"""
Provider registry for managing and selecting fine-tuning providers.

This module provides a centralized registry for all available fine-tuning
providers, enabling runtime provider selection and discovery.
"""

from typing import Dict, Type, List, Optional
from .base_provider import FinetuningProvider


class ProviderRegistry:
    """
    Central registry for managing fine-tuning provider implementations.
    """
    
    _providers: Dict[str, Type[FinetuningProvider]] = {}
    
    @classmethod
    def register(cls, provider_class: Type[FinetuningProvider]) -> None:
        """
        Register a fine-tuning provider.
        
        Args:
            provider_class: Provider class to register
            
        Raises:
            ValueError: If provider is already registered
        """
        provider_name = provider_class.get_provider_name()
        
        if provider_name in cls._providers:
            raise ValueError(f"Provider '{provider_name}' is already registered")
        
        cls._providers[provider_name] = provider_class
    
    @classmethod
    def get(cls, provider_name: str) -> Optional[Type[FinetuningProvider]]:
        """
        Get a provider class by name.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider class or None if not found
        """
        return cls._providers.get(provider_name)
    
    @classmethod
    def list_available(cls) -> List[Dict[str, str]]:
        """
        List all available (installed) providers.
        
        Returns:
            List of dictionaries containing provider information
        """
        available = []
        for name, provider_class in cls._providers.items():
            if provider_class.is_available():
                available.append({
                    "name": name,
                    "description": provider_class.get_provider_description(),
                })
        return available
    
    @classmethod
    def list_all(cls) -> List[Dict[str, str]]:
        """
        List all registered providers (including unavailable ones).
        
        Returns:
            List of dictionaries containing provider information
        """
        all_providers = []
        for name, provider_class in cls._providers.items():
            all_providers.append({
                "name": name,
                "description": provider_class.get_provider_description(),
                "available": provider_class.is_available(),
            })
        return all_providers
    
    @classmethod
    def is_available(cls, provider_name: str) -> bool:
        """
        Check if a provider is available for use.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            True if provider exists and is available, False otherwise
        """
        provider_class = cls.get(provider_name)
        if provider_class is None:
            return False
        return provider_class.is_available()


def get_provider(
    provider_name: str,
    model_name: str,
    task: str,
    compute_specs: str = "low_end"
) -> FinetuningProvider:
    """
    Factory function to instantiate a provider.
    
    Args:
        provider_name: Name of the provider to instantiate
        model_name: Model name or path
        task: Fine-tuning task
        compute_specs: Hardware profile
        
    Returns:
        Instantiated provider
        
    Raises:
        ValueError: If provider not found or not available
    """
    provider_class = ProviderRegistry.get(provider_name)
    
    if provider_class is None:
        available = ProviderRegistry.list_available()
        available_names = [p["name"] for p in available]
        raise ValueError(
            f"Provider '{provider_name}' not found. "
            f"Available providers: {', '.join(available_names)}"
        )
    
    if not provider_class.is_available():
        raise ValueError(
            f"Provider '{provider_name}' is registered but not available. "
            f"Please ensure all required dependencies are installed."
        )
    
    return provider_class(
        model_name=model_name,
        task=task,
        compute_specs=compute_specs
    )
