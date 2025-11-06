"""
Provider registry for managing available fine-tuning providers.

This module provides centralized provider registration and factory functionality
for dynamic provider enumeration and instantiation.
"""

from typing import Dict, List, Type, Optional
from .base_provider import BaseProvider


class ProviderRegistry:
    """
    Central registry for fine-tuning providers.
    
    Manages provider registration, availability checking, and factory instantiation.
    """
    
    _providers: Dict[str, Type[BaseProvider]] = {}
    
    @classmethod
    def register(cls, provider_class: Type[BaseProvider]) -> None:
        """
        Register a provider class.
        
        Args:
            provider_class: Provider class to register
        """
        provider_name = provider_class.get_provider_name()
        cls._providers[provider_name] = provider_class
        print(f"Registered provider: {provider_name}")
    
    @classmethod
    def get(cls, provider_name: str) -> Optional[Type[BaseProvider]]:
        """
        Get a provider class by name.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider class if found, None otherwise
        """
        return cls._providers.get(provider_name)
    
    @classmethod
    def is_available(cls, provider_name: str) -> bool:
        """
        Check if a provider is available.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            True if provider exists and its dependencies are available
        """
        provider_class = cls.get(provider_name)
        if provider_class is None:
            return False
        return provider_class.is_available()
    
    @classmethod
    def list_all(cls) -> List[Dict[str, any]]:
        """
        List all registered providers with availability status.
        
        Returns:
            List of provider information dictionaries
        """
        providers = []
        for name, provider_class in cls._providers.items():
            providers.append({
                "name": name,
                "available": provider_class.is_available(),
                "supported_tasks": provider_class.get_supported_tasks()
            })
        return providers
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List names of available providers only.
        
        Returns:
            List of available provider names
        """
        return [
            name for name, provider_class in cls._providers.items()
            if provider_class.is_available()
        ]
    
    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        model_name: str,
        task: str,
        compute_specs: str
    ) -> BaseProvider:
        """
        Factory method to create a provider instance.
        
        Args:
            provider_name: Name of the provider to create
            model_name: Model name for fine-tuning
            task: Task type
            compute_specs: Compute profile
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider not found or not available
        """
        provider_class = cls.get(provider_name)
        if provider_class is None:
            raise ValueError(f"Provider '{provider_name}' not found")
        
        if not provider_class.is_available():
            raise ValueError(
                f"Provider '{provider_name}' is not available. "
                f"Please install required dependencies."
            )
        
        return provider_class(model_name, task, compute_specs)
