"""
Provider factory for creating model provider instances.
"""
from typing import Optional

from .huggingface_provider import HuggingFaceProvider
from .unsloth_provider import UnslothProvider
from ..exceptions import ConfigurationError
from ..logging_config import logger


class ProviderFactory:
    """Factory for creating model provider instances."""

    _providers = {
        "huggingface": HuggingFaceProvider,
        "unsloth": UnslothProvider,
    }

    @classmethod
    def create_provider(cls, provider_name: str = "huggingface"):
        """
        Create a provider instance by name.

        Args:
            provider_name: Name of the provider ("huggingface", "unsloth")

        Returns:
            Provider instance

        Raises:
            ConfigurationError: If provider name is not recognized
        """
        logger.info(f"Creating provider: {provider_name}")

        provider_name = provider_name.lower()

        if provider_name not in cls._providers:
            raise ConfigurationError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_name]
        return provider_class()

    @classmethod
    def get_available_providers(cls) -> list:
        """
        Get list of available provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def register_provider(cls, name: str, provider_class):
        """
        Register a new provider.

        Args:
            name: Provider name
            provider_class: Provider class
        """
        logger.info(f"Registering provider: {name}")
        cls._providers[name.lower()] = provider_class
