"""
Provider abstraction layer for fine-tuning backends.

This module provides a unified interface for different fine-tuning providers
(e.g., HuggingFace, Unsloth) to enable pluggable backend implementations.
"""

from .base_provider import FinetuningProvider
from .huggingface_provider import HuggingFaceProvider
from .unsloth_provider import UnslothProvider
from .provider_registry import ProviderRegistry, get_provider

# Register providers
ProviderRegistry.register(HuggingFaceProvider)
ProviderRegistry.register(UnslothProvider)

__all__ = [
    "FinetuningProvider",
    "HuggingFaceProvider",
    "UnslothProvider",
    "ProviderRegistry",
    "get_provider",
]
