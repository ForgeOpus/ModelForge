"""
Provider module initialization and registration.

This module automatically registers all available providers at import time,
enabling dynamic provider enumeration based on installed dependencies.
"""

from .base_provider import BaseProvider
from .provider_registry import ProviderRegistry
from .provider_adapter import ProviderFinetuner
from .huggingface_provider import HuggingFaceProvider

# Register HuggingFace provider (always available if transformers is installed)
try:
    ProviderRegistry.register(HuggingFaceProvider)
except Exception as e:
    print(f"Warning: Could not register HuggingFace provider: {e}")

# Register Unsloth provider (soft dependency - only if installed)
try:
    from .unsloth_provider import UnslothProvider
    ProviderRegistry.register(UnslothProvider)
except ImportError:
    print("Unsloth provider not available (unsloth not installed)")
except Exception as e:
    print(f"Warning: Could not register Unsloth provider: {e}")

__all__ = [
    'BaseProvider',
    'ProviderRegistry',
    'ProviderFinetuner',
    'HuggingFaceProvider',
]
