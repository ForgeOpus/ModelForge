"""
Provider abstraction layer for ModelForge.
Enables support for multiple model providers (HuggingFace, Unsloth, local, etc.)
"""
from typing import Protocol, Any, Dict, Tuple, Optional
from abc import abstractmethod


class ModelProvider(Protocol):
    """
    Interface for model providers.
    Providers handle model loading, tokenizer loading, and model validation.
    """

    @abstractmethod
    def load_model(
        self,
        model_id: str,
        model_class: str,
        quantization_config: Optional[Any] = None,
        device_map: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """
        Load a model from the provider.

        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-2-7b")
            model_class: Model class name (e.g., "AutoModelForCausalLM")
            quantization_config: Optional quantization configuration
            device_map: Optional device mapping for model
            **kwargs: Additional provider-specific arguments

        Returns:
            Loaded model instance
        """
        ...

    @abstractmethod
    def load_tokenizer(self, model_id: str, **kwargs) -> Any:
        """
        Load a tokenizer from the provider.

        Args:
            model_id: Model identifier
            **kwargs: Additional provider-specific arguments

        Returns:
            Loaded tokenizer instance
        """
        ...

    @abstractmethod
    def validate_model_access(self, model_id: str, model_class: str) -> bool:
        """
        Check if the model is accessible.

        Args:
            model_id: Model identifier
            model_class: Model class name

        Returns:
            True if model is accessible, False otherwise
        """
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of the provider.

        Returns:
            Provider name (e.g., "huggingface", "unsloth")
        """
        ...
