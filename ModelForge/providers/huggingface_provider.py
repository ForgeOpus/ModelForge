"""
HuggingFace provider implementation.
Handles model loading from HuggingFace Hub.
"""
from typing import Any, Dict, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)
from huggingface_hub import errors as hf_errors

from ..exceptions import ModelAccessError, ProviderError
from ..logging_config import logger


class HuggingFaceProvider:
    """Provider for HuggingFace models."""

    def __init__(self):
        self.model_class_mapping = {
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
            "AutoModelForQuestionAnswering": AutoModelForQuestionAnswering,
        }

    def load_model(
        self,
        model_id: str,
        model_class: str,
        quantization_config: Optional[Any] = None,
        device_map: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """
        Load a model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model identifier
            model_class: Model class name
            quantization_config: Optional BitsAndBytesConfig
            device_map: Optional device mapping
            **kwargs: Additional arguments

        Returns:
            Loaded model instance

        Raises:
            ModelAccessError: If user doesn't have access to the model
            ProviderError: If model loading fails
        """
        logger.info(f"Loading model {model_id} with class {model_class}")

        if model_class not in self.model_class_mapping:
            raise ProviderError(
                f"Unsupported model class: {model_class}. "
                f"Supported: {list(self.model_class_mapping.keys())}"
            )

        model_cls = self.model_class_mapping[model_class]

        try:
            load_kwargs = {
                "device_map": device_map or {"": 0},
                "use_cache": False,
            }

            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config

            load_kwargs.update(kwargs)

            model = model_cls.from_pretrained(model_id, **load_kwargs)
            logger.info(f"Successfully loaded model {model_id}")
            return model

        except hf_errors.GatedRepoError as e:
            logger.error(f"Access denied to model {model_id}")
            raise ModelAccessError(
                f"You do not have access to model {model_id}. "
                f"Please visit https://huggingface.co/{model_id} to request access."
            ) from e

        except hf_errors.HfHubHTTPError as e:
            logger.error(f"HuggingFace HTTP error loading {model_id}: {e}")
            raise ProviderError(
                f"Network error loading model {model_id}. Please check your connection."
            ) from e

        except Exception as e:
            logger.error(f"Unexpected error loading model {model_id}: {e}")
            raise ProviderError(
                f"Failed to load model {model_id}: {str(e)}"
            ) from e

    def load_tokenizer(self, model_id: str, **kwargs) -> Any:
        """
        Load a tokenizer from HuggingFace Hub.

        Args:
            model_id: HuggingFace model identifier
            **kwargs: Additional arguments

        Returns:
            Loaded tokenizer instance

        Raises:
            ProviderError: If tokenizer loading fails
        """
        logger.info(f"Loading tokenizer for {model_id}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=kwargs.get("trust_remote_code", True)
            )

            # Configure tokenizer for training
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            logger.info(f"Successfully loaded tokenizer for {model_id}")
            return tokenizer

        except Exception as e:
            logger.error(f"Error loading tokenizer for {model_id}: {e}")
            raise ProviderError(
                f"Failed to load tokenizer for {model_id}: {str(e)}"
            ) from e

    def validate_model_access(self, model_id: str, model_class: str) -> bool:
        """
        Check if the model is accessible on HuggingFace Hub.

        Args:
            model_id: HuggingFace model identifier
            model_class: Model class name

        Returns:
            True if model is accessible, False otherwise
        """
        logger.info(f"Validating access to model {model_id}")

        if model_class not in self.model_class_mapping:
            logger.error(f"Unsupported model class: {model_class}")
            return False

        try:
            model_cls = self.model_class_mapping[model_class]
            # Try to load config only (lightweight check)
            from transformers import AutoConfig
            AutoConfig.from_pretrained(model_id)
            logger.info(f"Model {model_id} is accessible")
            return True

        except hf_errors.GatedRepoError:
            logger.error(f"Model {model_id} is gated - access denied")
            return False

        except Exception as e:
            logger.error(f"Model {model_id} validation failed: {e}")
            return False

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "huggingface"
