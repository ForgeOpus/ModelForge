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
        device_type: str = "cuda",
        **kwargs
    ) -> Any:
        """
        Load a model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model identifier
            model_class: Model class name
            quantization_config: Optional BitsAndBytesConfig
            device_map: Optional device mapping
            device_type: Device type ("cuda", "mps", or "cpu")
            **kwargs: Additional arguments

        Returns:
            Loaded model instance

        Raises:
            ModelAccessError: If user doesn't have access to the model
            ProviderError: If model loading fails
        """
        logger.info(f"Loading model {model_id} with class {model_class} on {device_type}")

        if model_class not in self.model_class_mapping:
            raise ProviderError(
                f"Unsupported model class: {model_class}. "
                f"Supported: {list(self.model_class_mapping.keys())}"
            )

        model_cls = self.model_class_mapping[model_class]

        try:
            load_kwargs = {
                "use_cache": False,
            }
            
            # Handle device_map based on device type
            if device_type == "mps":
                # MPS doesn't support device_map parameter in HuggingFace
                # Don't set device_map - will load to CPU by default, then move to MPS after
                pass
            elif device_type == "cpu":
                # For CPU, use explicit device_map
                load_kwargs["device_map"] = {"": "cpu"}
            else:
                # For CUDA, use provided device_map or default to GPU 0
                load_kwargs["device_map"] = device_map or {"": 0}

            # Only use quantization_config if not MPS (bitsandbytes doesn't support MPS)
            if quantization_config is not None and device_type != "mps":
                load_kwargs["quantization_config"] = quantization_config
            elif quantization_config is not None and device_type == "mps":
                logger.warning("Quantization config ignored on MPS device (bitsandbytes not supported)")

            load_kwargs.update(kwargs)

            model = model_cls.from_pretrained(model_id, **load_kwargs)
            
            # For MPS, explicitly move model to MPS device after loading
            if device_type == "mps":
                import torch
                logger.info("Moving model to MPS device...")
                model = model.to(torch.device("mps"))
            
            logger.info(f"Successfully loaded model {model_id} on {device_type}")
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
