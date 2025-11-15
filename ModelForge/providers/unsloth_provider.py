"""
Unsloth provider implementation.
Provides optimized model loading for faster fine-tuning.
"""
from typing import Any, Dict, Optional, Tuple

from ..exceptions import ProviderError
from ..logging_config import logger


class UnslothProvider:
    """
    Provider for Unsloth-optimized models.
    Unsloth provides 2x faster training with lower memory usage.
    """

    def __init__(self):
        self._check_unsloth_available()

    def _check_unsloth_available(self):
        """Check if unsloth is installed."""
        try:
            import unsloth
            logger.info("Unsloth is available")
        except ImportError:
            logger.warning(
                "Unsloth is not installed. "
                "Install with: pip install unsloth"
            )

    def load_model(
        self,
        model_id: str,
        model_class: str,
        quantization_config: Optional[Any] = None,
        device_map: Optional[Dict] = None,
        max_seq_length: int = 2048,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load a model using Unsloth optimizations.

        Args:
            model_id: Model identifier
            model_class: Model class name (Unsloth auto-detects)
            quantization_config: Quantization config (Unsloth uses different API)
            device_map: Device mapping (handled by Unsloth)
            max_seq_length: Maximum sequence length
            **kwargs: Additional arguments

        Returns:
            Tuple of (model, tokenizer) - Unsloth returns both together

        Raises:
            ProviderError: If Unsloth is not available or loading fails
        """
        logger.info(f"Loading model {model_id} with Unsloth optimizations")

        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ProviderError(
                "Unsloth is not installed. Install with: pip install unsloth"
            ) from e

        try:
            # Extract quantization settings
            load_in_4bit = False
            dtype = None

            if quantization_config is not None:
                # BitsAndBytesConfig object
                if hasattr(quantization_config, 'load_in_4bit'):
                    load_in_4bit = quantization_config.load_in_4bit
                elif hasattr(quantization_config, 'load_in_8bit'):
                    load_in_4bit = False  # Unsloth primarily uses 4-bit

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_seq_length,
                dtype=dtype,  # Auto-detect
                load_in_4bit=load_in_4bit,
            )

            # Remove hf_device_map attribute to prevent Accelerate device placement conflicts
            # Unsloth sets device_map="sequential" by default, which causes None device_index errors
            # Removing this attribute allows Accelerate to handle device placement correctly
            if hasattr(model, 'hf_device_map'):
                delattr(model, 'hf_device_map')
                logger.debug("Removed hf_device_map attribute for Accelerate compatibility")

            logger.info(f"Successfully loaded model {model_id} with Unsloth")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model with Unsloth: {e}")
            raise ProviderError(
                f"Failed to load model {model_id} with Unsloth: {str(e)}"
            ) from e

    def load_tokenizer(self, model_id: str, **kwargs) -> Any:
        """
        Load tokenizer (Unsloth loads it with the model).

        Note: For Unsloth, use load_model() which returns both model and tokenizer.

        Args:
            model_id: Model identifier
            **kwargs: Additional arguments

        Returns:
            Tokenizer instance
        """
        logger.warning(
            "Unsloth loads tokenizer with model. "
            "Use load_model() instead for best results."
        )

        # Fallback to standard loading
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id, **kwargs)

    def prepare_for_training(
        self,
        model: Any,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None,
        **kwargs
    ) -> Any:
        """
        Prepare model for training with Unsloth optimizations.

        Args:
            model: Model instance from load_model()
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: Target modules for LoRA (Unsloth auto-detects)
            **kwargs: Additional arguments

        Returns:
            PEFT model ready for training
        """
        logger.info("Preparing model for training with Unsloth")

        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ProviderError(
                "Unsloth is not installed. Install with: pip install unsloth"
            ) from e

        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules or [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                bias="none",
                use_gradient_checkpointing="unsloth",  # Unsloth optimization
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

            logger.info("Model prepared for training with Unsloth")
            return model

        except Exception as e:
            logger.error(f"Error preparing model for training: {e}")
            raise ProviderError(
                f"Failed to prepare model for training: {str(e)}"
            ) from e

    def validate_model_access(self, model_id: str, model_class: str) -> bool:
        """
        Check if the model is accessible.

        Args:
            model_id: Model identifier
            model_class: Model class name

        Returns:
            True if model is accessible
        """
        # Unsloth uses HuggingFace Hub, so same validation applies
        logger.info(f"Validating access to model {model_id}")

        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(model_id)
            logger.info(f"Model {model_id} is accessible")
            return True

        except Exception as e:
            logger.error(f"Model {model_id} validation failed: {e}")
            return False

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "unsloth"
