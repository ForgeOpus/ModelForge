"""
Quantization configuration factory.
Consolidates quantization logic to eliminate code duplication.
"""
import torch
from transformers import BitsAndBytesConfig
from typing import Optional

from ...logging_config import logger


class QuantizationFactory:
    """Factory for creating quantization configurations."""

    @staticmethod
    def create_config(
        use_4bit: bool = False,
        use_8bit: bool = False,
        compute_dtype: str = "float16",
        quant_type: str = "nf4",
        use_double_quant: bool = False,
    ) -> Optional[BitsAndBytesConfig]:
        """
        Create a BitsAndBytes quantization configuration.

        Args:
            use_4bit: Whether to use 4-bit quantization
            use_8bit: Whether to use 8-bit quantization
            compute_dtype: Compute dtype (float16, bfloat16, float32)
            quant_type: Quantization type (nf4, fp4)
            use_double_quant: Whether to use nested quantization

        Returns:
            BitsAndBytesConfig if quantization is enabled, None otherwise
        """
        if not use_4bit and not use_8bit:
            logger.info("No quantization enabled")
            return None

        # Convert compute dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype_torch = dtype_map.get(compute_dtype, torch.float16)

        if use_4bit:
            logger.info(
                f"Creating 4-bit quantization config: "
                f"dtype={compute_dtype}, quant_type={quant_type}, "
                f"double_quant={use_double_quant}"
            )
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_compute_dtype=compute_dtype_torch,
                bnb_4bit_use_double_quant=use_double_quant,
            )

        elif use_8bit:
            logger.info("Creating 8-bit quantization config")
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )

        return None

    @staticmethod
    def get_recommended_config(compute_profile: str = "low_end") -> dict:
        """
        Get recommended quantization settings based on compute profile.

        Args:
            compute_profile: Compute profile (low_end, mid_range, high_end)

        Returns:
            Dictionary of recommended settings
        """
        configs = {
            "low_end": {
                "use_4bit": True,
                "use_8bit": False,
                "compute_dtype": "float16",
                "quant_type": "nf4",
                "use_double_quant": False,
            },
            "mid_range": {
                "use_4bit": True,
                "use_8bit": False,
                "compute_dtype": "bfloat16",
                "quant_type": "nf4",
                "use_double_quant": True,
            },
            "high_end": {
                "use_4bit": False,
                "use_8bit": False,
                "compute_dtype": "bfloat16",
                "quant_type": "nf4",
                "use_double_quant": False,
            },
        }

        return configs.get(compute_profile, configs["low_end"])
