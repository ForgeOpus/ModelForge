from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from recommendation_engine.validation_models.recommendation import Recommendation
from utilities.parallelism_strategy import ParallelismStrategy
from utilities.parallelism_validator import validate_parallelism_compatibility


def make_recommendation(parallelism_strategy: ParallelismStrategy, peft_approach: str) -> MagicMock:
    """Build a minimal mock Recommendation with only the fields the validator inspects."""
    rec = MagicMock(spec=Recommendation)
    rec.parallelism_strategy = parallelism_strategy
    rec.peft_approach = peft_approach
    return rec


class TestValidateParallelismCompatibility:
    # ------------------------------------------------------------------
    # FSDP + PEFT combinations that must raise
    # ------------------------------------------------------------------

    def test_fsdp_with_lora_bf16_raises(self):
        rec = make_recommendation(ParallelismStrategy.FSDP, "lora_bf16")
        with pytest.raises(ValueError, match="lora_bf16"):
            validate_parallelism_compatibility(rec)

    def test_fsdp_with_qlora_4bit_raises(self):
        rec = make_recommendation(ParallelismStrategy.FSDP, "qlora_4bit")
        with pytest.raises(ValueError):
            validate_parallelism_compatibility(rec)

    def test_fsdp_with_qlora_8bit_raises(self):
        rec = make_recommendation(ParallelismStrategy.FSDP, "qlora_8bit")
        with pytest.raises(ValueError):
            validate_parallelism_compatibility(rec)

    # ------------------------------------------------------------------
    # FSDP + full_finetune must pass
    # ------------------------------------------------------------------

    def test_fsdp_with_full_finetune_passes(self):
        rec = make_recommendation(ParallelismStrategy.FSDP, "full_finetune")
        # Should not raise
        validate_parallelism_compatibility(rec)

    # ------------------------------------------------------------------
    # DATA_PARALLEL is always compatible regardless of PEFT tier
    # ------------------------------------------------------------------

    def test_data_parallel_with_lora_passes(self):
        rec = make_recommendation(ParallelismStrategy.DATA_PARALLEL, "lora_bf16")
        validate_parallelism_compatibility(rec)

    def test_data_parallel_with_qlora_passes(self):
        rec = make_recommendation(ParallelismStrategy.DATA_PARALLEL, "qlora_4bit")
        validate_parallelism_compatibility(rec)
