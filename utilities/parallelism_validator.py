from __future__ import annotations

from recommendation_engine.validation_models.recommendation import Recommendation
from utilities.parallelism_strategy import ParallelismStrategy


def validate_parallelism_compatibility(recommendation: Recommendation) -> None:
    if (
        recommendation.parallelism_strategy == ParallelismStrategy.FSDP
        and recommendation.peft_approach != "full_finetune"
    ):
        raise ValueError(
            f"FSDP parallelism is not compatible with PEFT approach '{recommendation.peft_approach}'. "
            "FSDP sharding cannot operate over LoRA adapter parameters. "
            "To fix this: either set parallelism_strategy to 'data_parallel', "
            "or set peft_approach to 'full_finetune'."
        )
