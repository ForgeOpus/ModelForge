from __future__ import annotations

import os
from pathlib import Path

from accelerate import notebook_launcher
from peft import LoraConfig

from finetuning_strategies.strategy_factory import FinetuningStrategies
from providers.provider_factory import ProviderFactory
from recommendation_engine.validation_models.recommendation import Recommendation


_TASK_SLUG_TO_PEFT_TASK_TYPE = {
    "causal_language_modeling": "CAUSAL_LM",
    "seq2seq_modeling": "SEQ_2_SEQ_LM",
    "token_classification": "TOKEN_CLS",
}


def assemble_and_train(
    recommendation: Recommendation,
    model_id: str,
    dataset,
    output_dir: str,
    cuda_visible_devices: str,
    callbacks: list,
    tensorboard_log_dir: str | None = None,
    resume_from_checkpoint: str | None = None,
) -> str:
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    strategy_cls = FinetuningStrategies.get_strategy_class(recommendation.strategy_slug)
    strategy = strategy_cls()

    provider_cls = ProviderFactory.get_provider(recommendation.provider_slug)
    provider = provider_cls()

    config_dict = {key: value for key, value in recommendation.strategy_config.model_dump().items() if value is not None}
    if tensorboard_log_dir:
        Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
        config_dict.update({
            "report_to": ["tensorboard"],
            "logging_dir": tensorboard_log_dir,
        })
    strategy.set_config({"output_dir": output_dir, **config_dict})

    if recommendation.peft_approach == "full_finetune" or recommendation.peft_config is None:
        peft_config = None
    else:
        task_type = _TASK_SLUG_TO_PEFT_TASK_TYPE[recommendation.task_slug]
        peft_config = LoraConfig(
            r=recommendation.peft_config.r,
            lora_alpha=recommendation.peft_config.lora_alpha,
            target_modules=recommendation.peft_config.target_modules,
            task_type=task_type,
        )

    notebook_launcher(
        lambda: strategy.finetune(
            model_id,
            dataset,
            recommendation.task_slug,
            provider,
            peft_approach=recommendation.peft_approach,
            output_dir=output_dir,
            peft_config=peft_config,
            callbacks=callbacks,
            resume_from_checkpoint=resume_from_checkpoint,
        ),
        num_processes=len(cuda_visible_devices.split(",")) if cuda_visible_devices else 1,
    )

    return output_dir
