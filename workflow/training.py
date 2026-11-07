from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from accelerate import notebook_launcher
from peft import LoraConfig

from finetuning_strategies.strategy_factory import FinetuningStrategies
from providers.provider_factory import ProviderFactory
from recommendation_engine.validation_models.recommendation import Recommendation
from dataset_manager.dataset_factory import DatasetFactory
from utilities.parallelism_strategy import ParallelismStrategy
from utilities.fsdp_config import build_fsdp_config


def _as_dict(config: Any) -> dict:
    """Normalise a config that may be a pydantic model or a plain dict.

    A Recommendation built in-process carries typed config objects, but one
    that has round-tripped through the web API is rebuilt from JSON into
    dicts, because Recommendation types both config fields as Any. Both reach
    this module, so both have to work."""
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        return config.model_dump()
    return dict(config)


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


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

    # The use-case contract describes user-facing names (for example
    # document/summary). Trainers consume canonical names instead. Normalize
    # once at the shared execution boundary so all launch surfaces train the
    # same data that validation accepted.
    dataset = DatasetFactory.normalize_columns(
        dataset, recommendation.data_format.column_aliases
    )

    provider_cls = ProviderFactory.get_provider(recommendation.provider_slug)
    provider = provider_cls()

    num_processes = len(cuda_visible_devices.split(",")) if cuda_visible_devices else 1

    config_dict = {key: value for key, value in _as_dict(recommendation.strategy_config).items() if value is not None}
    if tensorboard_log_dir:
        Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
        config_dict.update({
            "report_to": ["tensorboard"],
            "logging_dir": tensorboard_log_dir,
        })
    # FSDP is activated through the trainer's own fsdp/fsdp_config arguments so the Trainer's
    # internal Accelerator actually enables it. Guarded on num_processes >= 2 because the engine's
    # strategy choice may have been made against a different GPU set than this run (e.g. a stored
    # recommendation replayed via resume, or a single-GPU subset selected at launch).
    if recommendation.parallelism_strategy == ParallelismStrategy.FSDP and num_processes >= 2:
        config_dict.update(build_fsdp_config())
    strategy.set_config({"output_dir": output_dir, **config_dict})

    if recommendation.peft_approach == "full_finetune" or recommendation.peft_config is None:
        peft_config = None
    else:
        task_type = _TASK_SLUG_TO_PEFT_TASK_TYPE[recommendation.task_slug]
        peft_config = LoraConfig(
            r=_get(recommendation.peft_config, "r"),
            lora_alpha=_get(recommendation.peft_config, "lora_alpha"),
            target_modules=_get(recommendation.peft_config, "target_modules"),
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
        num_processes=num_processes,
    )

    return output_dir
