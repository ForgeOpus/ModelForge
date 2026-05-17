import os
from finetuning_strategies.strategy_factory import FinetuningStrategies
from providers.provider_factory import ProviderFactory
from peft import LoraConfig
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
) -> str:
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    strategy_cls = FinetuningStrategies.get_strategy_class(recommendation.strategy_slug)
    strategy = strategy_cls()

    provider_cls = ProviderFactory.get_provider(recommendation.provider_slug)
    provider = provider_cls()

    strategy.build_sft_config(output_dir=output_dir, **recommendation.strategy_config.model_dump())

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

    trainer = strategy.finetune(
        model_id,
        dataset,
        recommendation.task_slug,
        provider,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    if recommendation.peft_approach != "full_finetune":
        trainer.model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)

    return output_dir
