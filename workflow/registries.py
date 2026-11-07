"""Registry bootstrap helpers.

The core package uses import-time registration for use cases and strategies.
UI entrypoints call these helpers before reading registries so behavior stays
consistent across web, terminal, and notebook surfaces.
"""


def load_core_registries() -> None:
    import recommendation_engine.language_use_cases  # noqa: F401
    import finetuning_strategies.tasks  # noqa: F401
    import finetuning_strategies.strategies.sft  # noqa: F401
    import finetuning_strategies.strategies.dpo  # noqa: F401
    import finetuning_strategies.strategies.grpo  # noqa: F401
    import finetuning_strategies.strategies.qa  # noqa: F401
    import providers.huggingface  # noqa: F401
