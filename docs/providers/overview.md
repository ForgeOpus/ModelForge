# Provider overview

A **provider** is the part of ModelForge that loads a base model and its tokenizer for training. Strategies decide how to train. The provider decides where the weights come from and how they are instantiated. ModelForge 4 ships one provider, Hugging Face.

## What a provider does

The provider interface is two methods (`providers/provider.py`):

| Method | Called with | Returns |
|---|---|---|
| `load_model(model_name, model_class, **kwargs)` | The model ID or path, the task's `AutoModel` class, and the plan's `model_init_kwargs` | An instantiated model |
| `load_tokenizer(model_name)` | The model ID or path | A tokenizer with a pad token |

Everything else (datasets, chat templates, LoRA, trainers, saving) is handled by the strategy. A provider does not take part in planning, Hub search or the playground.

## Where it fits

```text
Recommendation.provider_slug ──► ProviderFactory.get_provider(slug) ──► provider instance
                                                                          │
strategy.finetune(model_id, dataset, task_slug, provider, ...) ◄──────────┘
    ├─ provider.load_model(model_id, task.model_class, **model_init_kwargs)
    ├─ provider.load_tokenizer(model_id)
    └─ trainer.train()
```

1. The planner sets `provider_slug` on every recommendation. It is always `"huggingface"`.
2. When a run starts, `workflow/training.py` looks the slug up with `ProviderFactory.get_provider` and creates an instance.
3. The strategy calls `load_model` with the model class for the task (for example `AutoModelForCausalLM`) and any loading options from the plan, such as a QLoRA `quantization_config`.
4. DPO with full fine-tuning calls `load_model` a second time, without options, for the frozen reference model.

## Available providers

| Provider | Slug | Libraries | Platforms |
|---|---|---|---|
| Hugging Face | `huggingface` | Transformers, TRL, PEFT, bitsandbytes | CUDA, Apple MPS, CPU |

`GET /api/system` lists registered provider slugs under `providers`. Every built-in strategy (`sft`, `dpo`, `grpo`, `qa`) declares `supported_providers = ["huggingface"]`.

!!! note "Unsloth was removed"
    ModelForge 3 also offered an Unsloth provider and let you pick a provider per run. ModelForge 4 removes Unsloth and the provider choice. Memory savings now come from the planner's tier selection, QLoRA, gradient checkpointing, packing and optional Liger kernels. See [Memory optimizations](../strategies/memory-optimizations.md).

## Choosing a provider

You don't. There is no provider field in the console, the terminal wizard or the run request. The provider comes from the plan's `provider_slug`. A plan sent to `POST /api/runs` with a `provider_slug` the strategy doesn't support is rejected with `Recommendation provider is not supported by this strategy.`

## Registration

Providers register themselves when their class is defined. `Provider.__init_subclass__` checks that the class sets string `name`, `description` and `slug` attributes and that the slug is unique, then adds it to `Provider._registry`. `workflow/registries.py` imports `providers.huggingface` at startup, so the Hugging Face provider is registered before any run.

```python
from workflow.registries import load_core_registries
from providers.provider_factory import ProviderFactory

load_core_registries()
ProviderFactory.get_available_providers()   # ['huggingface']
provider = ProviderFactory.get_provider("huggingface")()
```

An unknown slug raises `ValueError: Unsupported provider type: <slug>. Please choose one of [...]`.

## See also

- [Hugging Face provider](huggingface.md): loading, authentication, cache, Hub search and push.
- [Adding a provider](custom-providers.md)
- [Architecture](../contributing/architecture.md)
