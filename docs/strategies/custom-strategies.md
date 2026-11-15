# Custom strategies

This is a developer guide to adding a training strategy to ModelForge v4: the base class contract, how registration works, how to make the strategy reachable from a use case or objective, how to set its memory multipliers, and which tests to add. It assumes a source checkout. See [Development setup](../contributing/development.md).

A strategy owns one training algorithm. It declares what tasks and providers it supports and how much memory it needs, builds a hardware-aware default config, and runs training. It doesn't choose the model, the PEFT tier or the parallelism strategy. The recommendation engine does that. See [Strategies overview](overview.md).

## Where strategies plug in

```text
use case + objective
   │  RecommendationEngine._resolve_use_case_and_strategy
   ▼
strategy class  ── extra_model_copies, batch_multiplier ──► tier ceilings, model search
   │  build_recommended_config(peft_tier, hardware, chosen_model)
   ▼
Recommendation(strategy_slug, strategy_config, peft_config, ...)
   │  WorkflowService.validate_dataset ─► Strategy.validate_data_format(split)
   │  workflow/training.py::assemble_and_train
   ▼
strategy = Strategy()
strategy.set_config({...})
strategy.finetune(model_id, dataset, task_slug, provider, ...)   # once per GPU, via Accelerate
```

## The base class

`finetuning_strategies/strategies/strategy.py` defines `FinetuningStrategy(ABC)`.

### Required class attributes

`__init_subclass__` checks these when the class is defined and raises `ValueError` if one is missing or invalid:

| Attribute | Type | Rule |
|---|---|---|
| `slug` | `str` | Unique across all strategies. A duplicate raises `Duplicate slug '...'`. |
| `name` | `str` | Display name. |
| `description` | `str` | Shown in `/api/system`. |
| `supported_providers` | `list[str]` | Must be a **list** of strings, for example `["huggingface"]`. |
| `supported_tasks` | `list[str]` | Non-empty list of task slugs: `causal_language_modeling`, `seq2seq_modeling`, `question_answering` or `token_classification`. |
| `extra_model_copies` | `int` | `>= 0`. Additional full model copies in VRAM. |
| `batch_multiplier` | `float` | `> 0.0`. Activation memory relative to SFT's 1.0. |

### Abstract methods

These are the exact signatures in `strategy.py`:

```python
@abstractmethod
def set_config(self, config: Dict[str, Any]) -> None: ...

@abstractmethod
def finetune(self, model, data, task_slug: str, provider,
             peft_approach: Literal["full_finetune", "lora_bf16", "qlora_8bit", "qlora_4bit"],
             output_dir: str, peft_config=None, callbacks=None,
             resume_from_checkpoint: str | None = None) -> Any: ...

@classmethod
@abstractmethod
def build_recommended_config(cls, peft_tier: str, hardware: Any, chosen_model: Any) -> tuple[Any, Any | None]: ...
```

### What callers actually rely on

The abstract signatures are loose. The launch and validation code expects more than they state:

- **A no-argument constructor.** `assemble_and_train` calls `strategy_cls()`.
- **A static `validate_data_format(dataset)`.** It isn't declared on the base class, but `WorkflowService.validate_dataset` and the terminal wizard call `strategy_cls.validate_data_format(split)` on the alias-normalized dataset. Raise `ValueError` with a clear message on bad data. Without it, dataset validation fails with `AttributeError`.
- **`finetune` called as:**

    ```python
    strategy.finetune(
        model_id,                 # str: Hub ID, not a loaded model
        dataset,                  # datasets.Dataset, aliases already renamed to canonical columns
        recommendation.task_slug,
        provider,                 # Provider instance: use provider.load_model / load_tokenizer
        peft_approach=recommendation.peft_approach,
        output_dir=output_dir,
        peft_config=peft_config,  # peft.LoraConfig with task_type set, or None for full fine-tuning
        callbacks=callbacks,      # list of transformers.TrainerCallback
        resume_from_checkpoint=resume_from_checkpoint,
    )
    ```

    The first four arguments are positional. Name the first two parameters `model_name` and `dataset` as the shipped strategies do. No evaluation dataset is passed.

- **`set_config` receives a single dict** containing:
    - `output_dir`;
    - every non-`None` field of your recommended config. After a round trip through the HTTP API this is a plain dict, possibly edited by the user;
    - when TensorBoard is on: `report_to=["tensorboard"]`, `logging_dir`, `logging_first_step=True`;
    - under FSDP with 2+ GPUs: `fsdp`, `fsdp_config`, `gradient_checkpointing=False`, `use_liger_kernel=False`, `activation_offloading=False`.

    Your `set_config` must accept all of these. If it forwards them to a TRL or Transformers config class that lacks one of them, the run fails.

- **Pass `callbacks` to your Trainer.** `JobEventCallback` streams progress to the browser, records checkpoints from `args.output_dir/checkpoint-<step>` and stops training on cancel. Without it, runs show no progress and can't be cancelled.
- **Save to `output_dir`.** Use `trainer.save_model(output_dir)` for full fine-tuning, `trainer.model.save_pretrained(output_dir)` for adapters, and save the tokenizer. The playground and the model registry read from there.
- **Call `self.launch_tensorboard(args, trainer=trainer)`** before `trainer.train(...)`. It's a no-op unless `report_to` includes TensorBoard and the process is rank 0. It also points Transformers 5's TensorBoard callback at the run's log directory.

`build_recommended_config` returns `(strategy_config, peft_config_or_None)`, both Pydantic models. The engine stores them without inspection. Return `None` as the second item for `full_finetune`. A `peft_config` needs `r`, `lora_alpha` and `target_modules`; `assemble_and_train` reads exactly those three. You can reuse `LoraRecommendedConfig` from `sft.py`.

`hardware` is a `HardwareDetector.HardwareSpecs`. GPUs are dicts with `name` and `total_memory_gb`, which is a float, or `"N/A"` on MPS. `chosen_model` is a `RecommendedModel` with `model_id`, `num_parameters` and `downloads`.

## Skeleton

This example trains a causal LM with the Transformers `Trainer` on `prompt` and `completion` columns. Replace the training body with your algorithm. Everything around it matches what the launcher expects.

```python title="finetuning_strategies/strategies/my_strategy.py"
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

from datasets import Dataset, IterableDataset
from peft import PeftConfig, get_peft_model
from pydantic import BaseModel
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from finetuning_strategies.model_factory import TaskModels
from finetuning_strategies.strategies.strategy import FinetuningStrategy
from finetuning_strategies.strategies.sft import LoraRecommendedConfig
from providers.provider import Provider
from recommendation_engine.validation_models.recommendation import RecommendedModel
from utilities.hardware_detector import HardwareDetector
from utilities.precision import training_precision


class MyRecommendedConfig(BaseModel):
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_length: int
    learning_rate: float
    bf16: bool
    fp16: bool
    model_init_kwargs: dict
    num_train_epochs: float = 3.0
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42


QUANTIZATION = {
    "full_finetune": {},
    "lora_bf16": {},
    "qlora_8bit": {"quantization_config": {"load_in_8bit": True}},
    "qlora_4bit": {"quantization_config": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                           "bnb_4bit_use_double_quant": True}},
}

# Keys the launcher may add that TrainingArguments accepts.
_PASSTHROUGH = {"report_to", "logging_dir", "logging_first_step", "fsdp", "fsdp_config"}


class MyStrategy(FinetuningStrategy):
    name = "My Strategy"
    slug = "my_strategy"
    description = "One sentence on what this trains and what data it needs."
    supported_providers = ["huggingface"]
    supported_tasks = ["causal_language_modeling"]
    extra_model_copies = 0
    batch_multiplier = 1.0

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}

    @staticmethod
    def validate_data_format(dataset: Dataset | IterableDataset) -> bool:
        missing = {"prompt", "completion"} - set(dataset.column_names)
        if missing:
            raise ValueError(f"My Strategy requires prompt and completion columns. Missing: {', '.join(sorted(missing))}.")
        return True

    def set_config(self, config: Dict[str, Any]) -> None:
        # Accept every key; read only the ones this strategy uses. FSDP adds
        # gradient_checkpointing=False, use_liger_kernel and activation_offloading.
        self.config.update({k: v for k, v in config.items() if v is not None})

    @classmethod
    def build_recommended_config(
        cls,
        peft_tier: str,
        hardware: HardwareDetector.HardwareSpecs,
        chosen_model: RecommendedModel,
    ) -> tuple[MyRecommendedConfig, LoraRecommendedConfig | None]:
        if peft_tier not in QUANTIZATION:
            raise ValueError(f"Unknown peft_tier '{peft_tier}'.")
        raw_vram_gb = sum(g["total_memory_gb"] for g in hardware.gpus
                          if isinstance(g["total_memory_gb"], (int, float)))
        bf16, fp16 = training_precision(hardware)
        config = MyRecommendedConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16 if raw_vram_gb < 12 else 8,
            gradient_checkpointing=True,
            max_length=512 if raw_vram_gb < 12 else 1024,
            learning_rate=2e-5,
            bf16=bf16,
            fp16=fp16,
            # bitsandbytes needs CUDA: no quantization on CPU or MPS.
            model_init_kwargs=QUANTIZATION[peft_tier] if raw_vram_gb > 0 else {},
        )
        if peft_tier == "full_finetune":
            return config, None
        rank = 8 if chosen_model.num_parameters < 1_000_000_000 else 16
        return config, LoraRecommendedConfig(r=rank, lora_alpha=rank * 2, target_modules="all-linear")

    def finetune(
        self,
        model_name: str,
        dataset: Union[Dataset, IterableDataset],
        task_slug: str,
        provider: Provider,
        peft_approach: Literal["full_finetune", "lora_bf16", "qlora_8bit", "qlora_4bit"],
        output_dir: str,
        peft_config: Optional[PeftConfig] = None,
        callbacks: Optional[list] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Trainer:
        if task_slug not in self.supported_tasks:
            raise ValueError(f"My Strategy only supports {self.supported_tasks}; got '{task_slug}'.")
        self.validate_data_format(dataset)
        cfg = self.config
        task_class = TaskModels.get_task_class(task_slug)
        model = provider.load_model(model_name, task_class.model_class, **cfg.get("model_init_kwargs", {}))
        tokenizer = provider.load_tokenizer(model_name)

        max_length = cfg.get("max_length", 1024)
        dataset = dataset.map(
            lambda batch: tokenizer(
                [p + c for p, c in zip(batch["prompt"], batch["completion"])],
                truncation=True, max_length=max_length,
            ),
            batched=True, remove_columns=dataset.column_names,
        )
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
            learning_rate=cfg.get("learning_rate", 2e-5),
            num_train_epochs=cfg.get("num_train_epochs", 3.0),
            logging_steps=cfg.get("logging_steps", 10),
            save_steps=cfg.get("save_steps", 500),
            seed=cfg.get("seed", 42),
            bf16=cfg.get("bf16", False),
            fp16=cfg.get("fp16", False),
            **{k: cfg[k] for k in _PASSTHROUGH if k in cfg},
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=callbacks,
        )
        self.launch_tensorboard(args, trainer=trainer)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if peft_approach == "full_finetune":
            trainer.save_model(output_dir)
        else:
            trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        return trainer
```

## Register it

Subclasses register themselves when their module is imported. Nothing has to be added to `FinetuningStrategies` in `strategy_factory.py`: `get_strategy_class(slug)` and `list_strategies()` read the same `FinetuningStrategy._registry`. The module still has to be imported in two places:

1. `finetuning_strategies/strategies/__init__.py`:

    ```python
    from finetuning_strategies.strategies.my_strategy import MyStrategy
    __all__ = ["FinetuningStrategy", "SFT", "DPO", "GRPO", "ExtractiveQA", "MyStrategy"]
    ```

2. `workflow/registries.py::load_core_registries`, which the web, terminal and notebook entry points call before reading registries:

    ```python
    import finetuning_strategies.strategies.my_strategy  # noqa: F401
    ```

Check it:

```bash
python -c "from workflow.registries import load_core_registries; load_core_registries(); \
from finetuning_strategies.strategy_factory import FinetuningStrategies; \
print(FinetuningStrategies.get_strategy_class('my_strategy'))"
```

It also appears under `strategies` in `GET /api/system`.

## Make it reachable

A registered strategy isn't used until a use case resolves to it. The engine takes the use case's `strategy_slug` if set, otherwise `OBJECTIVE_STRATEGY_MAP[objective]`. It then checks that the use case's `task_slug` is in your `supported_tasks`. There are two ways in.

### Option A: a use case that always uses it

Set `strategy_slug` on a use case, as `ExtractiveQAUseCase` does with `"qa"`. That replaces the objective mapping for **every** objective of that use case. It's the right choice for a new use case with one supervised path. Adding a use case is covered in [Adding a use case](../contributing/use-cases.md).

```python
class MyUseCase(LanguageUseCase):
    ...
    task_slug = "causal_language_modeling"
    strategy_slug = "my_strategy"
    compatible_objectives = [TrainingObjective.SUPERVISED]
    data_formats = {TrainingObjective.SUPERVISED: DataFormatSpec(
        user_facing_description="A dataset with 'prompt' and 'completion' columns.",
        column_aliases={"prompt": "prompt", "completion": "completion"},
    )}
```

`column_aliases` maps source names to the canonical names your `validate_data_format` and `finetune` read. Every key is required unless it's listed in `optional_column_aliases`.

### Option B: a new objective

To offer the strategy as a new objective on existing use cases:

1. Add a member to `TrainingObjective` in `utilities/training_objective.py`.
2. Map it in `OBJECTIVE_STRATEGY_MAP` in `utilities/recommendation_engine.py`.
3. Map it in `_OBJECTIVE_TO_SLUG` in `cli/wizard.py`. The terminal wizard keeps its own copy of the mapping to decide which objectives to offer.
4. Add it to each use case's `compatible_objectives`, with a `DataFormatSpec` in `data_formats`. Use-case definition fails without the matching spec.
5. Add any objective-specific plan warnings in `RecommendationEngine.recommend`, as the preference objective does.

The browser console's objective selector lists `supervised`, `preference` and `reinforcement` explicitly, so a new objective also needs a frontend change before it shows up there. The HTTP API and notebook accept it as soon as the backend does.

Objectives whose strategy is missing, or doesn't support the use case's task, are hidden from `/api/use-cases` automatically and reported under `unavailable_use_cases` in `/api/system`.

### If you add a task

A new `FinetuningTask` also needs entries in `_TASK_SLUG_TO_PEFT_TASK_TYPE` (`workflow/training.py`) for LoRA, and in `_TASK_SLUG_TO_AUTO_CLASS` and `_TASK_SLUG_TO_PIPELINE` (`workflow/jobs.py`) so the playground can reload the result.

## VRAM multipliers

The planner turns raw VRAM into an adjusted budget before computing any tier ceiling (`RecommendationEngine._compute_adjusted_vram`):

```text
adjusted = (raw VRAM − (batch_multiplier − 1) × 2.0 GB) / (1 + extra_model_copies)
```

| Strategy | `extra_model_copies` | `batch_multiplier` | Adjusted on 24 GB |
|---|---|---|---|
| `sft` | 0 | 1.0 | 24 GB |
| `qa` | 0 | 1.0 | 24 GB |
| `grpo` | 0 | 4.0 | 18 GB |
| `dpo` | 1 | 2.0 | 11 GB |

How to choose them:

- **`extra_model_copies`:** count the full models resident besides the one being trained, such as a frozen reference, a teacher or a reward model. Each copy gets a whole baseline training budget, so this is deliberately conservative. DPO uses 1 even though its adapter tiers don't load a second copy.
- **`batch_multiplier`:** how many sequences' worth of activations a step holds compared with SFT. DPO processes chosen and rejected (2.0), GRPO four sampled completions (4.0).
- If the result is negative on a GPU machine, `recommend` raises `Strategy '...' requires more VRAM than is available on this hardware for any viable model size.`

The ceilings then follow from the adjusted budget. See [Full fine-tuning](full-finetuning.md) and [LoRA and QLoRA](qlora.md).

## Tests to add

Tests live in `tests/unit_tests/` as `*_test.py` and run with:

```bash
uv run pytest
```

Use the existing files as templates:

| Test | Model it on | What to assert |
|---|---|---|
| Registration and attribute validation | `finetuning_strategy_test.py` | The slug is in `FinetuningStrategy._registry`. Use the `clean_strategy_registry` fixture pattern for throwaway subclasses. |
| Factory lookup | `strategy_factory_test.py` | `FinetuningStrategies.get_strategy_class("my_strategy") is MyStrategy`. |
| Data contract | `grpo_test.py`, `sft_test.py` | `validate_data_format` accepts good columns and raises `ValueError` with your message for missing ones. |
| Recommended config | `grpo_test.py`, `qa_test.py` | For each tier and a few VRAM sizes (`HardwareDetector.HardwareSpecs` with fake `gpus`): quantization kwargs, no quantization at 0 GB, `None` adapter for `full_finetune`, and constraints such as divisibility. |
| Resolution | `grpo_test.py`, `recommendation_engine_test.py` | `RecommendationEngine._resolve_use_case_and_strategy(use_case, objective)` returns your class and the expected task. |
| Launch keys | `fsdp_config_test.py` | `set_config({"output_dir": ..., **config.model_dump(), **build_fsdp_config(), "report_to": ["tensorboard"], "logging_dir": ..., "logging_first_step": True})` doesn't raise. This catches the GRPO-style bug. |
| Training path | `dpo_training_test.py`, `seq2seq_training_test.py` | With a mocked provider and trainer: callbacks reach the Trainer, `resume_from_checkpoint` is forwarded, and the right save method is used per tier. |

## See also

- [Architecture](../contributing/architecture.md)
- [Adding a use case](../contributing/use-cases.md)
- [Custom providers](../providers/custom-providers.md)
- [Strategies overview](overview.md)
