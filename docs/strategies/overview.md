# Strategies overview

A training plan in ModelForge combines four independent choices: a use case, an objective, a PEFT tier and a parallelism strategy. This page explains what each one is, how ModelForge resolves them into a single run, and which combinations exist.

## The four layers

| Layer | What it decides | Values | Chosen by |
|---|---|---|---|
| **Use case** | The goal, the Hub `pipeline_tag` searched, the model head (task) and the dataset columns | `instruction_following`, `text_generation`, `summarization`, `extractive_qa` | You |
| **Objective** | How the model learns, and therefore which strategy (trainer) runs | `supervised`, `preference`, `reinforcement` | You, from the objectives the use case supports |
| **PEFT tier** | Whether every weight is trained, or a LoRA adapter on bf16, 8-bit or 4-bit base weights | `full_finetune`, `lora_bf16`, `qlora_8bit`, `qlora_4bit` | The recommendation engine, from your hardware and the chosen model |
| **Parallelism** | How work is split over several GPUs | `data_parallel`, `fsdp` | The recommendation engine; can be overridden over the HTTP API |

The layers are orthogonal. A DPO run can be full fine-tuning or QLoRA. An SFT run can be data parallel or FSDP. LoRA and QLoRA are **not** strategies in v4: they are tiers that any strategy runs with. See [LoRA and QLoRA](qlora.md).

```text
use case ──► task (model head) ─┐
                                ├─► strategy (trainer) ─► PEFT tier ─► parallelism ─► run
objective ──────────────────────┘
```

## Registered strategies

A strategy is a subclass of `FinetuningStrategy` in `finetuning_strategies/strategies/`. Five classes ship with v4. Four are registered strategies, and one internal route handles summarization:

| Slug | Name | Trainer | Tasks | Extra model copies | Batch multiplier |
|---|---|---|---|---|---|
| `sft` | Supervised Fine-Tuning | TRL `SFTTrainer` (causal LM) or Transformers `Seq2SeqTrainer` (seq2seq) | `causal_language_modeling`, `seq2seq_modeling` | 0 | 1.0 |
| `dpo` | Direct Preference Optimization | TRL `DPOTrainer` | `causal_language_modeling` | 1 | 2.0 |
| `grpo` | Group Relative Policy Optimization | TRL `GRPOTrainer` | `causal_language_modeling` | 0 | 4.0 |
| `qa` | Extractive Question Answering | Transformers `Trainer` | `question_answering` | 0 | 1.0 |

`extra_model_copies` and `batch_multiplier` feed the memory planner. See [Hardware and memory](../configuration/hardware-profiles.md) and [Custom strategies](custom-strategies.md#vram-multipliers).

The `token_classification` task is registered, but no use case or strategy uses it in v4.

## Which use case supports which strategy

| Use case | Task | Supervised | Preference | Reinforcement |
|---|---|---|---|---|
| Instruction Following | `causal_language_modeling` | `sft` ([SFT](sft.md)) | `dpo` ([DPO](dpo.md)) | `grpo` ([GRPO](grpo.md)) |
| Text Generation | `causal_language_modeling` | `sft` | `dpo` | `grpo` |
| Summarization | `seq2seq_modeling` | `sft`, routed to `Seq2SeqTrainer` | — | — |
| Extractive QA | `question_answering` | `qa` | — | — |

Instruction Following and Text Generation train the same kind of model with the same trainers. They differ only in the column aliases they accept and, for Instruction Following, in a model search limited to chat or instruct checkpoints. Column names per cell are on [Dataset formats](../configuration/dataset-formats.md).

## How the strategy is selected

The engine (`RecommendationEngine._resolve_use_case_and_strategy` in `utilities/recommendation_engine.py`) resolves the strategy in five steps:

1. Look up the use case by slug. An unknown slug raises `ValueError`.
2. Check that the objective is in the use case's `compatible_objectives`. If it isn't, the error lists the valid ones.
3. Take the use case's `strategy_slug` attribute if it defines one. Otherwise map the objective through `OBJECTIVE_STRATEGY_MAP`:

    ```python
    OBJECTIVE_STRATEGY_MAP = {
        TrainingObjective.SUPERVISED:    "sft",
        TrainingObjective.PREFERENCE:    "dpo",
        TrainingObjective.REINFORCEMENT: "grpo",
    }
    ```

    Only Extractive QA sets `strategy_slug` (`"qa"`). Summarization uses the default `sft`.

4. Check that the strategy slug is registered.
5. Check that the use case's `task_slug` is in the strategy's `supported_tasks`. A mismatch is reported as a misconfiguration of the use case or strategy.

`/api/system` and `/api/use-cases` list a use case only with the objectives that pass steps 3 to 5. A use case with none is left out of both, and `/api/system` lists it under `unavailable_use_cases` with the reason. See [REST API](../api-reference/rest-api.md).

The PEFT tier and parallelism strategy are chosen after the model. See [How planning works](../configuration/configuration-guide.md), [Full fine-tuning](full-finetuning.md) and [Multi-GPU training](multi-gpu.md).

## Native trainers: summarization and extractive QA

Two use cases don't go through TRL.

### Summarization

Summarization uses the `sft` strategy with the `seq2seq_modeling` task (`AutoModelForSeq2SeqLM`, for example T5 or BART). `SFT.finetune` detects that task and hands off to `SFT._finetune_seq2seq`, which uses Transformers' `Seq2SeqTrainer` and `DataCollatorForSeq2Seq`. TRL's `SFTTrainer` is built for causal LMs and produces invalid labels for encoder-decoder models.

- **Data:** `document` and `summary`, renamed to `prompt` and `completion` before training. Both are required.
- **Tokenization:** the prompt and the completion are each truncated to `max_length` (default 1024 if unset).
- **Config:** the plan is an `SFTRecommendedConfig`, but only the options shared with `Seq2SeqTrainingArguments` are forwarded: epochs, batch sizes, gradient accumulation, learning rate, weight decay, warmup, scheduler, optimizer, logging and save steps, gradient checkpointing, bf16/fp16, seed, `max_grad_norm` and FSDP settings. `packing`, `use_liger_kernel` and `activation_offloading` appear in the plan but have no effect.
- **PEFT:** adapters are applied with `peft.get_peft_model` and task type `SEQ_2_SEQ_LM`.

```json
{"document": "The council met on Tuesday to discuss the new bus routes...", "summary": "The council debated new bus routes."}
```

### Extractive QA

Extractive QA uses the `qa` strategy (`ExtractiveQA` in `finetuning_strategies/strategies/qa.py`) with `AutoModelForQuestionAnswering` and Transformers' `Trainer`. The model predicts start and end token positions of the answer inside the context.

- **Data:** `context`, `question` and `answers`. `answers` is a SQuAD-style dict with parallel `text` and `answer_start` lists. Offsets count characters in `context`. Only the first answer in each row is used as the training label.
- **Windows:** passages longer than `max_length` are split into overlapping windows with a stride of `doc_stride` tokens. A window that doesn't contain the answer is labelled with the CLS token position.
- **Tokenizer:** must have a CLS token, otherwise training raises `Extractive QA requires a tokenizer with a CLS token.`
- **PEFT:** quantized models go through `prepare_model_for_kbit_training`, then `get_peft_model` with task type `QUESTION_ANS`.

```json
{"context": "The library opened in 1999 on Main Street.",
 "question": "When did the library open?",
 "answers": {"text": ["1999"], "answer_start": [22]}}
```

Recommended QA configuration (`ExtractiveQA.build_recommended_config`). VRAM is the total across the selected GPUs. CPU-only and Apple MPS machines get the "under 12 GB" values.

| Setting | Under 12 GB | 12 GB or more |
|---|---|---|
| `per_device_train_batch_size` | 1 | 4 |
| `gradient_accumulation_steps` | 8 | 1 |
| `gradient_checkpointing` | `true` | `false` |
| `max_length` | 256 | 384 |
| `doc_stride` | 128 | 128 |
| `learning_rate` | `3e-5` | `3e-5` |
| `num_train_epochs` | 3.0 | 3.0 |
| `logging_steps` / `save_steps` | 10 / 500 | 10 / 500 |

LoRA for QA uses `r=16, lora_alpha=32` below 7B parameters and `r=32, lora_alpha=64` from 7B, with `target_modules="all-linear"`. Liger kernels, packing and activation offloading are not used for QA. The QA trainer reads only the keys listed above plus `bf16`, `fp16`, `model_init_kwargs`, `report_to`, `logging_dir`, `fsdp` and `fsdp_config`. Other keys in an edited plan are ignored.

## Defaults at a glance

| | SFT (causal) | DPO | GRPO | QA |
|---|---|---|---|---|
| Learning rate | `2e-5` | `1e-7` to `5e-6` by size and tier | `1e-6` full, `5e-6` adapters | `3e-5` |
| Epochs | 3 | 3 | 3 | 3 |
| Per-device batch | 1 | 1 | 1 | 1 or 4 |
| Gradient accumulation | 16 / 8 / 4 | 32 / 16 / 8 | 32 / 16 / 8 | 8 or 1 |
| Sequence length | `max_length` 512 / 1024 / 2048 | `max_length` 512 / 1024 / 2048 | `max_completion_length` 128 / 256 / 512 | `max_length` 256 / 384 |
| Scheduler, warmup | linear, 0.0 | cosine, 0.1 | cosine, 0.1 | Transformers defaults |
| Gradient checkpointing | on | on | on | on below 12 GB |
| Packing | on (`bfd`) | — | — | — |

Three-value cells are for total VRAM under 12 GB, 12 to under 32 GB, and 32 GB or more. The details and the code they come from are on each strategy page and on [Memory optimizations](memory-optimizations.md).

## Behaviour shared by all strategies

- **One process per GPU.** Every run goes through `workflow/training.py::assemble_and_train`, which launches the strategy's `finetune` through Accelerate. See [Multi-GPU training](multi-gpu.md#how-runs-are-launched).
- **No evaluation split.** `assemble_and_train` doesn't pass an evaluation dataset, so runs report training loss only. When a dataset has several splits, the first one is used for training.
- **Checkpoints every 500 steps** (`save_steps=500`). No `save_total_limit` is set, so all checkpoints are kept. See [Checkpoints](../user-guide/checkpoints.md).
- **Precision.** `bf16` when every selected CUDA GPU has compute capability 8.0 or newer, `fp16` otherwise. CPU and MPS train in full precision (`utilities/precision.py`).
- **Editing the plan.** The console shows the training config read-only. Over the HTTP API you can send an edited `recommendation` to `POST /api/runs`. Its `strategy_config` keys are passed to the strategy's `set_config` as sent, as long as `model_id` is the recommendation's top candidate. Choosing a different model rebuilds the plan. See [Training config reference](../api-reference/training-config.md).
- **Provider.** Every strategy supports only the `huggingface` provider in v4. See [Providers](../providers/overview.md).

## See also

- [Supervised fine-tuning (SFT)](sft.md)
- [Direct Preference Optimization (DPO)](dpo.md)
- [Group Relative Policy Optimization (GRPO)](grpo.md)
- [Full fine-tuning](full-finetuning.md) and [LoRA and QLoRA](qlora.md)
- [Use cases and tasks](../configuration/training-tasks.md)
