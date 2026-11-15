# Group Relative Policy Optimization (GRPO)

GRPO is ModelForge's reinforcement objective. The model samples several answers per prompt, each answer is scored by a reward function, and training pushes the model towards the higher-scoring answers. This page covers the `grpo` strategy, its built-in exact-match reward, defaults, memory and limitations.

## What it does

For each prompt, `GRPOTrainer` (TRL) generates `num_generations` completions, which is 4 in ModelForge. It scores each one and normalizes the scores within the group, so a completion is rewarded for being better than its siblings rather than for an absolute score. The implementation is `finetuning_strategies/strategies/grpo.py`.

ModelForge uses `beta = 0.0`, so there is no KL penalty against a reference model and no reference copy is loaded.

### The reward

GRPO runs exactly one reward function, `exact_match_reward`:

- **1.0** when the completion, with surrounding whitespace stripped, equals the row's `answer`, also stripped.
- **0.0** otherwise, including when `answer` isn't a string.

For conversational completions the content of the last message is compared. The comparison is case-sensitive and character-exact: `"51."` doesn't match `"51"`, and `"Paris"` doesn't match `"paris"`.

!!! warning "No custom rewards"
    ModelForge doesn't accept reward functions from a dataset, a request or a config file, by design. There is no dynamic import, expression evaluation or callback path. The only way to change the reward is to edit `exact_match_reward` in the source.

## When to use it

- The task has one checkable answer: arithmetic, classification labels, short extractions, fixed-format outputs.
- You can say whether an answer is right but don't have, or don't want to imitate, a full worked response.
- The base model already gets some answers right. If every completion scores 0, every group has the same reward and there is nothing to learn from. Running [SFT](sft.md) first often helps.

Don't use it to reward general helpfulness, style or long-form quality. Exact match can't score those.

## Use cases

| Use case | Objective | Task |
|---|---|---|
| Instruction Following | Reinforcement | `causal_language_modeling` |
| Text Generation | Reinforcement | `causal_language_modeling` |

Calling `GRPO.finetune` with any other task raises `ValueError`.

## Data

| Use case | Required | Aliases |
|---|---|---|
| Instruction Following | `prompt`, `answer` | `instruction` → `prompt`, `ideal_answer` → `answer` |
| Text Generation | `prompt`, `answer` | `input_text` → `prompt`, `ideal_continuation` → `answer` |

```json
{"prompt": "What is 17 × 3? Reply with the number only.", "answer": "51"}
```

```json
{"instruction": "Classify the sentiment as positive or negative: 'The update broke everything.'", "ideal_answer": "negative"}
```

Write prompts that pin down the output format, as in "Reply with the number only." The model has to produce the answer and nothing else to earn a reward.

String prompts are converted to a single user message (`normalize_prompts`) and the tokenizer's chat template is applied. Tokenizers without one get the `User:` / `Assistant:` fallback template. See [Dataset formats](../configuration/dataset-formats.md).

!!! warning "No streaming datasets"
    `GRPO.validate_data_format` rejects an `IterableDataset` with `GRPO does not support streaming datasets; load a finite dataset instead.` TRL rejects them before the first step. Use a local file or a downloaded Hub dataset.

## Default hyperparameters

`GRPO.build_recommended_config` produces a `GRPORecommendedConfig`. VRAM is the total across the selected GPUs. CPU and MPS machines get the "under 12 GB" values.

| Setting | Under 12 GB | 12 to under 32 GB | 32 GB or more |
|---|---|---|---|
| `max_completion_length` | 128 | 256 | 512 |
| `gradient_accumulation_steps` | 32 | 16 | 8 |

Fixed values:

| Setting | Value |
|---|---|
| `per_device_train_batch_size` | 1 |
| `num_generations` | 4 |
| `beta` | 0.0 |
| `gradient_checkpointing` | `true` |
| `use_liger_kernel` | `false` |
| `learning_rate` | `1e-6` for `full_finetune`, `5e-6` for adapter tiers |
| `bf16` / `fp16` | As for [SFT](sft.md#default-hyperparameters) |
| `num_train_epochs` | 3.0 |
| `weight_decay` | 0.0 |
| `warmup_ratio` | 0.1 |
| `lr_scheduler_type` | `"cosine"` |
| `optim` | `"adamw_torch"` |
| `max_grad_norm` | 1.0 |
| `logging_steps` / `save_steps` | 10 / 500 |
| `seed` | 42 |

LoRA for GRPO: `r=8` below 1B parameters, `r=16` from 1B, `lora_alpha = 2 × r`, `target_modules="all-linear"`.

TRL requires the generation batch (`per_device_train_batch_size × number of GPUs × gradient_accumulation_steps`) to be divisible by `num_generations`, and `num_generations` to be at least 2. Every recommended accumulation value is a multiple of 4, so the defaults satisfy this on any GPU count. If you edit the plan over the API, keep that rule or TRL raises at startup.

GRPO doesn't use packing or activation offloading. Sampling temperature and other generation settings are TRL's defaults.

## Memory

GRPO declares `extra_model_copies = 0` and `batch_multiplier = 4.0`:

```text
adjusted VRAM = raw VRAM − (4.0 − 1) × 2.0 GB = raw VRAM − 6 GB
```

On one 24 GB GPU that leaves 18 GB, so full fine-tuning fits models up to about (18 − 2) ÷ 18 ≈ 0.9B parameters. The cost is the four sampled completions held during each step, not a second model.

Generation is also slow compared to a forward pass. Expect GRPO steps to take noticeably longer than SFT steps on the same model.

## Outputs

Same layout as SFT: full weights for `full_finetune`, the adapter only for LoRA and QLoRA, plus the tokenizer, checkpoints and TensorBoard logs. See [SFT outputs](sft.md#outputs). TRL logs the mean reward, which is the clearest signal of whether training is working.

## Limitations

- Exact-match reward only. No partial credit, no normalization beyond stripping whitespace, no custom reward functions.
- No streaming datasets.
- No reference model (`beta = 0`), so nothing explicitly stops drift from the base model's behaviour on unrelated prompts.
- Liger kernels are not used.
- No evaluation split is used.

## See also

- [SFT](sft.md) to warm up a model before GRPO
- [DPO](dpo.md) when answers can be compared but not checked
- [Memory optimizations](memory-optimizations.md)
- [Dataset formats](../configuration/dataset-formats.md)
