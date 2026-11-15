# Direct Preference Optimization (DPO)

DPO trains a model to prefer one answer over another, using pairs of chosen and rejected responses instead of a reward model. This page covers the `dpo` strategy: data, including implicit-prompt pairs, defaults, memory, outputs and limitations.

## What it does

For each pair, DPO raises the likelihood of the chosen response relative to the rejected one, measured against a frozen reference copy of the starting model. `beta` controls how far the model may drift from that reference. ModelForge runs TRL's `DPOTrainer` (`finetuning_strategies/strategies/dpo.py`) with the sigmoid loss.

The reference model depends on the tier:

- **Full fine-tuning:** ModelForge loads a second copy of the base model, freezes it and passes it as `ref_model`.
- **LoRA and QLoRA:** no second copy. `DPOTrainer` uses the frozen base weights, with the adapter disabled, as the reference.

## When to use it

- The model already produces reasonable answers, and you want it to prefer some styles, levels of detail or safety behaviours over others.
- You can rank two responses more easily than you can write the ideal one.
- After [SFT](sft.md). DPO aligns existing behaviour. For instruction following, start from an instruct or chat checkpoint, or run SFT first. A chat template doesn't teach a base model to chat, and the plan includes a warning saying so.

## Use cases

| Use case | Objective | Task |
|---|---|---|
| Instruction Following | Preference | `causal_language_modeling` |
| Text Generation | Preference | `causal_language_modeling` |

## Data

| Use case | Required | Optional | Aliases |
|---|---|---|---|
| Instruction Following | `chosen`, `rejected` | `prompt` | `preferred_response` → `chosen`, `dispreferred_response` → `rejected`, `instruction` → `prompt` |
| Text Generation | `chosen`, `rejected` | `prompt` | `preferred_continuation` → `chosen`, `rejected_continuation` → `rejected`, `input_text` → `prompt` |

See [Dataset formats](../configuration/dataset-formats.md) for alias rules.

=== "Explicit prompt"

    ```json
    {"prompt": "Explain LoRA in one sentence.", "chosen": "LoRA trains two small low-rank matrices beside frozen weights.", "rejected": "It is a thing for models."}
    ```

=== "Implicit prompt (text)"

    ```json
    {"chosen": "Q: What is 2 + 2?\nA: 4", "rejected": "Q: What is 2 + 2?\nA: 5"}
    ```

=== "Implicit prompt (conversation)"

    ```json
    {"chosen":   [{"role": "user", "content": "What colour is the sky?"}, {"role": "assistant", "content": "Blue."}],
     "rejected": [{"role": "user", "content": "What colour is the sky?"}, {"role": "assistant", "content": "Green."}]}
    ```

=== "Assistant-only with prompt"

    ```json
    {"prompt": "What colour is the sky?",
     "chosen":   [{"role": "assistant", "content": "Blue."}],
     "rejected": [{"role": "assistant", "content": "Green."}]}
    ```

### How pairs are normalized

Every row, including evaluation rows, goes through `utilities/chat_format.py::normalize_preference` before TRL sees it. The result is always a shared `prompt` conversation plus assistant-only `chosen` and `rejected` turns.

| Input | What happens |
|---|---|
| Strings with a `prompt` | The prompt becomes a user turn and each response becomes an assistant turn. |
| Strings without a `prompt` (implicit) | TRL's `extract_prompt` takes the longest shared prefix of `chosen` and `rejected` as the prompt. In the example above the prompt is `"Q: What is 2 + 2?\nA:"`. If there is no shared prefix, the row is rejected: `Text preference pairs need a shared prompt. Add a prompt column.` |
| Full conversations | The shared leading turns become the prompt. The prompt must end with a `user` or `tool` turn. An explicit `prompt` column that disagrees with the conversations is rejected. |
| Assistant-only conversations | A `prompt` column is required. |
| One string and one conversation | Rejected: `Chosen and rejected must both be strings or both be conversations.` |

The tokenizer's chat template is then applied. Tokenizers without one get the same `User:` / `Assistant:` fallback template as SFT.

## Default hyperparameters

`DPO.build_recommended_config` produces a `DPORecommendedConfig`. VRAM is the total across the selected GPUs. CPU and MPS machines get the "under 12 GB" values.

| Setting | Under 12 GB | 12 to under 32 GB | 32 GB or more |
|---|---|---|---|
| `per_device_train_batch_size` | 1 | 1 | 1 |
| `gradient_accumulation_steps` | 32 | 16 | 8 |
| `max_length` | 512 | 1024 | 2048 |
| `activation_offloading` | `true` (GPU only) | `false` | `false` |

Gradient accumulation is twice SFT's at each size, to make up for the doubled activation cost of processing two responses.

Learning rate depends on tier and model size:

| Tier | Model size | `learning_rate` |
|---|---|---|
| `full_finetune` | any | `1e-7` |
| adapter tiers | under 1B | `5e-6` |
| adapter tiers | 1B to under 7B | `5e-7` |
| adapter tiers | 7B to under 13B | `2e-7` |
| adapter tiers | 13B or more | `1e-7` |

Fixed values:

| Setting | Value |
|---|---|
| `beta` | 0.1 |
| `loss_type` | `"sigmoid"` |
| `gradient_checkpointing` | `true` |
| `use_liger_kernel` | `true` when a CUDA GPU is present and `liger-kernel` is installed |
| `bf16` / `fp16` | As for [SFT](sft.md#default-hyperparameters) |
| `num_train_epochs` | 3.0 |
| `per_device_eval_batch_size` | 8 |
| `weight_decay` | 0.0 |
| `warmup_ratio` | 0.1 |
| `lr_scheduler_type` | `"cosine"` |
| `optim` | `"adamw_torch"` |
| `max_grad_norm` | 1.0 |
| `logging_steps` / `save_steps` | 10 / 500 |
| `seed` | 42 |

DPO doesn't pack. LoRA rank and alpha follow the same size table as [SFT](sft.md#lora-settings): `r` of 8, 16, 32 or 64 and `lora_alpha = 2 × r`, with `target_modules="all-linear"`.

## Memory

DPO declares `extra_model_copies = 1` and `batch_multiplier = 2.0`. The planner computes:

```text
adjusted VRAM = (raw VRAM − (2.0 − 1) × 2.0 GB) / (1 + 1)
```

On one 24 GB GPU that's (24 − 2) ÷ 2 = 11 GB, so full fine-tuning fits models up to about (11 − 2) ÷ 18 ≈ 0.5B parameters.

The extra copy is only loaded for full fine-tuning. Adapter tiers use the frozen base as the reference, so the planner over-estimates their memory slightly. That is a deliberate, conservative choice in the code.

## Outputs

Same layout as SFT: full weights for `full_finetune`, the adapter only for LoRA and QLoRA, plus the tokenizer, checkpoints every 500 steps and TensorBoard logs. See [SFT outputs](sft.md#outputs).

## Limitations

- Only the sigmoid loss is configured. Other TRL loss types are reachable only by editing `loss_type` in a recommendation sent to the HTTP API.
- No evaluation split is used, so TRL's reward accuracy and margin metrics are only logged for training batches.
- Pairs with no shared prompt, either explicit or as a common prefix, can't be trained.
- Small or random test models are excluded from recommendations because they can't produce meaningful preference behaviour.

## See also

- [SFT](sft.md), the usual first stage
- [GRPO](grpo.md), when an answer can be checked instead of compared
- [Full fine-tuning](full-finetuning.md) for the reference-model cost
- [Dataset formats](../configuration/dataset-formats.md)
