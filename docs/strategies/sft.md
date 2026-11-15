# Supervised fine-tuning (SFT)

Supervised fine-tuning trains a model to produce a target output for each input. This page covers the `sft` strategy for causal language models: the data it reads, the defaults the planner sets, memory use, outputs and limitations. Summarization also runs under `sft`, through a separate encoder-decoder path described in the [overview](overview.md#summarization).

## What it does

Each row pairs a prompt with the completion you want. Training raises the probability of the completion tokens given the prompt. ModelForge runs TRL's `SFTTrainer` (`finetuning_strategies/strategies/sft.py`).

For prompt/completion data, TRL computes the loss on the completion tokens only, which is TRL's default (`completion_only_loss=None`). The prompt is context, not a target.

## When to use it

- Teaching a format, a task, a tone or a domain vocabulary.
- As the first stage before [DPO](dpo.md). DPO aligns behaviour the model already has. It doesn't teach a base model to follow instructions.
- Whenever you can write the ideal answer for each input.

Use [GRPO](grpo.md) instead when you can check whether an answer is right, but can't or don't want to write full target outputs.

## Use cases

| Use case | Objective | Task | Trainer |
|---|---|---|---|
| Instruction Following | Supervised | `causal_language_modeling` | `SFTTrainer` |
| Text Generation | Supervised | `causal_language_modeling` | `SFTTrainer` |
| Summarization | Supervised | `seq2seq_modeling` | `Seq2SeqTrainer` ([details](overview.md#summarization)) |

## Data

Both causal use cases need `prompt` and `completion` columns.

| Use case | Canonical columns | Accepted aliases |
|---|---|---|
| Instruction Following | `prompt`, `completion` | `instruction` → `prompt`, `response` → `completion` |
| Text Generation | `prompt`, `completion` | `input_text` → `prompt`, `generated_text` → `completion` |

Aliases are renamed before validation and training. A dataset with both an alias and its canonical column (for example `instruction` and `prompt`) is rejected as ambiguous. Full rules are on [Dataset formats](../configuration/dataset-formats.md).

Example row (Instruction Following):

```json
{"prompt": "Summarise this ticket: User reports login fails after the 2.3 update.", "completion": "Customer cannot log in after the 2.3 update."}
```

Text Generation, using aliases:

```json
{"input_text": "The lighthouse keeper climbed the stairs", "generated_text": " one last time, lantern swinging in the wind."}
```

`prompt` and `completion` may also be conversations, that is lists of `{"role": ..., "content": ...}` messages.

### Chat formatting

Before training, string prompts and completions are converted to messages (`utilities/chat_format.py::normalize_supervised`):

```python
{"prompt": [{"role": "user", "content": prompt}],
 "completion": [{"role": "assistant", "content": completion}]}
```

The trainer then applies the tokenizer's chat template. If the tokenizer has none, ModelForge sets a plain fallback template that writes `User: ...` and `Assistant: ...` turns and ends assistant turns with the tokenizer's existing EOS token. No tokens are added to the vocabulary. The model's own template is kept when it has one.

!!! note "Other SFT formats"
    `SFT.validate_data_format` also accepts a `messages` column or a raw `text` column. Those formats are only reachable when calling the strategy directly. The use cases require `prompt` and `completion` (or their aliases), so a `text`-only file is rejected with `Dataset is missing required column(s)`.

## Default hyperparameters

`SFT.build_recommended_config` produces an `SFTRecommendedConfig`. VRAM below is the total across the selected GPUs. CPU-only and Apple MPS machines get the "under 12 GB" column without activation offloading.

| Setting | Under 12 GB | 12 to under 32 GB | 32 GB or more |
|---|---|---|---|
| `per_device_train_batch_size` | 1 | 1 | 1 |
| `gradient_accumulation_steps` | 16 | 8 | 4 |
| `max_length` | 512 | 1024 | 2048 |
| `activation_offloading` | `true` (GPU only) | `false` | `false` |

Fixed for every hardware profile:

| Setting | Value |
|---|---|
| `gradient_checkpointing` | `true` |
| `packing` | `true` |
| `packing_strategy` | `"bfd"` |
| `use_liger_kernel` | `true` when a CUDA GPU is present and `liger-kernel` is installed |
| `bf16` / `fp16` | `bf16` if every GPU has compute capability 8.0+, else `fp16`. Both `false` on CPU and MPS |
| `model_init_kwargs` | Set by the PEFT tier. See [LoRA and QLoRA](qlora.md#quantization-settings) |
| `num_train_epochs` | 3.0 |
| `learning_rate` | `2e-5` |
| `per_device_eval_batch_size` | 8 |
| `weight_decay` | 0.0 |
| `warmup_ratio` | 0.0 |
| `lr_scheduler_type` | `"linear"` |
| `optim` | `"adamw_torch"` |
| `max_grad_norm` | 1.0 |
| `logging_steps` | 10 |
| `save_steps` | 500 |
| `seed` | 42 |

Effective batch size is `per_device_train_batch_size × gradient_accumulation_steps × number of GPUs`. On one 24 GB GPU that's 1 × 8 × 1 = 8 sequences per optimizer step. With packing, each sequence can hold several examples.

Options not surfaced in the plan keep the values of `SFT.build_sft_config`: `truncation_mode="keep_start"`, `loss_type="nll"`, `assistant_only_loss=False`, `padding_free=False`, `report_to="none"` (switched to TensorBoard when logging is on).

### LoRA settings

When the plan picks an adapter tier, the `peft_config` is:

| Model size | `r` | `lora_alpha` | `target_modules` |
|---|---|---|---|
| under 1B | 8 | 16 | `"all-linear"` |
| 1B to under 7B | 16 | 32 | `"all-linear"` |
| 7B to under 13B | 32 | 64 | `"all-linear"` |
| 13B or more | 64 | 128 | `"all-linear"` |

See [LoRA and QLoRA](qlora.md) for how the tier is chosen.

## Memory

SFT is the baseline for the planner: `extra_model_copies = 0`, `batch_multiplier = 1.0`. Adjusted VRAM equals raw VRAM. On a single 24 GB GPU, full fine-tuning fits models up to about (24 − 2) ÷ 18 ≈ 1.2B parameters. Larger models get a LoRA or QLoRA tier. See [Hardware and memory](../configuration/hardware-profiles.md).

Packing, gradient checkpointing, activation offloading and Liger are explained on [Memory optimizations](memory-optimizations.md).

## Outputs

Written to the run's output directory:

| Tier | Saved |
|---|---|
| `full_finetune` | Full model weights (`trainer.save_model`), loadable with `from_pretrained` |
| `lora_bf16`, `qlora_8bit`, `qlora_4bit` | Adapter only (`trainer.model.save_pretrained`). Load it with PEFT on the same base model |

Every run also saves the tokenizer, including the chat template used in training, `checkpoint-<step>` folders every 500 steps and, when enabled, TensorBoard logs under `tensorboard/run-<id>`. Runs started from the browser or notebook also write `modelforge_config.json`, which records the base model, task, strategy and PEFT tier so the [Playground](../user-guide/playground.md) can reload the result.

## Limitations

- No evaluation split is used. Runs report training loss only.
- Prompts and completions longer than `max_length` together are truncated from the end (`truncation_mode="keep_start"`).
- A chat template gives data a consistent structure. It doesn't give an untrained model language ability: start from a capable base or instruct model.
- The plan's `target_modules` is `"all-linear"` for every architecture. There is no per-module control in the console.

## See also

- [DPO](dpo.md), usually run after SFT
- [Memory optimizations](memory-optimizations.md)
- [Dataset formats](../configuration/dataset-formats.md)
- [Strategies overview](overview.md)
