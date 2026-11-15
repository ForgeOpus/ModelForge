# Training config

This page lists every key ModelForge puts in a recommendation's `strategy_config` and `peft_config`, with the defaults and the values that scale with VRAM, plus the FSDP settings applied at launch. It ends with how to send your own values through `POST /api/runs` and when they are used as sent.

The console shows these values read-only in the plan's **Training config** panel. Changing them requires the [REST API](rest-api.md).

## Where the values come from

A recommendation is built in three steps:

1. The engine picks a tier (`full_finetune`, `lora_bf16`, `qlora_8bit` or `qlora_4bit`) and a model. See [How planning works](../configuration/configuration-guide.md).
2. The strategy's `build_recommended_config()` fills `strategy_config` and `peft_config` from the tier, the GPUs and the chosen model's parameter count.
3. At launch, ModelForge adds `output_dir`, TensorBoard and FSDP settings, drops keys whose value is `null`, and passes the rest to the trainer.

Which strategy builds the config depends on the use case and objective:

| Use case | Objective | Strategy | Trainer |
|---|---|---|---|
| Instruction following, Text generation | supervised | `sft` | TRL `SFTTrainer` |
| Instruction following, Text generation | preference | `dpo` | TRL `DPOTrainer` |
| Instruction following, Text generation | reinforcement | `grpo` | TRL `GRPOTrainer` |
| Summarization | supervised | `sft` | Transformers `Seq2SeqTrainer` |
| Extractive QA | supervised | `qa` | Transformers `Trainer` |

### What "VRAM" means here

The VRAM-scaled values below use the **total** VRAM of the GPUs the plan was made for (all detected GPUs, or the ones in `gpu_indices`). Two 16 GB cards count as 32 GB. The bands are:

- **under 12 GB**: below 12.0
- **12–32 GB**: 12.0 up to but not including 32.0
- **32 GB or more**

CPU-only machines and Apple Silicon get the "under 12 GB" values in these tables, except that activation offloading and Liger are off and quantization is removed.

## Values shared by every strategy

| Key | Value | Notes |
|---|---|---|
| `bf16` / `fp16` | by GPU | `bf16: true` when every planned GPU has compute capability 8.0 or newer (Ampere, Ada, Hopper and later). Otherwise `fp16: true`. Both `false` on CPU, on Apple MPS, or when PyTorch can't see CUDA. |
| `model_init_kwargs` | by tier | Passed to `from_pretrained` when the base model is loaded. See below. |

`model_init_kwargs` by tier:

=== "full_finetune, lora_bf16"

    ```json
    {}
    ```

=== "qlora_8bit"

    ```json
    {"quantization_config": {"load_in_8bit": true}}
    ```

=== "qlora_4bit"

    ```json
    {
      "quantization_config": {
        "load_in_4bit": true,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": true
      }
    }
    ```

The Hugging Face provider turns a `quantization_config` dict into a `BitsAndBytesConfig`. It also accepts the older top-level `load_in_4bit` / `load_in_8bit` keys from saved v3-era plans and converts them, but rejects a config that sets both, or sets both a legacy flag and `quantization_config`. On CPU and Apple MPS, `model_init_kwargs` is always `{}` because bitsandbytes quantization needs CUDA.

## SFT

Used for supervised Instruction Following and Text Generation (causal LMs), and for Summarization (sequence-to-sequence).

| Key | Default | Notes |
|---|---|---|
| `per_device_train_batch_size` | `1` | Effective batch = this × `gradient_accumulation_steps` × number of GPUs. |
| `gradient_accumulation_steps` | by VRAM | 16 / 8 / 4. See the table below. |
| `gradient_checkpointing` | `true` | |
| `packing` | `true` | Causal LMs only. Ignored for summarization. |
| `packing_strategy` | `"bfd"` | Best-fit decreasing; keeps examples whole. |
| `max_length` | by VRAM | 512 / 1024 / 2048. For summarization it caps both the document and the summary. |
| `use_liger_kernel` | by install | `true` when a CUDA GPU is detected and `liger-kernel` is importable. Ignored for summarization. |
| `activation_offloading` | by VRAM | `true` under 12 GB on CUDA. Ignored for summarization. |
| `num_train_epochs` | `3.0` | |
| `learning_rate` | `2e-5` | |
| `per_device_eval_batch_size` | `8` | No evaluation split is used by the built-in flows. |
| `weight_decay` | `0.0` | |
| `warmup_ratio` | `0.0` | |
| `lr_scheduler_type` | `"linear"` | |
| `optim` | `"adamw_torch"` | |
| `max_grad_norm` | `1.0` | Gradient clipping. |
| `logging_steps` | `10` | How often a `log` event with the loss is emitted. |
| `save_steps` | `500` | Checkpoint interval. All checkpoints are kept. |
| `seed` | `42` | |
| `bf16`, `fp16`, `model_init_kwargs` | | See [shared values](#values-shared-by-every-strategy). |

## DPO

Used for preference training of Instruction Following and Text Generation.

| Key | Default | Notes |
|---|---|---|
| `per_device_train_batch_size` | `1` | |
| `gradient_accumulation_steps` | by VRAM | 32 / 16 / 8. |
| `gradient_checkpointing` | `true` | |
| `max_length` | by VRAM | 512 / 1024 / 2048. |
| `use_liger_kernel` | by install | Same rule as SFT. |
| `activation_offloading` | by VRAM | `true` under 12 GB on CUDA. |
| `beta` | `0.1` | Strength of the pull toward the reference model. |
| `loss_type` | `"sigmoid"` | The standard DPO loss. |
| `learning_rate` | by tier and size | See below. |
| `num_train_epochs` | `3.0` | |
| `per_device_eval_batch_size` | `8` | |
| `weight_decay` | `0.0` | |
| `warmup_ratio` | `0.1` | |
| `lr_scheduler_type` | `"cosine"` | |
| `optim` | `"adamw_torch"` | |
| `max_grad_norm` | `1.0` | |
| `logging_steps` | `10` | |
| `save_steps` | `500` | |
| `seed` | `42` | |

DPO `learning_rate`:

| Tier / model size | `learning_rate` |
|---|---|
| `full_finetune` (any size) | `1e-7` |
| Adapter tiers, under 1B parameters | `5e-6` |
| Adapter tiers, 1B to under 7B | `5e-7` |
| Adapter tiers, 7B to under 13B | `2e-7` |
| Adapter tiers, 13B and above | `1e-7` |

## GRPO

Used for reinforcement training of Instruction Following and Text Generation. The reward is fixed: 1.0 when the completion, trimmed of surrounding whitespace, exactly equals the row's `answer`, otherwise 0.0. Custom reward functions are not supported.

| Key | Default | Notes |
|---|---|---|
| `per_device_train_batch_size` | `1` | |
| `gradient_accumulation_steps` | by VRAM | 32 / 16 / 8. |
| `gradient_checkpointing` | `true` | |
| `max_completion_length` | by VRAM | 128 / 256 / 512 tokens per sampled completion. |
| `num_generations` | `4` | Completions sampled per prompt. |
| `beta` | `0.0` | No KL penalty, so no reference model is loaded. |
| `use_liger_kernel` | `false` | Always off for GRPO. |
| `learning_rate` | `1e-6` full fine-tune, `5e-6` adapter tiers | |
| `num_train_epochs` | `3.0` | |
| `weight_decay` | `0.0` | |
| `warmup_ratio` | `0.1` | |
| `lr_scheduler_type` | `"cosine"` | |
| `optim` | `"adamw_torch"` | |
| `max_grad_norm` | `1.0` | |
| `logging_steps` | `10` | |
| `save_steps` | `500` | |
| `seed` | `42` | |

GRPO has no `max_length` or `activation_offloading` key. Prompt length is left to TRL's default.

## QA

Used for Extractive QA. The QA trainer reads only the keys in this table; any other key in `strategy_config` is ignored.

| Key | Under 12 GB | 12 GB or more |
|---|---|---|
| `per_device_train_batch_size` | `1` | `4` |
| `gradient_accumulation_steps` | `8` | `1` |
| `gradient_checkpointing` | `true` | `false` |
| `max_length` | `256` | `384` |
| `doc_stride` | `128` | `128` |
| `learning_rate` | `3e-5` | `3e-5` |
| `num_train_epochs` | `3.0` | `3.0` |
| `logging_steps` | `10` | `10` |
| `save_steps` | `500` | `500` |
| `bf16`, `fp16`, `model_init_kwargs` | see [shared values](#values-shared-by-every-strategy) | |

Contexts longer than `max_length` are split into overlapping windows that share `doc_stride` tokens. QA has two bands only; there is no 32 GB step.

## Values scaled by VRAM, side by side

| Key | Under 12 GB | 12–32 GB | 32 GB or more |
|---|---|---|---|
| `max_length` (SFT, DPO) | 512 | 1024 | 2048 |
| `activation_offloading` (SFT, DPO) | true | false | false |
| `gradient_accumulation_steps` (SFT) | 16 | 8 | 4 |
| `gradient_accumulation_steps` (DPO, GRPO) | 32 | 16 | 8 |
| `max_completion_length` (GRPO) | 128 | 256 | 512 |
| `max_length` (QA) | 256 | 384 | 384 |
| `per_device_train_batch_size` (QA) | 1 | 4 | 4 |
| `gradient_accumulation_steps` (QA) | 8 | 1 | 1 |
| `gradient_checkpointing` (QA) | true | false | false |

## peft_config

`peft_config` is `null` for `full_finetune`. For the adapter tiers it has three keys:

```json
{"r": 16, "lora_alpha": 32, "target_modules": "all-linear"}
```

| Key | Default | Notes |
|---|---|---|
| `r` | by strategy and model size | See below. |
| `lora_alpha` | 2 × `r` | The update is scaled by `lora_alpha / r`, so 2 at the defaults. |
| `target_modules` | `"all-linear"` | Every linear layer gets an adapter. PEFT resolves this for any architecture. |

`r` by model size:

| Strategy | Under 1B | 1B to under 7B | 7B to under 13B | 13B and above |
|---|---|---|---|---|
| SFT, DPO | 8 | 16 | 32 | 64 |
| GRPO | 8 | 16 | 16 | 16 |
| QA | 16 | 16 | 32 | 32 |

At launch ModelForge builds a PEFT `LoraConfig` from `r`, `lora_alpha` and `target_modules` only, with `task_type` set from the task (`CAUSAL_LM`, `SEQ_2_SEQ_LM`, `QUESTION_ANS` or `TOKEN_CLS`). Other keys you add to `peft_config`, such as `lora_dropout`, are ignored, and PEFT's own defaults apply.

## Keys added at launch

These are not in the recommendation. ModelForge sets them when the run starts:

| Key | Value | When |
|---|---|---|
| `output_dir` | the run's `output_dir` | Always. Overrides anything in `strategy_config`. |
| `report_to` | `["tensorboard"]` | TensorBoard enabled (the default). Otherwise the trainers default to `"none"`. |
| `logging_dir` | `<tensorboard_log_dir>/run-<id>` | TensorBoard enabled. |
| `logging_first_step` | `true` | TensorBoard enabled. |
| FSDP settings | see below | `parallelism_strategy` is `fsdp` and the run uses two or more GPUs. |

Keys whose value is `null` are dropped before the trainer is built, so the trainer's own default applies.

## FSDP settings

When the plan's `parallelism_strategy` is `fsdp` (picked automatically, or forced with `parallelism_strategy_override`) and the run launches on at least two GPUs, these overrides are merged into the trainer arguments:

```python
fsdp = "full_shard auto_wrap"
fsdp_config = {
    "min_num_params": 100_000_000,
    "backward_prefetch": "backward_pre",
    "state_dict_type": "FULL_STATE_DICT",
    "activation_checkpointing": True,
}
gradient_checkpointing = False   # FSDP checkpoints activations itself
use_liger_kernel = False
activation_offloading = False
```

Mixed precision comes from the existing `bf16` / `fp16` flags. The final model is gathered into a full state dict, so it loads like any other model. If the plan says `fsdp` but the run launches on one GPU (for example `parallelism_strategy_override: "fsdp"` with a single entry in `gpu_indices`), the FSDP keys are not applied and the run trains as a normal single-GPU job.

FSDP is only valid with `full_finetune`. Any request that combines `fsdp` with a LoRA or QLoRA tier is rejected with:

```text
FSDP parallelism is not compatible with PEFT approach 'lora_bf16'. FSDP sharding cannot operate over LoRA adapter parameters. To fix this: either set parallelism_strategy to 'data_parallel', or set peft_approach to 'full_finetune'.
```

See [Multi-GPU training](../strategies/multi-gpu.md) for when the planner picks FSDP.

## How keys reach the trainer

| Strategy / task | What happens to `strategy_config` |
|---|---|
| SFT, causal LM | Every key is passed to TRL's `SFTConfig`. Any `SFTConfig` or `TrainingArguments` field can be added. |
| SFT, summarization | Every key is first passed to `SFTConfig` (so names must still be valid there), then only these are copied into `Seq2SeqTrainingArguments`: `num_train_epochs`, `per_device_train_batch_size`, `per_device_eval_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `weight_decay`, `warmup_ratio`, `warmup_steps`, `lr_scheduler_type`, `optim`, `logging_steps`, `save_steps`, `gradient_checkpointing`, `bf16`, `fp16`, `report_to`, `logging_dir`, `seed`, `dataloader_num_workers`, `max_grad_norm`, `fsdp`, `fsdp_config`. `max_length` sets tokenization. |
| DPO | Every key is passed to TRL's `DPOConfig`. |
| GRPO | Every key is passed to TRL's `GRPOConfig`. |
| QA | Only the keys in the [QA table](#qa) are read. Others are ignored. |

For SFT, DPO and GRPO, an unknown key is not caught when the run is created. The trainer config raises when the run starts, the run moves to `failed`, and the error reads like `SFTConfig.__init__() got an unexpected keyword argument 'my_key'`.

## Overriding the plan

`POST /api/runs` accepts a `recommendation` field. What happens depends on whether you send one and which model you ask for.

**No `recommendation`.** The server builds a fresh plan for `model_id`, exactly as `POST /api/recommendations` with `model_id` would. The model must be on the Hub, match the task's pipeline tag, publish a parameter count, and fit under the largest tier's ceiling.

**A `recommendation` is sent.** It must be a complete recommendation object (the shape `POST /api/recommendations` returns). The server then checks:

1. It parses as a recommendation. Missing required fields return `400`.
2. `parallelism_strategy_override`, if set, replaces `parallelism_strategy`. FSDP with an adapter tier returns `400`.
3. `task_slug` and `strategy_slug` must match the request's use case and objective: `Recommendation does not match the selected use case and objective.`
4. `provider_slug` must be supported by the strategy (only `huggingface` is): `Recommendation provider is not supported by this strategy.`
5. `peft_approach` must be one of the four tiers: `Unknown fine-tuning approach.`

Then:

!!! note "When an edited recommendation is used as sent"
    If `model_id` equals `recommendation.recommended_models[0].model_id`, the recommendation is used exactly as sent: your `strategy_config`, `peft_config`, `peft_approach` and `parallelism_strategy` all take effect.

    If `model_id` is anything else, your edits are discarded. The server finds that model in `recommended_models` (or validates it on the Hub if it isn't listed) and builds a new plan for it from the current hardware. `parallelism_strategy_override` is still applied to the rebuilt plan.

To keep edits while training a model other than the top candidate, reorder `recommended_models` so your model is first, or replace the list with a single entry for it.

The recommendation actually used is stored on the run as `config_json`. Resuming a run replays that stored plan with the same model, so edits survive a resume.

### Example: shorter sequences and fewer epochs

```bash
curl -s -H "X-ModelForge-Token: $TOKEN" -H "Content-Type: application/json" \
     -d '{"use_case_slug":"instruction_following","objective":"supervised"}' \
     $MF/api/recommendations \
  | jq '.strategy_config.max_length = 512
        | .strategy_config.num_train_epochs = 1
        | .strategy_config.save_steps = 200' > plan.json

jq -n --slurpfile plan plan.json '{
  use_case_slug: "instruction_following",
  objective: "supervised",
  model_id: $plan[0].recommended_models[0].model_id,
  dataset_path: "./data/train.jsonl",
  output_dir: "./runs/short-01",
  recommendation: $plan[0]
}' | curl -H "X-ModelForge-Token: $TOKEN" -H "Content-Type: application/json" -d @- $MF/api/runs
```

### Example: force QLoRA 4-bit on a model that would fit in LoRA bf16

Change the tier and add the matching quantization settings. Both are needed: `peft_approach` decides how the result is saved and whether an adapter is attached, `model_init_kwargs` decides how the base model is loaded.

```bash
jq '.peft_approach = "qlora_4bit"
    | .strategy_config.model_init_kwargs = {
        "quantization_config": {
          "load_in_4bit": true,
          "bnb_4bit_quant_type": "nf4",
          "bnb_4bit_use_double_quant": true
        }
      }
    | .peft_config = (.peft_config // {"r": 16, "lora_alpha": 32, "target_modules": "all-linear"})' \
  plan.json > plan-4bit.json
```

ModelForge does not re-check memory for an edited plan. If your values need more VRAM than the plan allowed, the run fails with an out-of-memory error.

## See also

- [REST API: POST /api/runs](rest-api.md#post-apiruns)
- [Response shapes: recommendation](responses.md#recommendation)
- [Memory optimizations](../strategies/memory-optimizations.md): gradient checkpointing, packing, Liger, activation offloading.
- [LoRA and QLoRA](../strategies/qlora.md)
- [Performance and out-of-memory](../troubleshooting/performance.md)
