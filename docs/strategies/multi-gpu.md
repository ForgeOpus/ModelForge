# Multi-GPU training

With more than one CUDA GPU, ModelForge trains with either data parallelism or Fully Sharded Data Parallel (FSDP). This page explains the difference, the exact rule the planner uses to choose, the FSDP configuration, how to select GPUs and override the choice, and how runs are launched.

## Data parallel vs FSDP

| | Data parallel | FSDP |
|---|---|---|
| **Each GPU holds** | The whole model, its gradients and optimizer state | A shard of each, gathered layer by layer |
| **Improves** | Throughput: more batches per second | Capacity: models no single GPU can hold |
| **Communication** | Gradients once per step | Weights and gradients for every wrapped unit |
| **Tiers** | All | `full_finetune` only |
| **In ModelForge** | Default for every multi-GPU run | Chosen when a full fine-tune doesn't fit on the smallest GPU |

The two values are the `ParallelismStrategy` enum in `utilities/parallelism_strategy.py`: `data_parallel` and `fsdp`. ModelForge doesn't support DeepSpeed, tensor parallelism, pipeline parallelism or multi-node training.

## When the planner picks FSDP

`RecommendationEngine._select_parallelism_strategy` returns `fsdp` only when **all** of these hold:

1. Two or more CUDA GPUs are selected. CPU and Apple MPS are always `data_parallel`.
2. The tier is `full_finetune`. Adapter tiers are always `data_parallel`.
3. A model was chosen.
4. The model's full fine-tuning budget exceeds the smallest selected GPU:

    ```text
    num_parameters × 18 bytes / 1e9 + 2.0 GB  >  min(per-GPU VRAM)
    ```

Otherwise it returns `data_parallel`. The smallest GPU is used, not the average, because under data parallel every GPU has to hold a full replica.

Why this works: when the planner sets the full fine-tune ceiling, it sums VRAM across the selected GPUs. So the chosen model can be larger than any single card, and FSDP is what makes that ceiling real.

**Example.** Two 24 GB GPUs, SFT. The full fine-tune ceiling is (48 − 2) ÷ 18 ≈ 2.5B parameters. A 1B model needs about 20 GB and fits on each card, so the plan is `data_parallel`. A 2B model needs about 38 GB, more than either card, so the plan is `fsdp`.

The plan explains the decision in `rationale.parallelism_decision`, for example `FSDP selected: 2 GPUs detected, smallest-GPU VRAM (24.0 GB) insufficient to hold a full-fine-tuning replica on a single GPU. Sharding across 2 GPUs.` A multi-GPU warning is also added to `rationale.warnings`. See [Responses](../api-reference/responses.md).

## FSDP configuration

`utilities/fsdp_config.py::build_fsdp_config` returns these `TrainingArguments` overrides:

```python
{
    "fsdp": "full_shard auto_wrap",
    "fsdp_config": {
        "min_num_params": 100_000_000,
        "backward_prefetch": "backward_pre",
        "state_dict_type": "FULL_STATE_DICT",
        "activation_checkpointing": True,
    },
    "gradient_checkpointing": False,
    "use_liger_kernel": False,
    "activation_offloading": False,
}
```

| Key | Why |
|---|---|
| `full_shard auto_wrap` | Shard parameters, gradients and optimizer state. Wrap submodules automatically. |
| `min_num_params: 100_000_000` | Size-based wrapping. A unit below 100M parameters isn't worth its own all-gather. It works across model families without naming the transformer block class. |
| `backward_prefetch: backward_pre` | Fetch the next unit's parameters before the current backward finishes. |
| `state_dict_type: FULL_STATE_DICT` | Gather a full state dict at save time, so the output loads like any other model. |
| `activation_checkpointing: True` | FSDP checkpoints activations itself. |
| `gradient_checkpointing: False` | Enabling both raises in the HF Trainer and adds a redundant all-gather. |
| `use_liger_kernel`, `activation_offloading: False` | Incompatible with FSDP parameter sharding. |

Mixed precision isn't set here. The Trainer derives it from the plan's `bf16` / `fp16` flags.

The overrides are merged into the strategy config in `workflow/training.py::assemble_and_train` before `strategy.set_config`. They go through the Trainer's own `fsdp` and `fsdp_config` arguments rather than a pre-built Accelerate plugin, because the Trainer builds its own `Accelerator` and would overwrite one. They are applied only when the recommendation says `fsdp` **and** the run has two or more processes. A stored plan replayed on one GPU, for example a resume with a different GPU set, silently runs as a single process.

## FSDP and PEFT

FSDP can't shard LoRA or QLoRA adapter parameters. `utilities/parallelism_validator.py::validate_parallelism_compatibility` rejects the combination:

```text
FSDP parallelism is not compatible with PEFT approach 'lora_bf16'. FSDP sharding cannot operate over
LoRA adapter parameters. To fix this: either set parallelism_strategy to 'data_parallel', or set
peft_approach to 'full_finetune'.
```

The check runs after every recommendation in `WorkflowService.recommend`, and again in `TrainingJobManager.start_run` after any override is applied. Over HTTP the error is returned as `400`, before any training thread starts.

## Selecting GPUs

| Surface | How GPUs are chosen |
|---|---|
| Browser console | Always every detected GPU. It sends `gpu_indices: []`. |
| HTTP API | `gpu_indices` on `POST /api/recommendations` and `POST /api/runs`, 0-based. Empty or omitted means all GPUs. Out-of-range indices raise `GPU indices must be between 0 and N.` |
| Terminal wizard | Prompts `Select GPUs (e.g. 1,2 or 'all')` when there are two or more, 1-based. One GPU is selected automatically. |
| Notebook | Always all detected GPUs. |

Pass the same `gpu_indices` to both calls. The recommendation is planned against the selected GPUs only, and the run trains on them. The selection is stored with the run, and resuming from a checkpoint reuses it.

```bash
curl -X POST http://127.0.0.1:8000/api/runs \
  -H "Content-Type: application/json" \
  -d '{"use_case_slug": "text_generation", "objective": "supervised",
       "model_id": "Qwen/Qwen2.5-0.5B", "dataset_path": "./data/train.jsonl",
       "output_dir": "./out", "gpu_indices": [0, 1]}'
```

Authentication headers are omitted here. See [REST API](../api-reference/rest-api.md).

## Overriding the strategy

`POST /api/runs` accepts `parallelism_strategy_override`, either `"data_parallel"` or `"fsdp"`. It replaces the plan's `parallelism_strategy` before validation, so `"fsdp"` with an adapter tier is rejected as above. The console, terminal and notebook don't expose it.

```json
{"use_case_slug": "text_generation", "objective": "supervised", "model_id": "...",
 "dataset_path": "./data/train.jsonl", "output_dir": "./out",
 "gpu_indices": [0, 1], "parallelism_strategy_override": "data_parallel"}
```

Forcing `data_parallel` for a model that needed FSDP will most likely run out of memory. Forcing `fsdp` on a single GPU has no effect.

## How runs are launched

Every surface ends in `assemble_and_train`:

1. Set `CUDA_VISIBLE_DEVICES` to the selected indices, for example `0,1`.
2. Count the processes: one per entry in that list, or 1 when there is no GPU.
3. Merge the FSDP overrides if they apply, then call `strategy.set_config`.
4. Call Accelerate's `notebook_launcher(..., num_processes=N)`, which runs `strategy.finetune` once per GPU. With one process it runs in place. With more it forks worker processes.

The Hugging Face Trainer inside each process handles distributed sampling and gradient synchronisation. Effective batch size is `per_device_train_batch_size × gradient_accumulation_steps × N`.

In the browser and notebook, training runs in a background thread of the ModelForge server, and only one run can be active at a time. ModelForge doesn't use `accelerate launch` or `torchrun`, and there is no Accelerate config file to edit.

## See also

- [Full fine-tuning](full-finetuning.md)
- [Memory optimizations](memory-optimizations.md)
- [Hardware and memory](../configuration/hardware-profiles.md)
- [REST API](../api-reference/rest-api.md)
