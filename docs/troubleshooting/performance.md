# Performance and out-of-memory

This page explains how to make ModelForge runs faster and what to do when a run, or the Playground, runs out of memory. ModelForge's plan is tuned to fit first and be fast second, so most speed gains come from giving back some of that safety margin through an edited recommendation.

Most changes here need the [REST API](../api-reference/rest-api.md), because the console shows the training config read-only. See [Training config: overriding the plan](../api-reference/training-config.md#overriding-the-plan) for how to send edited values.

## What the plan optimizes for

Every plan uses `per_device_train_batch_size: 1` with gradient accumulation to reach a useful effective batch, gradient checkpointing on (except QA on 12 GB or more), and a fixed 2 GB activation budget. That's the configuration least likely to run out of memory, and it's also slower than what many GPUs can manage.

| Plan setting | Memory effect | Speed cost |
|---|---|---|
| Batch size 1 with accumulation (4 to 32 steps) | Activations for one sequence at a time | GPU underused on large cards |
| Gradient checkpointing | Stores a fraction of activations | Roughly 20 to 30% slower |
| Activation offloading (SFT, DPO under 12 GB) | Moves saved activations to CPU RAM | Extra transfers every step |
| QLoRA 8-bit / 4-bit | 1 or 0.5 bytes per frozen weight | Dequantization on every forward pass |

## Making training faster

### Check the basics first

- **Is it actually on the GPU?** If the plan shows `bf16: false` and `fp16: false` on an NVIDIA machine, PyTorch can't see CUDA and the run is on the CPU. See [Common issues](common-issues.md#a-gpu-is-listed-but-training-runs-on-the-cpu-or-precision-is-fp32).
- **Is something else using the GPU?** Check `nvidia-smi` during a run. A desktop session, notebook or second job slows training as well as risking OOM.
- **Is the first run slow because of downloads?** The base model is downloaded to the Hugging Face cache on first use. Later runs reuse it.
- **Precision.** bf16 is chosen automatically on compute capability 8.0 and newer (Ampere, Ada, Hopper). Older GPUs use fp16, which is fine for speed but can overflow; see [below](#loss-becomes-nan-on-older-gpus).

### Install Liger kernels (Linux)

```bash
pip install "modelforge-finetuning[gpu]"
```

Liger replaces common layers (RMSNorm, RoPE, SwiGLU, cross-entropy) with fused Triton kernels that use less memory and are often a little faster. SFT and DPO turn it on automatically when it's installed and a CUDA GPU is present. GRPO, QA and Summarization don't use it, and it's off under FSDP. It needs Triton, so it's Linux-only. See [Memory optimizations](../strategies/memory-optimizations.md).

### Train fewer, fuller steps

| Change | How | Notes |
|---|---|---|
| Fewer epochs | `strategy_config.num_train_epochs` (default 3.0) | Often the biggest win. Watch the loss: if it flattens after one epoch, stop there. |
| Larger per-device batch, less accumulation | Raise `per_device_train_batch_size`, lower `gradient_accumulation_steps` by the same factor | Keeps the effective batch the same and uses the GPU better. Costs activation memory: try 2 or 4 and check `nvidia-smi`. |
| Shorter sequences | Lower `max_length` (SFT, DPO, QA) or `max_completion_length` (GRPO) | Cost grows faster than linearly with length. Only if your examples are shorter anyway, or truncation is acceptable. |
| Gradient checkpointing off | `gradient_checkpointing: false` | Roughly 20 to 30% faster, but activations for every layer stay in memory. Only with plenty of headroom. |
| Activation offloading off | `activation_offloading: false` (SFT, DPO) | Only under 12 GB is it on; turning it off there often causes OOM. |
| Fewer GRPO samples | `num_generations` (default 4) | Fewer completions per prompt per step. Must stay at least 2, and TRL requires the effective batch (`per_device_train_batch_size` × `gradient_accumulation_steps` × GPUs) to be divisible by it. Lower `max_completion_length` too if answers are short. |
| Data loader workers | Add `dataloader_num_workers: 2` (SFT, DPO, GRPO, Summarization) | Helps when tokenization or loading is the bottleneck, which shows as low GPU utilization. |

### Packing

Supervised Instruction Following and Text Generation runs pack several short examples into each row (`packing: true`, `packing_strategy: "bfd"`), so little compute is spent on padding. Leave it on. DPO, GRPO, Summarization and QA don't pack.

### Use more GPUs (Linux)

With two or more GPUs, runs launch one process per GPU. Data parallel (the default) divides the steps per epoch across GPUs, which is close to a linear speed-up when the model fits on each card. FSDP is chosen only when a full fine-tune doesn't fit on one GPU; it adds communication every layer and is about capacity, not speed. See [Multi-GPU training](../strategies/multi-gpu.md).

### Save less often, keep fewer checkpoints

Checkpoints are written every `save_steps` (500) and all of them are kept. Each one includes optimizer state, which for full fine-tuning is several times the model size. On slow disks, raise `save_steps`. For SFT, DPO and GRPO you can also add `save_total_limit` to `strategy_config` to keep only the most recent few. Resuming needs at least one checkpoint, so don't set it to 0. ModelForge still lists every checkpoint it saw saved, so with a limit some listed checkpoints point at folders the trainer has since deleted.

## Fixing out-of-memory errors

The plan is an estimate, not a guarantee. When a run fails with `CUDA out of memory` (or `torch.OutOfMemoryError`), the run's `error` shows the message. ModelForge doesn't retry automatically.

### Common causes

- **Something else is using the GPU.** The planner counts total VRAM, not free VRAM. A desktop session, a notebook, another training job, or a Playground streaming session in another server all take from the same budget.
- **Long examples.** The 2 GB activation budget assumes rows up to `max_length` at batch size 1. Large models near the ceiling with long rows can exceed it.
- **DPO and GRPO.** DPO processes chosen and rejected together and, for full fine-tuning, holds a frozen reference copy. GRPO generates four completions per prompt. Both are more sensitive than SFT.
- **Uneven GPUs.** Under data parallel every GPU holds the whole model, including the smallest one. The planner accounts for the smallest card for adapter tiers, but other processes on that card still count.
- **Edited plans.** ModelForge doesn't re-check memory for an edited recommendation. Larger batches, longer sequences or a higher LoRA rank can push it over.

### What to change, in order

| Fix | How |
|---|---|
| Free the GPU | Close other GPU processes, then check with `nvidia-smi`. Close any open Playground session. |
| Pick a smaller model | Choose a lower row in the recommendation table, or type a smaller model ID. |
| Install Liger kernels | `pip install "modelforge-finetuning[gpu]"` on Linux, then generate the plan again. Cuts activation memory substantially for SFT and DPO. |
| Shorten examples | Trim or split very long rows in your dataset, or lower `max_length` through the API. |
| Turn on memory savers | Through the API: `activation_offloading: true` (SFT, DPO), `gradient_checkpointing: true` (QA on 12 GB or more has it off). |
| Use a lower tier | Send a recommendation with `peft_approach` set to `qlora_8bit` or `qlora_4bit` and the matching `model_init_kwargs`. See the [example](../api-reference/training-config.md#example-force-qlora-4-bit-on-a-model-that-would-fit-in-lora-bf16). CUDA only. |
| Lower the LoRA rank | `peft_config.r` (and `lora_alpha` to 2 × r). Small effect; adapters are a minor part of memory. |
| Add GPUs | With two or more, large full fine-tunes switch to FSDP automatically. |
| Resume | If it failed partway through, fix the cause and [resume from the last checkpoint](../user-guide/checkpoints.md). |

### Reading the plan before you start

The plan's warnings flag tight situations:

- `No single-GPU tier is viable for this hardware. Even QLoRA 4-bit may OOM. Consider multi-GPU setup.`
- `QLoRA 4-bit was selected because the chosen model exceeds the unquantized and 8-bit training budgets.` The model is near the top of what fits.
- `Available disk space (...) may be insufficient ...` is about disk, not VRAM, but a full disk fails a run just as surely.

`vram_estimate.max_params_by_tier` shows how close the chosen model is to its tier's ceiling. A model at 95% of the ceiling has little margin. See [Hardware and memory](../configuration/hardware-profiles.md) for the budget arithmetic.

### Loss becomes NaN on older GPUs

On GPUs older than Ampere the plan uses fp16, which has a narrow range. If the logged loss becomes `nan` or the gradient norm explodes, lower `learning_rate`, or set `fp16: false` to train in full precision (slower, more memory).

## Out of memory in the Playground

The Playground has two paths, and they behave differently.

### One-shot replies (`POST /api/playground/infer`)

These check memory before loading and refuse rather than crash. Weights are loaded in **float32**, so a model needs about 4 bytes per parameter across GPU, RAM and disk, plus room for the context. Budgets:

- **CPU RAM:** keeps back the larger of 2 GB and a quarter of available RAM when idle, and the larger of 8 GB and half of available RAM while a run is training.
- **GPU:** only used when no run is training. Each GPU keeps the larger of 1 GB and 10% of its memory free.
- **Disk:** used last, in a temporary folder deleted afterwards. Needs free space of twice the float32 model size plus 1 GB.

| Refusal | What to do |
|---|---|
| `Not enough available RAM for safe inference.` (or `... while training is active.`) | Close other programs, wait for training to finish, or use a smaller model. |
| `Not enough memory to execute the model, even with disk offload.` | Choose a smaller context window, shorten the conversation, or free RAM. |
| `Not enough disk space for safe model offload.` | Point `MODELFORGE_OFFLOAD_DIR` at a drive with more space. |
| `Cannot safely use a GPU while training is active.` | Expected during training; the reply runs on CPU and disk instead. |
| `This conversation needs N tokens, exceeding the M-token limit...` | Start a new chat or choose a larger context window. |

```bash
export MODELFORGE_OFFLOAD_DIR=/mnt/fast-ssd/mf-offload   # must exist
modelforge run
```

Disk offload is slow: a reply can take minutes when most of the model is on disk. CPU inference also competes with a CPU-bound training run.

### Streaming sessions

Streaming sessions load the model once with `device_map="auto"` in its saved precision and keep it loaded until closed. They aren't budgeted the way one-shot replies are, so a model too large for the GPU spills to CPU through Accelerate, and one that doesn't fit at all fails with `Could not load this model: ...`. They refuse to open while a run is training. Close the session when you're done: an open session holds GPU memory and blocks new training runs.

## See also

- [Memory optimizations](../strategies/memory-optimizations.md)
- [Hardware and memory](../configuration/hardware-profiles.md)
- [Training config](../api-reference/training-config.md)
- [Multi-GPU training](../strategies/multi-gpu.md)
- [Common issues](common-issues.md)
