# Hardware and memory

ModelForge plans every run from the hardware it detects. It turns GPU memory into a budget, works out the largest model each training method can fit, and scales a few training settings to the memory available. This page explains what is detected, how the budget is calculated, and what changes on multi-GPU, Apple Silicon and CPU-only machines.

!!! info "No more hardware profiles"
    ModelForge 3 sorted machines into low, mid and high-end profiles with fixed settings per profile. ModelForge 4 has no profiles. The planner computes a parameter ceiling for each training method directly from your VRAM, so the plan changes smoothly with the hardware instead of jumping between three presets.

## What is detected

`utilities/hardware_detector.py` collects:

| Field | Source | Used for |
|---|---|---|
| `gpus` | NVML (NVIDIA driver), or PyTorch MPS | The VRAM budget, precision, parallelism |
| `ram_gb` | `psutil` total memory | CPU-only parameter ceiling |
| `physical_core_count` | `psutil` | Reported only |
| `disk_storage_gb`, `available_disk_storage_gb` | `psutil` usage of the filesystem root `/` | Low-disk warning |

See it with `GET /api/hardware`, or at the start of the terminal wizard:

```json
{
  "physical_core_count": 16,
  "disk_storage_gb": 1862.0,
  "available_disk_storage_gb": 940.5,
  "ram_gb": 62.7,
  "gpus": [{"name": "NVIDIA GeForce RTX 4090", "total_memory_gb": 23.99}]
}
```

Memory figures are in GiB (bytes ÷ 1024³).

### CUDA, MPS or CPU

| Machine | What is detected | Treated as |
|---|---|---|
| NVIDIA GPU(s) | One entry per GPU NVML reports, with its **total** memory | CUDA. VRAM is the sum of the selected GPUs. |
| Apple Silicon with MPS available | One entry, `Apple Silicon GPU`, memory `"N/A"` | MPS. Counted as 8 GB. NVML is not queried. |
| Anything else | No GPUs | CPU-only. |

Notes:

- Detection reads **total** GPU memory, not free memory. Memory used by other processes, a desktop session or a loaded playground model is not subtracted.
- Only NVIDIA GPUs are detected. If NVML is unavailable or fails, the machine is treated as CPU-only.
- NVML lists every NVIDIA GPU in the machine. To plan and train on a subset, pass `gpu_indices` to the API, or choose GPUs in the terminal wizard. The browser console always uses every detected GPU.

### Precision

Training precision comes from the GPUs that will run training (`utilities/precision.py`):

| Hardware | `bf16` | `fp16` |
|---|---|---|
| CUDA, every GPU with compute capability 8.0 or newer (Ampere, Ada, Hopper and later) | true | false |
| CUDA, any GPU older than 8.0 (for example Turing or Volta) | false | true |
| Apple MPS | false | false (fp32) |
| CPU-only | false | false (fp32) |

If PyTorch can't read a GPU's compute capability, fp16 is used.

## The memory model

Rather than simulate every allocation, the planner uses a fixed cost per parameter for each training method (**tier**) plus two constants:

| Constant | Value | Covers |
|---|---|---|
| Activation budget | 2.0 GB | Activations for one sequence at a time with gradient checkpointing on |
| Adapter cost | 0.1 GB | LoRA adapters, their optimizer state and QLoRA dequantization buffers |

| Tier | Bytes per parameter | What sits on the GPU |
|---|---|---|
| `full_finetune` | 18 | Weights, gradients and Adam optimizer state for every parameter |
| `lora_bf16` | 2 | Frozen 16-bit weights plus adapters |
| `qlora_8bit` | 1 | Frozen 8-bit weights plus adapters |
| `qlora_4bit` | 0.5 | Frozen 4-bit NF4 weights plus adapters |

### Raw VRAM

The sum of total memory across the selected GPUs. MPS is 8 GB. CPU-only is 0.

### Per-strategy adjustment

Some objectives hold more than one model or more than one sequence in memory. The planner converts raw VRAM into an **adjusted** budget first:

```text
adjusted = (raw − (batch_multiplier − 1) × 2.0 GB) / (1 + extra_model_copies)
```

| Strategy | Extra model copies | Batch multiplier | Why |
|---|---|---|---|
| `sft` | 0 | 1.0 | One model, one sequence. |
| `dpo` | 1 | 2.0 | A frozen reference model (counted for adapter tiers too, to stay conservative), and chosen and rejected answers in each step. |
| `grpo` | 0 | 4.0 | Four sampled completions per prompt. |
| `qa` | 0 | 1.0 | One model, one window. |

A negative adjusted budget on a GPU machine stops planning with `Strategy '<slug>' requires more VRAM than is available...`. GRPO needs more than 6 GB raw, DPO more than 2 GB.

### Tier ceilings

The largest trainable model for each tier, in billions of parameters:

| Tier | Ceiling |
|---|---|
| `full_finetune` | (adjusted − 2.0) / 18 |
| `lora_bf16` | (adjusted − 2.1) / 2 |
| `qlora_8bit` | (adjusted − 2.1) / 1 |
| `qlora_4bit` | (adjusted − 2.1) / 0.5 |

A ceiling below zero is 0. The planner searches the Hub up to the largest ceiling, takes the largest model on the shortlist, and gives it the least quantized tier whose ceiling it fits under. See [How planning works](configuration-guide.md#step-6-picking-the-model-and-tier).

## Worked examples

### A single GPU

Ceilings in billions of parameters, for one GPU of each size:

| GPU | Strategy | Adjusted | Full | LoRA bf16 | QLoRA 8-bit | QLoRA 4-bit |
|---|---|---|---|---|---|---|
| 8 GB | sft / qa | 8 GB | 0.33 | 2.95 | 5.9 | 11.8 |
| 8 GB | dpo | 3 GB | 0.06 | 0.45 | 0.9 | 1.8 |
| 8 GB | grpo | 2 GB | 0 | 0 | 0 | 0 |
| 12 GB | sft / qa | 12 GB | 0.56 | 4.95 | 9.9 | 19.8 |
| 12 GB | dpo | 5 GB | 0.17 | 1.45 | 2.9 | 5.8 |
| 12 GB | grpo | 6 GB | 0.22 | 1.95 | 3.9 | 7.8 |
| 16 GB | sft / qa | 16 GB | 0.78 | 6.95 | 13.9 | 27.8 |
| 16 GB | dpo | 7 GB | 0.28 | 2.45 | 4.9 | 9.8 |
| 16 GB | grpo | 10 GB | 0.44 | 3.95 | 7.9 | 15.8 |
| 24 GB | sft / qa | 24 GB | 1.22 | 10.95 | 21.9 | 43.8 |
| 24 GB | dpo | 11 GB | 0.5 | 4.45 | 8.9 | 17.8 |
| 24 GB | grpo | 18 GB | 0.89 | 7.95 | 15.9 | 31.8 |
| 80 GB | sft / qa | 80 GB | 4.33 | 38.95 | 77.9 | 155.8 |
| 80 GB | dpo | 39 GB | 2.06 | 18.45 | 36.9 | 73.8 |
| 80 GB | grpo | 74 GB | 4.0 | 35.95 | 71.9 | 143.8 |

!!! example "SFT on a 24 GB GPU"
    Adjusted VRAM is 24 GB. The full fine-tuning ceiling is (24 − 2) / 18 ≈ 1.22B.

    - A 0.5B model trains as `full_finetune`.
    - A 3B model trains as `lora_bf16` (3 ≤ 10.95).
    - An 8B model trains as `lora_bf16` (8 ≤ 10.95).
    - A 14B model trains as `qlora_8bit` (14 ≤ 21.9).
    - A 32B model trains as `qlora_4bit` (32 ≤ 43.8), with a warning.

!!! example "DPO on a 24 GB GPU"
    Adjusted VRAM is (24 − 1 × 2) / 2 = 11 GB. A 1.5B model fits LoRA bf16 (1.5 ≤ 4.45). An 8B model needs QLoRA 8-bit (8 ≤ 8.9).

!!! example "GRPO on an 8 GB GPU"
    Adjusted VRAM is 8 − 3 × 2 = 2 GB, which leaves nothing after the activation and adapter reserve. Every ceiling is 0, the plan warns `No single-GPU tier is viable`, and no model is found. Use SFT, or a larger GPU.

The plan's **VRAM estimate** (`vram_estimate` in the API) shows the same numbers for your machine: `raw_vram_gb`, `adjusted_vram_gb`, the two constants, `bytes_per_param` for the chosen tier and `max_params_by_tier`.

### Multiple GPUs


The planner adds the memory of every selected GPU, but uses that sum only where it is safe:

- **Full fine-tuning** uses the summed budget, because FSDP can shard parameters, gradients and optimizer state across GPUs.
- **Adapter tiers** (LoRA, QLoRA) use the **smallest** GPU's budget, because each data-parallel process keeps a full copy of the frozen weights.

!!! example "SFT on two 24 GB GPUs"
    Raw VRAM is 48 GB.

    - Full fine-tuning ceiling from the sum: (48 − 2) / 18 ≈ 2.56B.
    - Adapter ceilings from one 24 GB card: LoRA bf16 10.95B, 8-bit 21.9B, 4-bit 43.8B.

    A 1B model at full fine-tuning needs 1 × 18 + 2 = 20 GB, which fits one card, so the plan uses **data parallel**. A 2B model needs 38 GB, which doesn't fit one card, so the plan uses **FSDP** across both. An 8B model is planned as LoRA bf16 with data parallel.

When the tier is an adapter tier, `adjusted_vram_gb` in the plan reports the smallest GPU's budget. Training starts one process per selected GPU through Accelerate. See [Multi-GPU training](../strategies/multi-gpu.md) for data parallel vs. FSDP and the override.

### Apple Silicon (MPS)

- Memory is counted as a flat 8 GB, whatever the machine's unified memory.
- `qlora_8bit` and `qlora_4bit` are disabled, because bitsandbytes doesn't support MPS. The plan is either `full_finetune` (up to 0.33B for SFT) or `lora_bf16` (up to 2.95B for SFT).
- Training runs in fp32, and the plan warns that MPS is capped at LoRA bf16.
- Parallelism is always data parallel on a single device.
- GRPO's adjusted budget on MPS is 2 GB, so no tier is viable for reinforcement training. DPO's ceilings are small (0.45B for LoRA).

See [macOS (Apple Silicon)](../installation/macos-mps.md).

### CPU-only

- The only tier is `qlora_4bit`, with a ceiling of the smaller of 500M parameters and RAM in bytes ÷ 3. On a 16 GB machine that's 500M.
- Quantization is switched off in the generated config (bitsandbytes needs CUDA), so it trains a LoRA adapter on full-precision weights.
- fp32, no Liger kernels, no activation offloading.
- The plan warns that training will be extremely slow.

## VRAM-scaled defaults

A few training settings in the generated `strategy_config` depend on **total** raw VRAM, summed across the selected GPUs. MPS and CPU-only machines get the smallest settings.

| Setting | Below 12 GB | 12 to 32 GB | 32 GB and above |
|---|---|---|---|
| SFT, DPO `max_length` | 512 | 1024 | 2048 |
| SFT, DPO `activation_offloading` | true (false on CPU and MPS) | false | false |
| SFT `gradient_accumulation_steps` | 16 | 8 | 4 |
| DPO `gradient_accumulation_steps` | 32 | 16 | 8 |
| GRPO `max_completion_length` | 128 | 256 | 512 |
| GRPO `gradient_accumulation_steps` | 32 | 16 | 8 |

| Setting | Below 12 GB | 12 GB and above |
|---|---|---|
| QA `max_length` | 256 | 384 |
| QA `per_device_train_batch_size` | 1 | 4 |
| QA `gradient_accumulation_steps` | 8 | 1 |
| QA `gradient_checkpointing` | true | false |

Other settings don't scale with memory: SFT, DPO and GRPO always use `per_device_train_batch_size = 1` with gradient checkpointing on. SFT always enables packing. Liger kernels are enabled for SFT and DPO when a CUDA GPU is present and `liger-kernel` is installed. Quantization settings (`model_init_kwargs.quantization_config`) follow the tier. See [Memory optimizations](../strategies/memory-optimizations.md) and the [training config reference](../api-reference/training-config.md).

!!! note "Summed VRAM and per-GPU limits"
    These thresholds use the total across GPUs. Two 16 GB GPUs count as 32 GB and get `max_length = 2048`, even though each process runs on a 16 GB card. If that runs out of memory, send an edited plan with a smaller `max_length`. See [How planning works](configuration-guide.md#through-the-http-api).

## What the budget leaves out

The estimates are conservative planning bounds, not measured guarantees.

- The 2 GB activation budget assumes batch size 1 with gradient checkpointing. Very long sequences can exceed it.
- Other processes on the GPU, a desktop session, or a model loaded in the playground use memory the planner thinks is free.
- System RAM is never counted as extra VRAM, and CPU offloading for training is not configured.
- The disk check compares free space on `/` with the model's weight size only. Checkpoints need more, and your output directory may be on another disk.

If a run still runs out of memory, see [Performance and out-of-memory](../troubleshooting/performance.md).

## See also

- [How planning works](configuration-guide.md)
- [Multi-GPU training](../strategies/multi-gpu.md)
- [LoRA and QLoRA](../strategies/qlora.md)
- [Full fine-tuning](../strategies/full-finetuning.md)
- [Performance and out-of-memory](../troubleshooting/performance.md)
