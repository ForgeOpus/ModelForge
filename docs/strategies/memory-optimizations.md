# Memory optimizations

ModelForge turns on several techniques that trade compute for memory, or that remove wasted work. This page covers gradient checkpointing, packing, sequence lengths by VRAM, activation offloading and Liger kernels: what each one does, when the planner enables it, and how to check.

All of these are fields in the plan's `strategy_config`, set by each strategy's `build_recommended_config`. "VRAM" below means the total across the selected GPUs as the strategy sees it. CPU-only and Apple MPS machines get the smallest (under 12 GB) settings.

## Summary

| | `sft` (causal) | Summarization | `dpo` | `grpo` | `qa` | Under FSDP |
|---|---|---|---|---|---|---|
| Gradient checkpointing | on | on | on | on | on below 12 GB | replaced by FSDP activation checkpointing |
| Packing (`bfd`) | on | no effect | — | — | — | unchanged |
| Activation offloading | GPU below 12 GB | no effect | GPU below 12 GB | — | — | off |
| Liger kernels | if installed and GPU present | no effect | if installed and GPU present | off | — | off |

"No effect" means the field is in the plan, but the encoder-decoder trainer doesn't receive it. See [Summarization](overview.md#summarization).

## Gradient checkpointing

The backward pass needs every layer's activations from the forward pass. Gradient checkpointing keeps only some of them and recomputes the rest during backward. Activation memory falls to a fraction of its usual size, and steps get slower, typically by 20 to 30 percent.

- `sft`, `dpo`, `grpo`: `gradient_checkpointing: true` on all hardware.
- `qa`: `true` below 12 GB, `false` from 12 GB.
- Under FSDP, `build_fsdp_config` sets `gradient_checkpointing: false` and `fsdp_config.activation_checkpointing: true`. The HF Trainer raises if both are enabled. See [Multi-GPU training](multi-gpu.md#fsdp-configuration).

The planner's 2.0 GB activation budget assumes gradient checkpointing with a per-device batch size of 1.

## Packing

Without packing, each training row holds one example padded to the batch's longest sequence, and short examples waste most of the row. Packing concatenates several examples into each sequence of up to `max_length` tokens.

ModelForge enables packing for causal SFT only, meaning Instruction Following and Text Generation with the supervised objective:

```json
{"packing": true, "packing_strategy": "bfd"}
```

`bfd` is TRL's best-fit decreasing strategy. It sorts examples by length and places each whole example into the first sequence with room, so examples aren't split across sequences.

DPO, GRPO, QA and summarization don't pack.

## Sequence length

Longer sequences cost more activation memory, so the plan sets the length from VRAM:

| VRAM | SFT / DPO `max_length` | GRPO `max_completion_length` | QA `max_length` |
|---|---|---|---|
| under 12 GB (and CPU, MPS) | 512 | 128 | 256 |
| 12 to under 32 GB | 1024 | 256 | 384 |
| 32 GB or more | 2048 | 512 | 384 |

- **SFT:** tokens beyond `max_length` are dropped from the end (`truncation_mode="keep_start"`). With packing, it's the length of each packed sequence.
- **Summarization:** prompt and completion are each truncated to `max_length`.
- **DPO:** `max_length` caps prompt plus response, applied by TRL.
- **GRPO:** `max_completion_length` caps each sampled completion. Prompt length is TRL's default.
- **QA:** contexts longer than `max_length` are split into overlapping windows with `doc_stride=128`, so no answer is lost.

Gradient accumulation also scales with VRAM: SFT uses 16, 8 or 4 steps, and DPO and GRPO 32, 16 or 8, with a per-device batch of 1.

## Activation offloading

Activation offloading moves saved activations to CPU memory between the forward and backward passes, then brings them back. It costs PCIe transfers, and so speed, but lets small cards finish runs that would otherwise run out of memory. ModelForge uses TRL's implementation (`activation_offloading` in `SFTConfig` and `DPOConfig`).

| Strategy | `activation_offloading` |
|---|---|
| `sft`, `dpo` | `true` when there is a GPU and VRAM is under 12 GB, otherwise `false` |
| CPU and MPS | `false`, because there is no GPU memory to relieve |
| `grpo`, `qa`, summarization | Not used |
| Under FSDP | `false` |

There is no CPU offloading of parameters or optimizer state, and system RAM isn't counted as VRAM.

## Liger kernels

Liger is a set of fused Triton kernels for common transformer layers such as RMSNorm, RoPE, SwiGLU and cross-entropy. They compute the same result with less memory, mostly by never materialising large intermediate tensors like the full logits, and are often a little faster.

### Install

Liger is an optional extra, because it needs CUDA and Triton and has no CPU or Windows wheels:

```bash
pip install "modelforge-finetuning[gpu]"
```

That installs `liger-kernel>=0.5.0`. Install it into the same environment as ModelForge, then restart the server. DPO checks for the package when its module is imported, while SFT checks each time it builds a plan.

### When it's on

| Strategy | `use_liger_kernel` |
|---|---|
| `sft` (causal), `dpo` | `true` when VRAM > 0 (a CUDA GPU is selected) and `liger_kernel` is importable |
| `grpo` | Always `false` |
| `qa`, summarization | Not used |
| Under FSDP | `false` |

TRL raises an error, rather than falling back, when `use_liger_kernel` is `true` and the package is missing. That's why ModelForge probes for it before enabling it.

### Checking it

Look for `use_liger_kernel: true` in the plan's **Training config** panel, or in `strategy_config` from `POST /api/recommendations`. If it reads `false` on a CUDA machine, the package isn't importable from the server's environment:

```bash
python -c "import liger_kernel; print('liger ok')"
```

### Compatibility

Liger patches specific architectures, including Llama, Mistral, Qwen, Gemma and Phi. Other models train normally without the benefit. If a run fails inside a Liger kernel, uninstall `liger-kernel` and generate the plan again, so that the new plan has `use_liger_kernel: false`.

## Changing these settings

The console shows the plan read-only. To change a value such as `max_length` or `packing`, send an edited `recommendation` to `POST /api/runs`. It's used as sent when `model_id` is the recommendation's top candidate. See [Training config reference](../api-reference/training-config.md) and [Performance and out-of-memory errors](../troubleshooting/performance.md).

## See also

- [Full fine-tuning](full-finetuning.md) and [LoRA and QLoRA](qlora.md)
- [Multi-GPU training](multi-gpu.md)
- [Hardware and memory](../configuration/hardware-profiles.md)
