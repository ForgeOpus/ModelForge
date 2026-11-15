# Full fine-tuning

Full fine-tuning updates every weight in the model. It is the `full_finetune` PEFT tier, available to every strategy, and the most memory-hungry way to train. This page covers its cost, when the planner picks it, precision, outputs, and how it interacts with DPO and multi-GPU training.

## What it costs

Each trained parameter needs its weight, its gradient and two Adam optimizer states. The planner budgets this at **18 bytes per parameter** plus a **2.0 GB** activation budget (`ACTIVATION_BUDGET_GB` in `utilities/recommendation_engine.py`):

```text
full fine-tune ceiling (parameters) = (adjusted VRAM − 2.0 GB) / 18 bytes
```

| Model size | Budget |
|---|---|
| 0.5B | about 11 GB |
| 1B | about 20 GB |
| 3B | about 56 GB |
| 7B | about 128 GB |

Adjusted VRAM is raw VRAM after the strategy's overhead. For one 24 GB GPU:

| Strategy | Adjusted VRAM | Largest full fine-tune |
|---|---|---|
| `sft`, `qa` | 24 GB | about 1.2B |
| `grpo` | 18 GB | about 0.9B |
| `dpo` | 11 GB | about 0.5B |

These are conservative planning bounds, not measured guarantees. See [Hardware and memory](../configuration/hardware-profiles.md).

## When the planner uses it

The planner doesn't start from the tier. It works out each tier's parameter ceiling, searches the Hub up to the largest of them, and takes the largest model in the popular shortlist. It then picks the **least quantized tier that fits that model**, in the order `full_finetune`, `lora_bf16`, `qlora_8bit`, `qlora_4bit`.

So the plan says `full_finetune` when the chosen model is small enough for it. If the shortlist contains a model too large for full fine-tuning, the plan uses an adapter tier for that larger model instead. Choosing a smaller model in the console, or passing its `model_id` to `POST /api/recommendations`, re-plans for that model and can switch the run to full fine-tuning.

Hardware rules:

- **Multiple CUDA GPUs:** the full fine-tune ceiling uses the **sum** of the selected GPUs' VRAM. A model that doesn't fit on the smallest GPU is then sharded with [FSDP](multi-gpu.md). Adapter tiers are limited by the smallest GPU instead.
- **Apple MPS:** memory is counted as 8 GB. Full fine-tuning is allowed, and the only other tier is `lora_bf16`.
- **CPU only:** never full fine-tuning. The tier is always `qlora_4bit` with quantization switched off. See [LoRA and QLoRA](qlora.md#cpu-and-apple-silicon).

The console has no control to force a tier. The full algorithm is on [How planning works](../configuration/configuration-guide.md).

## Precision

`utilities/precision.py::training_precision` sets `bf16` when every selected CUDA GPU has compute capability 8.0 or newer (Ampere, Ada, Hopper, Blackwell), and `fp16` otherwise. CPU and MPS runs train in full fp32. Base weights are loaded with no quantization (`model_init_kwargs = {}`).

## With each strategy

| Strategy | What full fine-tuning adds |
|---|---|
| `sft` | Nothing beyond the weights. |
| `dpo` | A second, frozen copy of the model is loaded as `ref_model`. The planner divides the remaining VRAM by 2 for every DPO tier. The learning rate is fixed at `1e-7`. |
| `grpo` | No reference copy (`beta = 0`). The learning rate is `1e-6`, against `5e-6` for adapters. |
| `qa` | Nothing beyond the weights. The learning rate stays `3e-5`. |

## Outputs

`trainer.save_model(output_dir)` writes a complete model: config, weights as safetensors, and the tokenizer. It's as large as the base model and loads with `from_pretrained` without PEFT. Under FSDP the shards are gathered into one full state dict (`state_dict_type: FULL_STATE_DICT`), so the output looks the same.

Checkpoints are full copies too, including optimizer state, every 500 steps. They aren't pruned, so check free disk space on long runs. The plan warns when free disk space is below the model's weight footprint at 4 bytes per parameter.

## Limitations

- The memory cost limits single-GPU full fine-tuning to models of around 1B parameters on consumer cards.
- FSDP is only used for full fine-tuning, and only when a model doesn't fit on the smallest GPU. See [Multi-GPU training](multi-gpu.md).
- There is no CPU offloading of parameters or optimizer state. System RAM isn't treated as extra VRAM.

## See also

- [LoRA and QLoRA](qlora.md)
- [Multi-GPU training](multi-gpu.md)
- [Memory optimizations](memory-optimizations.md)
- [Performance and out-of-memory errors](../troubleshooting/performance.md)
