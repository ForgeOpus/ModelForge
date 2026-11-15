# LoRA and QLoRA

LoRA trains a small low-rank adapter beside frozen base weights. QLoRA does the same with the base weights quantized to 8 or 4 bits. In v4 these are **PEFT tiers**, not strategies: SFT, DPO, GRPO and QA can each run with any tier. This page covers the three adapter tiers, how one is chosen, the adapter and quantization settings, and what gets saved.

!!! note "Changed in v4"
    In v3, QLoRA was a strategy you picked. In v4 you pick an objective, and the recommendation engine picks the tier from your hardware and the chosen model.

## The idea

Full fine-tuning updates every weight matrix `W`. LoRA freezes `W` and learns an update `B × A`, two thin matrices with rank `r`. The update is scaled by `lora_alpha / r` and added to the layer's output. Only `A` and `B` get gradients and optimizer state, so the large per-parameter training cost disappears.

The frozen weights still have to sit on the GPU. QLoRA stores them in 8-bit or 4-bit form with bitsandbytes and dequantizes them on the fly during each forward pass.

## The tiers

| Tier | Base weights | Bytes per parameter | 7B model weights |
|---|---|---|---|
| `full_finetune` | Trained, bf16/fp16 | 18 | 126 GB (see [Full fine-tuning](full-finetuning.md)) |
| `lora_bf16` | Frozen, unquantized | 2 | 14 GB |
| `qlora_8bit` | Frozen, 8-bit | 1 | 7 GB |
| `qlora_4bit` | Frozen, 4-bit NF4 with double quantization | 0.5 | 3.5 GB |

Each adapter tier also reserves 0.1 GB for adapters and their optimizer state (`ADAPTER_COST_GB`) and 2.0 GB for activations:

```text
lora_bf16 ceiling  = (adjusted VRAM − 2.1 GB) / 2
qlora_8bit ceiling = (adjusted VRAM − 2.1 GB) / 1
qlora_4bit ceiling = (adjusted VRAM − 2.1 GB) / 0.5
```

With several GPUs, adapter ceilings use the **smallest** selected GPU, because every data-parallel process holds a full copy of the frozen model. FSDP is never used with adapters.

## How a tier is chosen

1. The engine computes every tier's ceiling for the strategy's adjusted VRAM.
2. It searches the Hub up to the largest ceiling and takes the largest model in the popular shortlist.
3. It picks the least quantized tier that fits that model: `full_finetune`, then `lora_bf16`, `qlora_8bit`, `qlora_4bit`.

So full fine-tuning isn't picked automatically just because some model fits it. If the shortlist contains a larger model that only fits with an adapter, that model and its adapter tier win. The full algorithm is on [How planning works](../configuration/configuration-guide.md).

For example, on one 16 GB GPU with DPO, an 8B model gets `qlora_4bit`, a 3B model `qlora_8bit`, a 1.5B model `lora_bf16` and a 135M model `full_finetune`. These cases come from `tests/unit_tests/recommendation_selection_test.py`.

When 4-bit was chosen even though LoRA bf16 had room for some model, the plan warns: `QLoRA 4-bit was selected because the chosen model exceeds the unquantized and 8-bit training budgets.` To train with less quantization, choose a smaller model.

### CPU and Apple Silicon

| Hardware | Tier | Notes |
|---|---|---|
| CPU only | Always `qlora_4bit` | Ceiling is the smaller of 500M parameters and RAM ÷ 3 bytes. bitsandbytes needs CUDA, so quantization is switched off and the adapter trains on full-precision weights. Very slow. |
| Apple MPS | `full_finetune` or `lora_bf16` | Memory is counted as 8 GB. bitsandbytes doesn't support MPS, so the 8-bit and 4-bit ceilings are set to 0. |

## Configure it

### Adapter settings

The recommendation's `peft_config` holds three values. For a 3B model under SFT:

```json
{
  "r": 16,
  "lora_alpha": 32,
  "target_modules": "all-linear"
}
```

| Strategy | `r` by model size | `lora_alpha` |
|---|---|---|
| `sft`, `dpo` | 8 under 1B · 16 under 7B · 32 under 13B · 64 from 13B | 2 × `r` |
| `grpo` | 8 under 1B · 16 from 1B | 2 × `r` |
| `qa` | 16 under 7B · 32 from 7B | 2 × `r` |

`target_modules` is always `"all-linear"`. PEFT resolves it per architecture to every linear layer except the output head, so attention and MLP projections both get adapters.

At launch, `workflow/training.py` builds a `peft.LoraConfig` from these values and a task type derived from the task:

| Task | `task_type` |
|---|---|
| `causal_language_modeling` | `CAUSAL_LM` |
| `seq2seq_modeling` | `SEQ_2_SEQ_LM` |
| `question_answering` | `QUESTION_ANS` |

Other `LoraConfig` fields keep PEFT's defaults: `lora_dropout=0.0`, `bias="none"`, no rsLoRA.

### Quantization settings

The tier sets `model_init_kwargs` in the strategy config. The Hugging Face provider converts the dict to a `BitsAndBytesConfig` when it loads the model.

| Tier | `model_init_kwargs` |
|---|---|
| `full_finetune`, `lora_bf16` | `{}` |
| `qlora_8bit` | `{"quantization_config": {"load_in_8bit": true}}` |
| `qlora_4bit` | `{"quantization_config": {"load_in_4bit": true, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": true}}` |

On CPU and MPS the dict is always `{}`. The 4-bit compute dtype isn't set, so bitsandbytes uses its default. For backward compatibility the provider also accepts the older top-level `load_in_4bit` / `load_in_8bit` flags in saved plans, but not together with `quantization_config`.

## What gets saved

Adapter runs save only the adapter with `trainer.model.save_pretrained(output_dir)`: `adapter_config.json` and `adapter_model.safetensors`, typically a few MB to a few hundred MB, plus the tokenizer. ModelForge doesn't merge the adapter into the base weights.

Runs started from the browser or notebook also write `modelforge_config.json` with the base model, task, strategy and tier. The [Playground](../user-guide/playground.md) uses it to reload the adapter on the right base model. To use the adapter elsewhere, load the base model and apply the adapter with `peft.PeftModel.from_pretrained`.

## Choosing a rank

Start with the default. Raise `r` if training loss plateaus well above what you expect, and lower it if the model overfits a small dataset. Because every linear layer already gets an adapter, rank is the main capacity knob. You can change it only by sending an edited `recommendation` to `POST /api/runs`. See [Training config reference](../api-reference/training-config.md).

## Limitations

- No control in the console to force a tier or change the rank.
- No adapter merging, and no DoRA, rsLoRA or other LoRA variants.
- QLoRA tiers need CUDA. They are disabled on CPU and MPS.
- Adapter runs can't use FSDP. A request that combines FSDP with an adapter tier is rejected. See [Multi-GPU training](multi-gpu.md#fsdp-and-peft).

## See also

- [Full fine-tuning](full-finetuning.md)
- [How planning works](../configuration/configuration-guide.md)
- [Hardware and memory](../configuration/hardware-profiles.md)
- [Strategies overview](overview.md)
