# Configuration Guide

Comprehensive guide to all ModelForge configuration options.

## Overview

ModelForge provides extensive configuration options for fine-tuning. This guide covers all available settings.

## Basic Configuration

### Required Fields

```json
{
  "task": "text-generation",
  "model_name": "meta-llama/Llama-3.2-3B",
  "dataset": "/path/to/dataset.jsonl",
  "num_train_epochs": 3
}
```

- **task**: Training task (`text-generation`, `summarization`, `extractive-question-answering`)
- **model_name**: HuggingFace model ID or local path
- **dataset**: Path to JSONL dataset file
- **num_train_epochs**: Number of training epochs

### Provider Selection

```json
{
  "provider": "huggingface"  // or "unsloth"
}
```

- **huggingface**: Standard provider (default)
- **unsloth**: 2x faster training (Linux/WSL/Docker only)

See [Provider Documentation](../providers/overview.md) for details.

### Strategy Selection

```json
{
  "strategy": "sft"  // or "qlora", "rlhf", "dpo"
}
```

- **sft**: Supervised fine-tuning (default)
- **qlora**: Quantized LoRA for memory efficiency
- **rlhf**: Reinforcement Learning from Human Feedback
- **dpo**: Direct Preference Optimization

See [Strategy Documentation](../strategies/overview.md) for details.

### Schema Validation Rules

ModelForge validates configuration combinations at startup:

- **DPO/RLHF** strategies require `"task": "text-generation"` — using them with summarization or QA tasks will raise a validation error
- **Unsloth** provider requires `"task": "text-generation"` — encoder-decoder models are not supported
- **Unsloth** provider requires a fixed `max_seq_length` (cannot be `-1`)

## Training Parameters

### Epoch and Batch Settings

```json
{
  "num_train_epochs": 3,
  "per_device_train_batch_size": 4,
  "per_device_eval_batch_size": 4,
  "gradient_accumulation_steps": 4
}
```

### Learning Rate

```json
{
  "learning_rate": 2e-4,
  "lr_scheduler_type": "cosine",
  "warmup_steps": 100
}
```

### Optimization

```json
{
  "optim": "adamw_torch",  // or "adamw_8bit", "sgd"
  "weight_decay": 0.01,
  "max_grad_norm": 1.0
}
```

## LoRA Configuration

```json
{
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "target_modules": "all-linear"
}
```

- **lora_r**: LoRA rank (8, 16, 32, 64)
- **lora_alpha**: LoRA alpha (usually 2x rank)
- **lora_dropout**: Dropout rate
- **target_modules**: Modules to apply LoRA

## Quantization

> **Note**: Quantization requires the `[quantization]` extra: `pip install modelforge-finetuning[quantization]`

```json
{
  "use_4bit": true,
  "use_8bit": false,
  "bnb_4bit_compute_dtype": "float16",
  "bnb_4bit_quant_type": "nf4"
}
```

## Mixed Precision

```json
{
  "bf16": true,   // For Ampere+ GPUs (RTX 30xx/40xx)
  "fp16": false   // For older GPUs
}
```

## Sequence Length

```json
{
  "max_seq_length": 2048  // or 512, 1024, 4096, 8192
}
```

**Note**: When using Unsloth, `max_seq_length` cannot be `-1` (auto-inference).

## Evaluation

```json
{
  "eval_split": 0.2,
  "eval_steps": 100,
  "evaluation_strategy": "steps",
  "save_strategy": "steps",
  "save_steps": 500
}
```

## Logging

```json
{
  "logging_steps": 10,
  "logging_strategy": "steps",
  "report_to": "tensorboard"
}
```

## Hardware Profiles

Instead of manual configuration, use hardware profiles:

```json
{
  "compute_specs": "low_end"  // or "mid_range", "high_end"
}
```

See [Hardware Profiles](hardware-profiles.md) for details.

## Complete Example

```json
{
  "task": "text-generation",
  "model_name": "meta-llama/Llama-3.2-3B",
  "provider": "unsloth",
  "strategy": "qlora",
  "dataset": "/path/to/dataset.jsonl",
  "max_seq_length": 2048,
  "num_train_epochs": 3,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "learning_rate": 2e-4,
  "lora_r": 64,
  "lora_alpha": 16,
  "use_4bit": true,
  "bf16": true,
  "eval_split": 0.2,
  "eval_steps": 100,
  "save_steps": 500,
  "logging_steps": 10
}
```

## Next Steps

- [Dataset Formats](dataset-formats.md) - Prepare your data
- [Training Tasks](training-tasks.md) - Understand task types
- [Hardware Profiles](hardware-profiles.md) - Optimize for your GPU
- [API Reference](../api-reference/training-config.md) - Full schema

---

For detailed API schema, see [Training Configuration Schema](../api-reference/training-config.md).
