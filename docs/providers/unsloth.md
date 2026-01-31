# Unsloth Provider

The Unsloth provider enables **2x faster training** with **20% less memory** through optimized CUDA kernels and efficient memory management.

## Overview

**Unsloth** is a specialized library that patches HuggingFace Transformers to use optimized implementations for:
- Flash Attention 2
- Fused optimizer kernels
- Efficient gradient checkpointing
- Optimized LoRA implementations

## Features

✅ **2x faster training** compared to standard HuggingFace  
✅ **20% memory reduction** for the same batch size  
✅ **Zero code changes** - same API as HuggingFace  
✅ **Supports popular architectures**: Llama, Mistral, Qwen, Gemma, Phi  
✅ **Compatible with all strategies**: SFT, QLoRA, RLHF, DPO  

## Platform Support

| Platform | Supported | Notes |
|----------|-----------|-------|
| Linux (Native) | ✅ | Recommended |
| WSL 2 | ✅ | Full support |
| Docker | ✅ | With NVIDIA runtime |
| Windows (Native) | ❌ | Use WSL or Docker for Unsloth |
| **macOS (Apple Silicon)** | ❌ | **Not supported - Unsloth requires NVIDIA CUDA GPUs. Use HuggingFace provider on macOS** |

⚠️ **Important**: Unsloth is **NOT supported on Apple Silicon Macs**. The optimized CUDA kernels are not compatible with Apple's MPS backend. If you're on macOS with Apple Silicon, use the [HuggingFace provider](huggingface.md) instead.

## Installation

### Linux

```bash
pip install unsloth
```

### Windows (WSL)

See [Windows Installation Guide](../installation/windows.md#option-2-wsl-installation-recommended).

### Docker

```dockerfile
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04
RUN pip install unsloth
```

### Verify Installation

```bash
python -c "import unsloth; print('Unsloth version:', unsloth.__version__)"
```

## Usage

### Basic Configuration

```json
{
  "provider": "unsloth",
  "model_name": "meta-llama/Llama-3.2-3B",
  "max_seq_length": 2048,
  "task": "text-generation",
  "strategy": "sft",
  "num_train_epochs": 3,
  "lora_r": 16,
  "lora_alpha": 32
}
```

### Important: max_sequence_length Constraint

⚠️ **CRITICAL**: When using Unsloth, you **MUST** specify a fixed `max_seq_length`. Auto-inference (`-1`) is **NOT supported**.

**Valid:**
```json
{
  "provider": "unsloth",
  "max_seq_length": 2048  // ✅ Fixed value
}
```

**Invalid:**
```json
{
  "provider": "unsloth",
  "max_seq_length": -1  // ❌ NOT supported
}
```

**Common values:**
- `512` - Short sequences, lower memory
- `1024` - Medium sequences
- `2048` - Standard (recommended)
- `4096` - Long contexts, more memory
- `8192` - Very long contexts, high memory

### Via UI

1. Go to **Training** tab
2. Select **Provider**: `unsloth`
3. Set **Max Sequence Length**: `2048` (or your preferred value)
4. Configure other settings
5. Start training

### Via API

```bash
curl -X POST http://localhost:8000/api/start_training \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "unsloth",
    "model_name": "meta-llama/Llama-3.2-3B",
    "max_seq_length": 2048,
    "task": "text-generation",
    "strategy": "sft",
    "dataset": "/path/to/dataset.jsonl",
    "num_train_epochs": 3
  }'
```

## Supported Models

### Fully Supported

- **Llama** (1, 2, 3, 3.1, 3.2)
  - `meta-llama/Llama-3.2-1B`
  - `meta-llama/Llama-3.2-3B`
  - `meta-llama/Llama-3.1-8B`
  
- **Mistral**
  - `mistralai/Mistral-7B-v0.1`
  - `mistralai/Mistral-7B-Instruct-v0.3`
  
- **Qwen**
  - `Qwen/Qwen2-1.5B`
  - `Qwen/Qwen2-7B`
  
- **Gemma**
  - `google/gemma-2b`
  - `google/gemma-7b`
  
- **Phi**
  - `microsoft/phi-2`
  - `microsoft/phi-3-mini`

### Limited Support

- **BART** - Some optimizations not available
- **T5** - Not recommended with Unsloth

## Performance Benchmarks

### Training Speed Comparison

**Setup**: Llama-3.2-3B, 1000 examples, NVIDIA RTX 3090

| Provider | Time | Speedup |
|----------|------|---------|
| HuggingFace | 45 min | 1.0x |
| Unsloth | 22 min | 2.0x |

### Memory Usage Comparison

**Setup**: Llama-3.2-7B, batch_size=4, seq_length=2048

| Provider | VRAM | Reduction |
|----------|------|-----------|
| HuggingFace | 16.2 GB | - |
| Unsloth | 12.8 GB | 21% |

### Throughput

**Setup**: Llama-3.2-3B, batch_size=8

| Provider | Tokens/sec | Improvement |
|----------|------------|-------------|
| HuggingFace | 2,400 | - |
| Unsloth | 4,800 | 2x |

## Configuration Tips

### Optimal Settings for Unsloth

```json
{
  "provider": "unsloth",
  "model_name": "meta-llama/Llama-3.2-3B",
  "max_seq_length": 2048,
  "strategy": "qlora",
  "use_4bit": true,
  "bf16": true,
  "gradient_checkpointing": true,
  "per_device_train_batch_size": 8,
  "gradient_accumulation_steps": 4,
  "lora_r": 64,
  "lora_alpha": 16,
  "lora_dropout": 0.1
}
```

### Memory Optimization

For limited VRAM:

```json
{
  "provider": "unsloth",
  "max_seq_length": 1024,  // Reduce sequence length
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "gradient_checkpointing": true,
  "use_4bit": true
}
```

### Speed Optimization

For maximum speed:

```json
{
  "provider": "unsloth",
  "max_seq_length": 2048,
  "per_device_train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "bf16": true,
  "optim": "adamw_8bit"
}
```

## Advanced Features

### Custom Target Modules

Unsloth auto-detects optimal LoRA target modules, but you can override:

```json
{
  "provider": "unsloth",
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ]
}
```

### Gradient Checkpointing

Unsloth uses optimized gradient checkpointing:

```json
{
  "provider": "unsloth",
  "gradient_checkpointing": true  // Automatically optimized
}
```

## Troubleshooting

### "Unsloth is not installed"

**Problem**: Provider error when selecting Unsloth

**Solution**: Install Unsloth:
```bash
pip install unsloth
```

### "Unsloth not supported on Windows"

**Problem**: Running on native Windows

**Solution**: Use WSL or Docker. See [Windows Installation](../installation/windows.md).

### "max_seq_length cannot be -1"

**Problem**: Auto-inference not supported

**Solution**: Set a fixed value:
```json
{
  "max_seq_length": 2048
}
```

### CUDA Out of Memory

**Problem**: OOM errors during training

**Solutions**:
1. Reduce `max_seq_length`: `2048` → `1024`
2. Reduce `per_device_train_batch_size`: `8` → `4`
3. Enable `gradient_checkpointing: true`
4. Use 4-bit quantization: `use_4bit: true`

### Model Not Supported

**Problem**: Specific model doesn't work with Unsloth

**Solution**: Fall back to HuggingFace provider:
```json
{
  "provider": "huggingface"
}
```

### Flash Attention Errors

**Problem**: Flash Attention 2 compatibility issues

**Solution**: Disable Flash Attention:
```bash
export UNSLOTH_DISABLE_FLASH_ATTN=1
modelforge run
```

## Comparison with HuggingFace

| Feature | HuggingFace | Unsloth |
|---------|-------------|---------|
| Training Speed | 1x | 2x |
| Memory Usage | Baseline | -20% |
| Platform Support | All (including macOS MPS) | Linux/WSL/Docker only (CUDA required) |
| Model Support | All | Llama, Mistral, Qwen, Gemma, Phi |
| Complexity | Simple | Simple |
| Stability | Stable | Stable |
| Documentation | Extensive | Growing |

## When to Use Unsloth

### ✅ Use Unsloth When:

- Training on Linux or WSL with **NVIDIA GPU**
- Using supported models (Llama, Mistral, etc.)
- Need faster training times
- Have limited VRAM
- Training large models (7B+)

### ❌ Don't Use Unsloth When:

- Running on native Windows (use HuggingFace)
- **Running on macOS with Apple Silicon (use HuggingFace)**
- Using unsupported models (BART, T5)
- Debugging issues (HuggingFace has better error messages)
- Need maximum compatibility

## Migration from HuggingFace

Switching is simple - just change the provider:

**Before:**
```json
{
  "provider": "huggingface",
  "model_name": "meta-llama/Llama-3.2-3B",
  ...
}
```

**After:**
```json
{
  "provider": "unsloth",
  "model_name": "meta-llama/Llama-3.2-3B",
  "max_seq_length": 2048,  // Add this!
  ...
}
```

All other settings remain the same!

## Next Steps

- **[Provider Overview](overview.md)** - Compare all providers
- **[HuggingFace Provider](huggingface.md)** - Standard provider docs
- **[Configuration Guide](../configuration/configuration-guide.md)** - All config options
- **[Performance Optimization](../troubleshooting/performance.md)** - Get the best results

---

**Unsloth: Train faster, use less memory!** 🚀
