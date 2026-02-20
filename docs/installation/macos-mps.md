# macOS Installation Guide (Apple Silicon MPS)

Install and run ModelForge on Apple Silicon Macs (M1/M2/M3/M4) with MPS (Metal Performance Shaders) acceleration.

## Overview

ModelForge supports Apple Silicon Macs through PyTorch's MPS backend. This provides GPU acceleration for training on Apple's M-series chips.

**⚠️ Important Limitations:**
- **HuggingFace provider only** - Unsloth provider is not supported on MPS
- **No 4-bit/8-bit quantization** - bitsandbytes library doesn't support MPS
- **Smaller models recommended** - Unified memory architecture limits model sizes
- **FP16 precision** - Use `fp16: true` for best performance

## Prerequisites

- **macOS** 12.3 or later
- **Apple Silicon** Mac (M1/M2/M3/M4)
- **Python 3.11.x** (Python 3.12 not yet supported)
- **HuggingFace Account** with access token ([Get one here](https://huggingface.co/settings/tokens))

## Installation

### 1. Install Python

Using Homebrew:

```bash
brew install python@3.11
```

Or download from [python.org](https://www.python.org/downloads/).

### 2. Create Virtual Environment

```bash
python3.11 -m venv modelforge-env
source modelforge-env/bin/activate
```

### 3. Install ModelForge

```bash
# Install ModelForge
pip install modelforge-finetuning

# Install PyTorch with MPS support (no CUDA needed)
pip install torch torchvision torchaudio
```

**Note**: The standard PyTorch installation includes MPS support on macOS. No special CUDA installation needed.

### 4. Set HuggingFace Token

```bash
export HUGGINGFACE_TOKEN=your_token_here
```

Or create a `.env` file:

```bash
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

### 5. Run ModelForge

```bash
modelforge run
```

Open your browser to **http://localhost:8000** and start training!

## Configuration for MPS

### Recommended Settings

When training on Apple Silicon with MPS:

```json
{
  "provider": "huggingface",
  "device": "mps",
  "model_name": "qwen/Qwen2.5-3B",
  "task": "text-generation",
  "strategy": "sft",
  
  "use_4bit": false,
  "use_8bit": false,
  "fp16": true,
  "bf16": false,
  
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "gradient_checkpointing": true,
  "max_seq_length": 1024,
  
  "num_train_epochs": 3,
  "learning_rate": 2e-4,
  "dataset": "/path/to/dataset.jsonl"
}
```

### Key Settings Explained

**Provider:**
- ✅ `"provider": "huggingface"` - **Supported**
- ❌ `"provider": "unsloth"` - **Not supported** (CUDA only)

**Device:**
- `"device": "mps"` - Explicitly use MPS
- `"device": "auto"` - Auto-detects and uses MPS if available

**Quantization:**
- ❌ `"use_4bit": false` - **Required** (bitsandbytes doesn't support MPS)
- ❌ `"use_8bit": false` - **Required** (bitsandbytes doesn't support MPS)

**Precision:**
- ✅ `"fp16": true` - **Recommended** for MPS
- ⚠️ `"bf16": false` - bfloat16 support on MPS varies by model

**Batch Size:**
- Use `per_device_train_batch_size: 1-2` to fit in unified memory
- Increase `gradient_accumulation_steps` to compensate (8-16)

**Sequence Length:**
- Keep `max_seq_length` at 512-1024 for smaller models
- 2048 may work for 3B models on M1 Pro/Max or M2

## Recommended Models

Models that work well on Apple Silicon with MPS:

### Small Models (8-16GB Unified Memory)

- **Qwen2.5-3B** - `qwen/Qwen2.5-3B`
- **Phi-3 Mini** - `microsoft/phi-3-mini-4k-instruct`
- **Llama 3.2 1B** - `meta-llama/Llama-3.2-1B`
- **Llama 3.2 3B** - `meta-llama/Llama-3.2-3B`

### Medium Models (32GB+ Unified Memory)

- **Qwen2.5-7B** - `qwen/Qwen2.5-7B`
- **Llama 3.1 8B** - `meta-llama/Llama-3.1-8B` (tight fit)
- **Mistral 7B** - `mistralai/Mistral-7B-v0.3`

**Note**: Larger models (13B+) are generally not feasible on MPS without quantization.

## Verifying MPS Support

Check if MPS is available:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

If MPS is available, you should see:
```
MPS available: True
MPS built: True
```

## Performance Expectations

### Training Speed

MPS training is **slower than NVIDIA GPUs** but **much faster than CPU**:

| Hardware | Relative Speed | Example: 3B Model, 1000 samples |
|----------|---------------|----------------------------------|
| M1/M2 (8GB) | 1x | ~3-4 hours |
| M1 Pro/Max (32GB) | 1.5x | ~2-2.5 hours |
| RTX 3090 (CUDA) | 3-4x | ~45-60 min |

### Memory Usage

Apple Silicon uses **unified memory** (shared between CPU and GPU):

- **Model Size**: ~2x parameters in GB (FP16)
  - 3B model: ~6GB
  - 7B model: ~14GB
- **Training Overhead**: +30-50% memory for gradients, optimizer states

**Recommendations:**
- 8GB unified memory: 1-3B models only
- 16GB unified memory: 3B models comfortably
- 32GB+ unified memory: 3-7B models

## Troubleshooting

### "MPS backend out of memory"

**Problem**: Model too large for available unified memory

**Solutions**:
1. Use a smaller model (3B instead of 7B)
2. Reduce `max_seq_length` (512 instead of 2048)
3. Reduce `per_device_train_batch_size` to 1
4. Enable `gradient_checkpointing: true`
5. Close other applications to free memory

### "Unsloth provider is not supported on Apple MPS"

**Problem**: Trying to use Unsloth with MPS

**Solution**: Change `"provider"` to `"huggingface"` in your config:

```json
{
  "provider": "huggingface",
  "device": "mps"
}
```

### "4-bit quantization via bitsandbytes is not supported on MPS"

**Problem**: Trying to use quantization on MPS

**Solution**: Disable quantization:

```json
{
  "use_4bit": false,
  "use_8bit": false,
  "fp16": true
}
```

### Training is very slow

**Problem**: MPS is slower than CUDA

**Expected Behavior**: This is normal. MPS is 3-5x slower than high-end NVIDIA GPUs but still much faster than CPU.

**Tips to improve speed**:
- Use smaller models (1-3B)
- Reduce `max_seq_length`
- Enable `gradient_checkpointing: false` if you have enough memory
- Close other applications

### "RuntimeError: MPS does not support..."

**Problem**: Some operations are not yet supported on MPS

**Solution**: This is a PyTorch limitation. Try:
1. Update to the latest PyTorch: `pip install --upgrade torch`
2. Use a different model architecture
3. Report the issue to PyTorch

## Comparison: MPS vs CUDA

| Feature | Apple MPS | NVIDIA CUDA |
|---------|-----------|-------------|
| **Providers** | HuggingFace only | HuggingFace + Unsloth |
| **Quantization** | ❌ Not supported | ✅ 4-bit, 8-bit |
| **Model Sizes** | 1-7B (limited by memory) | 1-70B+ with quantization |
| **Training Speed** | Slower (1x baseline) | Faster (2-4x with Unsloth) |
| **Memory** | Unified (shared CPU/GPU) | Dedicated VRAM |
| **Cost** | Built into Mac | Requires separate GPU |

## When to Use MPS

### ✅ Use MPS When:

- You have an Apple Silicon Mac and want to train locally
- Training small to medium models (1-7B)
- Learning and experimenting with fine-tuning
- Don't need the fastest training speeds
- Don't want to invest in a separate GPU

### ❌ Don't Use MPS When:

- Need maximum training speed (use NVIDIA GPU with Unsloth)
- Training very large models (13B+)
- Need 4-bit/8-bit quantization
- Training in production at scale

## Next Steps

- **[Configuration Guide](../configuration/configuration-guide.md)** - Detailed configuration options
- **[Dataset Formats](../configuration/dataset-formats.md)** - Preparing training data
- **[HuggingFace Provider](../providers/huggingface.md)** - Provider documentation
- **[Common Issues](../troubleshooting/common-issues.md)** - General troubleshooting
- **[FAQ](../troubleshooting/faq.md)** - Frequently asked questions

---

**ModelForge on Apple Silicon: Training LLMs on your Mac!** 🍎⚡
