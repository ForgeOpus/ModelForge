# Performance Optimization

Advanced techniques for maximizing ModelForge performance.

## Overview

This guide covers advanced optimization techniques to get the best training performance from ModelForge on your hardware.

## Quick Wins

### 1. Use Unsloth Provider

**Impact**: 2x faster training  
**Effort**: Low

```json
{
  "provider": "unsloth"
}
```

**Requirements**:
- Linux or WSL (not native Windows)
- Compatible model architecture (Llama, Mistral, Qwen, Gemma, Phi)

**Speedup**: Unsloth provides 2x faster training through optimized kernels.

---

### 2. Enable BF16 on Ampere+ GPUs

**Impact**: 20-30% faster  
**Effort**: Low

```json
{
  "bf16": true,
  "fp16": false
}
```

**Requirements**:
- NVIDIA RTX 30xx, 40xx, or A100 GPUs
- PyTorch 2.0+

**Benefits**:
- Faster training than FP16
- Better numerical stability
- No accuracy degradation

---

### 3. Optimize Batch Size

**Impact**: Variable (can be 2-3x faster)  
**Effort**: Medium

**Strategy**:
1. Start with smallest batch size (1)
2. Gradually increase until you hit OOM
3. Back off to the largest size that fits

```json
{
  "per_device_train_batch_size": 4,  // Increase from default 1
  "gradient_accumulation_steps": 2   // Decrease from default 4
}
```

**Formula**:
```
Effective Batch Size = per_device_train_batch_size × gradient_accumulation_steps
```

Keep effective batch size constant while maximizing `per_device_train_batch_size`.

---

### 4. Use Flash Attention

**Impact**: 20-40% faster for long sequences  
**Effort**: Medium

Flash Attention is automatically used by Unsloth when available.

**Manual Installation**:
```bash
pip install flash-attn --no-build-isolation
```

**Benefits**:
- Faster attention computation
- Lower memory usage
- Enables longer sequences

---

### 5. Reduce Sequence Length

**Impact**: Proportional to reduction  
**Effort**: Low

```json
{
  "max_seq_length": 1024  // Instead of 2048
}
```

**Trade-off**: Shorter context windows, but much faster training.

**Recommendations**:
- Text generation: 1024-2048
- Summarization: 512-1024
- Question answering: 512

---

## VRAM Optimization

### Memory Usage Formula

```
VRAM ≈ (Model Size × Precision Factor) + (Batch Size × Sequence Length × Hidden Size)
```

### Reduce Memory Usage

#### 1. Use 4-bit Quantization

**Savings**: ~75% model memory

> **Note**: Quantization requires the `[quantization]` extra: `pip install modelforge-finetuning[quantization]`

```json
{
  "use_4bit": true,
  "bnb_4bit_compute_dtype": "bfloat16",
  "bnb_4bit_quant_type": "nf4"
}
```

#### 2. Enable Gradient Checkpointing

**Savings**: ~30-50% activation memory  
**Cost**: 10-15% slower

```json
{
  "gradient_checkpointing": true
}
```

#### 3. Reduce LoRA Rank

**Savings**: Proportional to rank reduction

```json
{
  "lora_r": 16  // Instead of 64
}
```

**Trade-off**: May reduce model quality with very low ranks.

#### 4. Use Paged Optimizers

**Savings**: Optimizer states paged to CPU RAM

```json
{
  "optim": "paged_adamw_32bit"
}
```

---

## CPU Optimization

### 1. Set Number of Workers

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
```

### 2. Use Multiple CPU Cores

Set in environment:
```bash
export OMP_NUM_THREADS=8  # Use 8 CPU cores
```

### 3. Pin Memory

```json
{
  "dataloader_pin_memory": true
}
```

---

## Data Loading Optimization

### 1. Preprocess Dataset

Tokenize dataset once before training:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("json", data_files="data.jsonl")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.save_to_disk("preprocessed_dataset")
```

Then use preprocessed dataset in training.

### 2. Group by Length

**Impact**: 10-20% faster

```json
{
  "group_by_length": true
}
```

Groups similar-length sequences to minimize padding.

### 3. Enable Packing

**Impact**: 20-40% faster for short sequences

```json
{
  "packing": true
}
```

Packs multiple sequences into one to maximize GPU utilization.

**Caution**: May reduce quality for some tasks.

---

## Training Hyperparameter Tuning

### Learning Rate

**Higher LR** = faster convergence but less stable

**Recommendations**:
- Start with default (2e-4)
- Increase to 3e-4 or 5e-4 if training is stable
- Decrease to 1e-4 if seeing instability

```json
{
  "learning_rate": 3e-4
}
```

### Warmup Ratio

**Lower warmup** = faster initial training

```json
{
  "warmup_ratio": 0.03  // 3% of training
}
```

### Learning Rate Scheduler

```json
{
  "lr_scheduler_type": "cosine"  // Recommended
}
```

**Options**:
- `cosine`: Best for most cases
- `linear`: Simpler, sometimes faster
- `constant`: No decay, fastest but may not converge well

---

## Model-Specific Optimizations

### Llama Models

```json
{
  "provider": "unsloth",
  "use_4bit": true,
  "bf16": true,
  "max_seq_length": 2048,
  "gradient_checkpointing": true
}
```

### Qwen Models

```json
{
  "provider": "unsloth",
  "use_4bit": true,
  "bf16": true,
  "max_seq_length": 2048
}
```

### T5/BART Models

```json
{
  "provider": "huggingface",  // Unsloth doesn't support encoder-decoder
  "use_4bit": true,
  "bf16": true,
  "max_seq_length": 1024
}
```

---

## Hardware-Specific Optimizations

### RTX 30xx/40xx (Ampere/Ada)

```json
{
  "provider": "unsloth",
  "bf16": true,
  "fp16": false,
  "use_4bit": true,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 2
}
```

**Key Features**:
- BF16 support
- Tensor cores
- High memory bandwidth

### RTX 20xx (Turing)

```json
{
  "provider": "unsloth",
  "fp16": true,
  "bf16": false,
  "use_4bit": true,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4
}
```

**Key Features**:
- FP16 support (no BF16)
- Tensor cores

### GTX 16xx/10xx (Pascal/Turing)

```json
{
  "provider": "huggingface",  // Unsloth may not work well
  "fp16": true,
  "use_4bit": true,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "max_seq_length": 512
}
```

**Limitations**:
- Limited FP16 support
- No tensor cores
- Lower memory bandwidth

---

## Benchmarking

### Measure Training Speed

Monitor:
- **Iterations per second**
- **Samples per second**
- **Seconds per epoch**

### Find Optimal Settings

```python
# Test different batch sizes
for batch_size in [1, 2, 4, 8]:
    for grad_accum in [16, 8, 4, 2]:
        if batch_size * grad_accum == 16:  # Keep effective batch size constant
            print(f"Testing: batch={batch_size}, grad_accum={grad_accum}")
            # Run short training and measure speed
```

### Performance Metrics

Track these metrics:
- VRAM usage (`nvidia-smi`)
- Training speed (samples/sec)
- Time per epoch
- GPU utilization (should be >90%)

---

## Common Bottlenecks

### 1. Low GPU Utilization

**Symptoms**: GPU utilization < 80%

**Causes**:
- CPU preprocessing bottleneck
- Small batch size
- Data loading slow

**Solutions**:
- Increase batch size
- Preprocess dataset
- Increase data loading workers

### 2. OOM (Out of Memory)

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Reduce batch size
2. Increase gradient accumulation
3. Enable 4-bit quantization
4. Reduce sequence length
5. Enable gradient checkpointing
6. Use smaller LoRA rank

### 3. Slow Data Loading

**Symptoms**: GPU waits for data

**Solutions**:
- Preprocess dataset
- Increase num_workers
- Use faster storage (SSD vs HDD)
- Enable data caching

### 4. Gradient Overflow (FP16)

**Symptoms**: Loss becomes NaN

**Solutions**:
- Use BF16 instead of FP16
- Reduce learning rate
- Enable gradient clipping
- Use mixed precision with loss scaling

---

## Advanced Techniques

### 1. Gradient Accumulation Optimization

Find optimal balance:
```python
total_batch_size = 16

# Test configurations
configs = [
    (1, 16),   # batch=1, grad_accum=16
    (2, 8),    # batch=2, grad_accum=8
    (4, 4),    # batch=4, grad_accum=4
    (8, 2),    # batch=8, grad_accum=2
]

# Larger batch_size usually faster if it fits in VRAM
```

### 2. Mixed Precision Training

```json
{
  "bf16": true,
  "fp16": false,
  "tf32": true  // Enable TF32 on Ampere+
}
```

### 3. Compilation (PyTorch 2.0+)

```python
import torch
model = torch.compile(model, mode="max-autotune")
```

**Benefits**: 10-30% speedup  
**Caution**: Experimental, may cause issues

### 4. Multi-GPU Training

For multiple GPUs, ModelForge automatically uses DataParallel.

```bash
CUDA_VISIBLE_DEVICES=0,1 modelforge
```

---

## Performance Checklist

Before training, ensure:

- [ ] Using Unsloth provider (if compatible)
- [ ] BF16 enabled (Ampere+ GPUs)
- [ ] 4-bit quantization enabled
- [ ] Batch size maximized (no OOM)
- [ ] Gradient checkpointing enabled
- [ ] Dataset preprocessed
- [ ] Sequence length optimized for task
- [ ] Flash Attention installed
- [ ] GPU utilization > 90%

---

## Troubleshooting Performance Issues

### Training is Slower Than Expected

1. Check GPU utilization: `nvidia-smi dmon`
2. Verify Unsloth is being used (check logs)
3. Ensure BF16 is enabled on compatible GPUs
4. Check if CPU is bottleneck (htop)
5. Verify no background processes using GPU

### Memory Usage Too High

1. Enable 4-bit quantization
2. Reduce batch size
3. Reduce sequence length
4. Lower LoRA rank
5. Enable gradient checkpointing

### Training is Unstable

1. Reduce learning rate
2. Increase warmup steps
3. Enable gradient clipping
4. Switch to BF16 from FP16
5. Check dataset quality

---

## Performance Comparison

### Training Time (1000 examples, 3 epochs)

| Configuration | Time | Speedup |
|---------------|------|---------|
| Baseline (HF, FP16, batch=1) | 90 min | 1x |
| + Unsloth | 45 min | 2x |
| + BF16 | 35 min | 2.6x |
| + Larger batch (4) | 25 min | 3.6x |
| + Optimized settings | 20 min | 4.5x |

**Hardware**: RTX 4070 (12GB), Llama-3.1-8B, QLoRA

---

## Next Steps

- **[Hardware Profiles](../configuration/hardware-profiles.md)** - Optimize for your GPU
- **[Configuration Guide](../configuration/configuration-guide.md)** - All settings
- **[Troubleshooting](common-issues.md)** - Fix common problems

---

**Maximize your training performance!** Every optimization counts.
