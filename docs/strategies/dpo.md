# DPO Strategy (Direct Preference Optimization)

Simpler and more stable alternative to RLHF for aligning models with human preferences.

## Overview

DPO (Direct Preference Optimization) is a breakthrough training method that achieves RLHF-quality results without the complexity of reward models and reinforcement learning. It directly optimizes the model using preference pairs.

## What is DPO?

**DPO** simplifies preference learning by:
- ✅ **No reward model needed** - Direct optimization on preferences
- ✅ **More stable training** - No RL instability issues
- ✅ **Simpler implementation** - Supervised learning approach
- ✅ **Better performance** - Often matches or exceeds RLHF
- ✅ **Lower memory** - No value head or PPO buffers

## Features

✅ **Simpler than RLHF** - No reward model or PPO required  
✅ **More stable** - Standard supervised training  
✅ **High quality** - Matches RLHF performance  
✅ **Faster training** - More efficient than RLHF  
✅ **Less memory** - Lower VRAM requirements  
✅ **Easy to tune** - Fewer hyperparameters  

## When to Use DPO

### ✅ Use DPO When:

- Aligning model outputs with human preferences
- Training conversational AI or assistants
- Have preference pairs (chosen/rejected examples)
- Want RLHF-quality without complexity
- More stable training than RLHF
- Mid-range to high-end hardware (12GB+ VRAM)

### ❌ Don't Use DPO When:

- First time fine-tuning (start with SFT)
- Limited VRAM (< 12GB) - use QLoRA instead
- Simple supervised learning task (use SFT)
- Don't have preference data (use SFT)

## Dataset Format

DPO requires the same preference pair format as RLHF:

```jsonl
{"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris.", "rejected": "I don't know."}
{"prompt": "Explain photosynthesis", "chosen": "Photosynthesis is the process by which plants convert light energy...", "rejected": "Plants make food."}
{"prompt": "Write a professional email", "chosen": "Dear [Name],\n\nI hope this email finds you well...", "rejected": "Hey, what's up?"}
```

**Required Fields**:
- `prompt`: Input prompt or question
- `chosen`: Preferred/better response (high-quality)
- `rejected`: Non-preferred/worse response (lower-quality)

### Creating Good Preference Pairs

#### ✅ Good Examples

**High Contrast**:
```json
{
  "prompt": "Explain machine learning to a beginner",
  "chosen": "Machine learning is a type of AI where computers learn from examples without being explicitly programmed. For instance, a spam filter learns to identify spam emails by looking at thousands of examples.",
  "rejected": "It's when computers learn stuff."
}
```

**Clear Preference**:
```json
{
  "prompt": "Write a polite decline",
  "chosen": "Thank you for your invitation. Unfortunately, I won't be able to attend, but I appreciate you thinking of me.",
  "rejected": "No thanks."
}
```

#### ❌ Bad Examples

**Too Similar**:
```json
{
  "prompt": "What is 2+2?",
  "chosen": "The answer is 4.",
  "rejected": "2+2 equals 4."
}
```

**No Clear Winner**:
```json
{
  "prompt": "Describe a sunset",
  "chosen": "The sunset was beautiful.",
  "rejected": "The sunset was pretty."
}
```

## Configuration

### Basic DPO Configuration

```json
{
  "strategy": "dpo",
  "task": "text-generation",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset": "/path/to/preference-data.jsonl",
  "provider": "huggingface",
  
  "num_train_epochs": 3,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "learning_rate": 5e-7,
  
  "lora_r": 16,
  "lora_alpha": 32,
  "use_4bit": true,
  "bf16": true
}
```

### Advanced DPO Configuration

```json
{
  "strategy": "dpo",
  "task": "text-generation",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset": "/path/to/preference-data.jsonl",
  "provider": "unsloth",
  
  "num_train_epochs": 3,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 2,
  "learning_rate": 5e-7,
  
  "lora_r": 64,
  "lora_alpha": 128,
  "use_4bit": true,
  "bf16": true,
  
  "max_seq_length": 2048,
  "warmup_ratio": 0.1,
  "eval_split": 0.1
}
```

## How DPO Works

### Training Process

```
1. Load Pre-trained Model
        ↓
2. Apply LoRA Adapters
        ↓
3. Load Preference Dataset
        ↓
4. DPO Training Loop:
   - Process prompt
   - Score chosen response
   - Score rejected response
   - Optimize to prefer chosen
        ↓
5. Save Fine-tuned Model
```

### Mathematical Objective

DPO directly optimizes the model to:
- Increase probability of chosen responses
- Decrease probability of rejected responses
- Stay close to reference model (via KL penalty)

```
Loss = -log(σ(β * log(π(chosen)/π_ref(chosen)) - β * log(π(rejected)/π_ref(rejected))))
```

Where:
- `π` = Policy (fine-tuned model)
- `π_ref` = Reference model (base model)
- `β` = Temperature parameter (controls strength)
- `σ` = Sigmoid function

## Hardware Requirements

### Minimum Requirements

- **GPU**: 12GB+ VRAM
- **RAM**: 16GB+ system RAM
- **Recommended**: Mid-range to high-end profile

### Memory Usage

DPO uses **less memory** than RLHF:

**Example** (7B model):
- SFT with 4-bit: ~6-8 GB VRAM
- DPO with 4-bit: ~8-10 GB VRAM
- RLHF with 4-bit: ~10-12 GB VRAM

## Recommended Settings by Hardware

### Mid Range (12-16GB VRAM)

```json
{
  "strategy": "dpo",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "use_4bit": true,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "lora_r": 16,
  "max_seq_length": 1024
}
```

### High End (16GB+ VRAM)

```json
{
  "strategy": "dpo",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "use_4bit": true,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 2,
  "lora_r": 64,
  "max_seq_length": 2048
}
```

## Hyperparameter Tuning

### Learning Rate

DPO uses **very low learning rates**:

- **SFT**: 2e-4
- **DPO**: 5e-7 (recommended)
- **Range**: 1e-7 to 1e-6

**Why?** DPO is sensitive to learning rate. Too high causes instability.

### Beta Parameter

The beta (β) parameter controls preference strength:

- **Low (0.1)**: Weak preference signal
- **Medium (0.5)**: Balanced (recommended)
- **High (2.0)**: Strong preference signal

**Default**: Usually set internally by TRL library.

### LoRA Configuration

```json
{
  "lora_r": 16,        // 16, 32, or 64
  "lora_alpha": 32,    // Usually 2x rank
  "lora_dropout": 0.05 // Low dropout
}
```

### Training Epochs

```json
{
  "num_train_epochs": 3  // DPO can use more epochs than RLHF
}
```

DPO is **more stable** than RLHF, so you can safely use 2-4 epochs.

## Evaluation

DPO models are evaluated using:

1. **Accuracy**: How often model prefers chosen over rejected
2. **Reward Margin**: Difference in scores between chosen and rejected
3. **KL Divergence**: Deviation from base model
4. **Human Evaluation**: Manual quality assessment

## Common Issues

### Training Instability

**Problem**: Loss oscillates or doesn't converge

**Solutions**:
- Lower learning rate (try 1e-7)
- Increase warmup steps
- Check dataset quality
- Ensure clear preference distinctions

### High Memory Usage

**Problem**: DPO runs out of memory

**Solutions**:
- Use 4-bit quantization
- Reduce batch size
- Increase gradient accumulation
- Reduce sequence length
- Use smaller model

### Poor Preference Learning

**Problem**: Model doesn't learn preferences well

**Solutions**:
- Check dataset has clear chosen/rejected distinctions
- Increase training epochs (2-4)
- Adjust beta parameter
- Use more diverse preference data
- Ensure sufficient data (500+ examples)

## DPO vs RLHF vs SFT

### Comparison Table

| Feature | SFT | DPO | RLHF |
|---------|-----|-----|------|
| **Complexity** | Simple | Medium | High |
| **Stability** | Very Stable | Stable | Less Stable |
| **Quality** | High | Very High | Very High |
| **Speed** | Fast | Medium | Slow |
| **Memory** | Low | Medium | High |
| **Setup** | Easy | Easy | Complex |
| **Preference Learning** | No | Yes | Yes |
| **Reward Model** | No | No | Not required (DPO-based in ModelForge) |

### When to Use Each

```
Need basic fine-tuning?
    → SFT

Have preference data + want simplicity?
    → DPO

Have preference data + want conservative defaults?
    → RLHF

Limited VRAM?
    → QLoRA (with any strategy)
```

> **Note**: In ModelForge, both RLHF and DPO strategies use TRL's `DPOTrainer`. RLHF uses more conservative hyperparameter defaults (lr=1.41e-5, 1 epoch, warmup=0.1) compared to DPO (lr=5e-7, 3 epochs).

## Example: Training a Helpful Assistant

### Step 1: Prepare Preference Dataset

```jsonl
{"prompt": "How do I reset my password?", "chosen": "To reset your password: 1. Click 'Forgot Password' 2. Enter your email 3. Check your inbox for the reset link 4. Follow the instructions in the email", "rejected": "Just reset it."}
{"prompt": "What's the weather like?", "chosen": "I don't have access to real-time weather data. Please check a weather service like weather.com or your local weather app for current conditions.", "rejected": "It's sunny."}
```

### Step 2: Configure Training

```json
{
  "strategy": "dpo",
  "task": "text-generation",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset": "/data/assistant-preferences.jsonl",
  "provider": "unsloth",
  
  "num_train_epochs": 3,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 2,
  "learning_rate": 5e-7,
  "warmup_ratio": 0.1,
  
  "lora_r": 64,
  "lora_alpha": 128,
  "use_4bit": true,
  "bf16": true,
  
  "max_seq_length": 2048,
  "eval_split": 0.1
}
```

### Step 3: Train and Evaluate

Monitor:
- ✅ Preference accuracy should increase
- ✅ Reward margin should grow
- ✅ KL divergence should stay reasonable (< 10)

## Best Practices

1. ✅ **Start with SFT** - Fine-tune base model first, then apply DPO
2. ✅ **High-quality data** - Clear preference distinctions are critical
3. ✅ **Diverse prompts** - Cover various scenarios
4. ✅ **Low learning rate** - 5e-7 is a good starting point
5. ✅ **Monitor KL** - Ensure model doesn't drift too far
6. ✅ **Sufficient data** - 500+ preference pairs recommended
7. ✅ **Validate preferences** - Ensure chosen is actually better

## Advanced Topics

### Two-Stage Training

For best results, use two-stage approach:

**Stage 1: SFT**
```json
{
  "strategy": "sft",
  "dataset": "/data/demonstrations.jsonl",
  "num_train_epochs": 3
}
```

**Stage 2: DPO**
```json
{
  "strategy": "dpo",
  "model_name": "/path/to/sft-model",
  "dataset": "/data/preferences.jsonl",
  "num_train_epochs": 3
}
```

### Dataset Augmentation

Generate synthetic preferences:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")

prompt = "Explain recursion"
responses = generator(prompt, num_return_sequences=2, max_length=100)

# Manual or automated ranking
chosen = responses[0] if better_quality(responses[0], responses[1]) else responses[1]
rejected = responses[1] if chosen == responses[0] else responses[0]
```

## Next Steps

- **[RLHF Strategy](rlhf.md)** - Compare with RLHF approach
- **[SFT Strategy](sft.md)** - Start here before DPO
- **[QLoRA Strategy](qlora.md)** - Memory-efficient training
- **[Strategy Overview](overview.md)** - Compare all strategies

---

**DPO: RLHF quality without the complexity!** 🚀 The modern way to align models.
