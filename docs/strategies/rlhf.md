# RLHF Strategy (Reinforcement Learning from Human Feedback)

Advanced fine-tuning using human preference learning, powered by DPO (Direct Preference Optimization) internally.

## Overview

RLHF is an advanced training strategy that aligns language models with human preferences. In ModelForge, RLHF uses DPO (Direct Preference Optimization) as its underlying implementation — no reward model is needed.

## What is RLHF in ModelForge?

ModelForge's **RLHF** strategy uses DPO internally, which is the modern approach to preference learning:

- **No reward model needed** — preferences are learned directly from chosen/rejected pairs
- **No reinforcement learning loop** — uses standard supervised optimization
- **Conservative defaults** — lower learning rate and fewer epochs than DPO for stability

> **Note**: Both the RLHF and DPO strategies in ModelForge use TRL's `DPOTrainer` internally. RLHF uses more conservative hyperparameter defaults suited for alignment tasks.

## Features

✅ **Human-aligned outputs** - Learn from human preferences
✅ **No reward model needed** - Direct optimization on preference pairs
✅ **Stable training** - DPO-based, no RL instability
✅ **Conservative defaults** - Tuned for alignment tasks
⚠️ **Requires preference data** - Needs prompt/chosen/rejected format
⚠️ **Text-generation only** - Schema validation enforces `"task": "text-generation"`

## When to Use RLHF

### ✅ Use RLHF When:

- Aligning model outputs with human preferences
- Training conversational AI or assistants
- Have preference pairs (prompt/chosen/rejected)
- Want conservative, stability-focused defaults
- Quality matters more than speed

### ❌ Don't Use RLHF When:

- First time fine-tuning (start with SFT)
- Limited VRAM (< 12GB)
- Simple supervised learning task
- Don't have preference data

## Dataset Format

RLHF requires datasets with preference pairs:

```jsonl
{"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris.", "rejected": "I don't know."}
{"prompt": "Explain quantum computing", "chosen": "Quantum computing uses quantum bits...", "rejected": "It's complicated."}
{"prompt": "Write a haiku about coding", "chosen": "Code flows like water\nBugs hide in silent shadows\nDebug brings the light", "rejected": "Coding is fun"}
```

**Required Fields**:
- `prompt`: Input prompt or question
- `chosen`: Preferred/better response
- `rejected`: Non-preferred/worse response

### Dataset Preparation

1. **Collect Human Feedback**: Get humans to rank multiple responses
2. **Create Preference Pairs**: Pair each prompt with chosen and rejected responses
3. **Quality Control**: Ensure clear preference distinctions
4. **Balance Dataset**: Include diverse prompts and preferences

### Example Dataset Creation

```python
# From human ratings
ratings = [
    {"prompt": "Tell me a joke", "response_a": "Why did the chicken...", "response_b": "Haha funny", "preference": "a"},
]

# Convert to RLHF format
rlhf_data = []
for item in ratings:
    rlhf_data.append({
        "prompt": item["prompt"],
        "chosen": item["response_a"] if item["preference"] == "a" else item["response_b"],
        "rejected": item["response_b"] if item["preference"] == "a" else item["response_a"]
    })
```

## Configuration

### Basic RLHF Configuration

```json
{
  "strategy": "rlhf",
  "task": "text-generation",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset": "/path/to/preference-data.jsonl",
  "provider": "huggingface",

  "num_train_epochs": 1,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "learning_rate": 1.41e-5,

  "lora_r": 16,
  "lora_alpha": 32,
  "use_4bit": true,
  "bf16": true
}
```

### Advanced RLHF Configuration

```json
{
  "strategy": "rlhf",
  "task": "text-generation",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset": "/path/to/preference-data.jsonl",
  "provider": "unsloth",

  "num_train_epochs": 1,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "learning_rate": 1.41e-5,

  "lora_r": 64,
  "lora_alpha": 16,
  "use_4bit": true,
  "bf16": true,

  "max_seq_length": 2048,
  "warmup_ratio": 0.1,
  "eval_split": 0.1
}
```

## How RLHF Works

### Training Process

```
1. Load Pre-trained Model
        ↓
2. Apply LoRA Adapters
        ↓
3. Load Preference Dataset
        ↓
4. DPO Optimization:
   - Process prompt
   - Score chosen response
   - Score rejected response
   - Optimize to prefer chosen
        ↓
5. Save Fine-tuned Model
```

### Key Differences from SFT

| Aspect | SFT | RLHF |
|--------|-----|------|
| **Objective** | Minimize loss on examples | Maximize preference margin |
| **Training** | Supervised learning | Direct preference optimization |
| **Dataset** | Input-output pairs | Preference pairs |
| **Complexity** | Simple | Medium |
| **Speed** | Fast | Medium |
| **Quality** | High | Very High |

## Hardware Requirements

### Minimum Requirements

- **GPU**: 12GB+ VRAM
- **RAM**: 16GB+ system RAM
- **Recommended**: Mid-range to high-end profile
- **Provider**: HuggingFace or Unsloth

### Memory Usage

RLHF memory usage is similar to DPO since both use the DPOTrainer internally.

**Example** (7B model):
- SFT with 4-bit: ~6-8 GB VRAM
- RLHF with 4-bit: ~8-10 GB VRAM

## Recommended Settings by Hardware

### Mid Range (12-16GB VRAM)

```json
{
  "strategy": "rlhf",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "use_4bit": true,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "lora_r": 16,
  "max_seq_length": 1024
}
```

### High End (16GB+ VRAM)

```json
{
  "strategy": "rlhf",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "use_4bit": true,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "lora_r": 64,
  "max_seq_length": 2048
}
```

## Hyperparameter Tuning

### Learning Rate

RLHF uses a **conservative learning rate** by default:

- **SFT**: 2e-4
- **RLHF**: 1.41e-5 (default)
- **DPO**: 5e-7

### LoRA Configuration

```json
{
  "lora_r": 16,        // Can use 16, 32, 64
  "lora_alpha": 32,    // Usually 2x rank
  "lora_dropout": 0.05 // Lower dropout for RLHF
}
```

### Training Epochs

RLHF defaults to **fewer epochs** than DPO:

- **SFT**: 3-5 epochs
- **DPO**: 3 epochs (default)
- **RLHF**: 1 epoch (default)

## Evaluation

RLHF models are evaluated using:

1. **Preference Accuracy**: How often model prefers chosen over rejected
2. **Reward Margin**: Difference in scores between chosen and rejected
3. **Human Evaluation**: Manual quality assessment

## Common Issues

### High Memory Usage

**Problem**: RLHF runs out of memory

**Solutions**:
- Use 4-bit quantization
- Reduce batch size to 1
- Increase gradient accumulation
- Reduce sequence length
- Use smaller model

### Unstable Training

**Problem**: Loss oscillates or diverges

**Solutions**:
- Lower learning rate (1e-5 or lower)
- Increase warmup steps
- Use gradient clipping
- Check dataset quality

## Comparison with DPO

Both RLHF and DPO use TRL's `DPOTrainer` internally. The difference is in default hyperparameters:

| Setting | RLHF Default | DPO Default |
|---------|-------------|-------------|
| **Learning Rate** | 1.41e-5 | 5e-7 |
| **Epochs** | 1 | 3 |
| **Warmup Ratio** | 0.1 | 0.1 |
| **Batch Size** | 1 | 2 |

**Recommendation**: Try DPO first for most use cases. Use RLHF when you want more conservative training defaults.

## Example: Training a Helpful Assistant

```json
{
  "strategy": "rlhf",
  "task": "text-generation",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset": "/data/helpful-assistant-preferences.jsonl",
  "provider": "unsloth",

  "num_train_epochs": 1,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "learning_rate": 1.41e-5,
  "warmup_ratio": 0.1,

  "lora_r": 64,
  "lora_alpha": 128,
  "use_4bit": true,
  "bf16": true,

  "max_seq_length": 2048,
  "eval_split": 0.1
}
```

## Best Practices

1. ✅ Start with SFT before RLHF
2. ✅ Use high-quality preference data
3. ✅ Use lower learning rates than SFT
4. ✅ Limit to 1-2 epochs
5. ✅ Include diverse prompts in dataset
6. ✅ Evaluate with human feedback
7. ✅ Both RLHF and DPO use DPOTrainer internally — choose based on desired defaults

## Next Steps

- **[DPO Strategy](dpo.md)** - Same underlying trainer, different defaults
- **[SFT Strategy](sft.md)** - Start here before RLHF
- **[QLoRA Strategy](qlora.md)** - Memory-efficient training
- **[Strategy Overview](overview.md)** - Compare all strategies

---

**RLHF: Human-aligned AI made simple with DPO under the hood!**
