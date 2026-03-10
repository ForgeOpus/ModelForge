# Adding Model Configurations

Learn how to contribute model recommendations to ModelForge.

## Overview

ModelForge uses a modular configuration system for model recommendations. Each hardware profile has its own JSON configuration file specifying recommended models for different tasks.

## Directory Structure

```
ModelForge/model_configs/
├── low_end.json         # 4-8GB VRAM
├── mid_range.json       # 8-16GB VRAM
└── high_end.json        # 16GB+ VRAM
```

## Configuration Schema

Each configuration file follows this structure:

```json
{
  "profile": "profile_name",
  "tasks": {
    "task_name": {
      "primary": "best_model_id",
      "alternatives": ["model1", "model2", "model3"]
    }
  }
}
```

### Fields

- **profile** (string): Hardware profile name (must match filename without .json)
- **tasks** (object): Task configurations
  - **task_name** (string): One of: `text-generation`, `summarization`, `extractive-question-answering`
  - **primary** (string): Default recommended model for this task/profile
  - **alternatives** (array): List of additional recommended models

## Hardware Profiles

### low_end.json (4-8GB VRAM)

**Target Hardware:**
- NVIDIA GTX 1060 (6GB)
- NVIDIA GTX 1070 (8GB)
- NVIDIA RTX 3050 (8GB)

**Model Criteria:**
- Parameter count: < 3B
- VRAM usage with 4-bit: < 6GB
- Fast inference

**Example:**
```json
{
  "profile": "low_end",
  "tasks": {
    "text-generation": {
      "primary": "meta-llama/Llama-3.2-1B",
      "alternatives": [
        "microsoft/phi-2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
      ]
    },
    "summarization": {
      "primary": "facebook/bart-base",
      "alternatives": [
        "google-t5/t5-small",
        "sshleifer/distilbart-cnn-12-6"
      ]
    }
  }
}
```

### mid_range.json (8-16GB VRAM)

**Target Hardware:**
- NVIDIA RTX 3060 (12GB)
- NVIDIA RTX 2080 Ti (11GB)
- NVIDIA RTX 3080 (10-12GB)

**Model Criteria:**
- Parameter count: 3-7B
- VRAM usage with 4-bit: 6-12GB
- Good quality/performance balance

**Example:**
```json
{
  "profile": "mid_range",
  "tasks": {
    "text-generation": {
      "primary": "meta-llama/Llama-3.2-3B",
      "alternatives": [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/phi-3-mini-4k-instruct"
      ]
    },
    "summarization": {
      "primary": "facebook/bart-large",
      "alternatives": [
        "google-t5/t5-base",
        "philschmid/bart-large-cnn-samsum"
      ]
    }
  }
}
```

### high_end.json (16GB+ VRAM)

**Target Hardware:**
- NVIDIA RTX 3090 (24GB)
- NVIDIA RTX 4090 (24GB)
- NVIDIA A100 (40-80GB)

**Model Criteria:**
- Parameter count: 7B+
- Highest quality
- State-of-the-art performance

**Example:**
```json
{
  "profile": "high_end",
  "tasks": {
    "text-generation": {
      "primary": "meta-llama/Llama-3.1-8B-Instruct",
      "alternatives": [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2-7B-Instruct"
      ]
    },
    "summarization": {
      "primary": "facebook/bart-large-cnn",
      "alternatives": [
        "google-t5/t5-large",
        "google/pegasus-xsum"
      ]
    }
  }
}
```

## Adding Models

### 1. Identify Hardware Profile

Determine which profile(s) the model fits:

- Test VRAM usage with 4-bit quantization
- Consider inference speed
- Evaluate output quality

### 2. Edit Configuration File

Open the appropriate JSON file:

```bash
cd ModelForge/model_configs/
nano mid_range.json  # or low_end.json, high_end.json
```

### 3. Add Model

**As Primary (replaces current default):**
```json
{
  "text-generation": {
    "primary": "new-org/new-model-7b",  // Changed
    "alternatives": ["old-primary-model", "other-model"]
  }
}
```

**As Alternative (adds to list):**
```json
{
  "text-generation": {
    "primary": "current-primary-model",
    "alternatives": [
      "existing-model-1",
      "existing-model-2",
      "new-org/new-model-7b"  // Added
    ]
  }
}
```

### 4. Validate JSON

Ensure valid JSON syntax:

```bash
python -m json.tool mid_range.json
```

### 5. Test Locally

```bash
# Run ModelForge
modelforge

# Check that model appears in recommendations
# Try training with the new model
```

### 6. Submit Pull Request

```bash
git checkout -b add-model-recommendations
git add ModelForge/model_configs/
git commit -m "feat: add new-model-7b to mid_range recommendations"
git push origin add-model-recommendations
```

Create PR on GitHub with description of:
- Model name and organization
- Why it's a good fit for this profile
- Test results (VRAM usage, quality, speed)

## Model Selection Criteria

### Quality Criteria

✅ **DO include models that:**
- Are publicly accessible on HuggingFace
- Have appropriate licenses (MIT, Apache 2.0, etc.)
- Perform well on relevant benchmarks
- Are actively maintained
- Have good documentation
- Work with standard Transformers library

❌ **DON'T include models that:**
- Are gated without clear access process
- Have restrictive licenses
- Are deprecated or unmaintained
- Require special dependencies
- Have known critical issues
- Are inappropriate for general use

### Hardware Compatibility

Verify VRAM usage:

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_id = "your-model/model-name"

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto"
)

# Check VRAM usage
import nvidia_smi
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print(f"VRAM used: {info.used / 1024**3:.2f} GB")
```

### Performance Testing

Test training speed:

```python
from time import time

start = time()
# Run sample training
duration = time() - start

print(f"Training time: {duration:.2f} seconds")
```

## Task-Specific Considerations

### Text Generation

- Focus on instruction-tuned models
- Prefer models with chat templates
- Consider context window size
- Test prompt following ability

**Good examples:**
- `meta-llama/Llama-3.2-3B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `microsoft/phi-3-mini-4k-instruct`

### Summarization

- Prefer models trained on summarization tasks
- Check ROUGE scores on standard benchmarks
- Consider domain (news, legal, medical, etc.)

**Good examples:**
- `facebook/bart-large-cnn`
- `google-t5/t5-base`
- `philschmid/bart-large-cnn-samsum`

### Question Answering

- Prefer models trained on QA datasets
- Check F1 and EM scores on SQuAD
- Consider retrieval-augmented use cases

**Good examples:**
- `deepset/roberta-base-squad2`
- `bert-large-uncased-whole-word-masking-finetuned-squad`

## Best Practices

### 1. Order Alternatives by Quality

List best alternatives first:

```json
{
  "alternatives": [
    "highest-quality-model",
    "good-quality-model",
    "acceptable-model"
  ]
}
```

### 2. Include Diversity

Provide options with different trade-offs:
- Speed vs quality
- Size vs performance
- General vs specialized

### 3. Keep Updated

- Remove deprecated models
- Add new state-of-the-art models
- Update based on community feedback

### 4. Document Changes

In PR description, include:
- Benchmark results
- VRAM measurements
- Training speed tests
- Quality comparisons

## Example PR Description

```markdown
## Add Qwen2-7B to mid_range recommendations

**Model**: Qwen/Qwen2-7B-Instruct

**Profile**: mid_range (8-16GB VRAM)

**Task**: text-generation

**Tests**:
- VRAM usage (4-bit): 8.2 GB ✅
- Training speed: ~25 tokens/sec on RTX 3060 ✅
- Quality: Excellent instruction following
- License: Apache 2.0 ✅

**Benchmarks**:
- MMLU: 68.2
- GSM8K: 76.5
- HumanEval: 52.1

**Why add**:
- Performs better than current alternatives
- Efficient memory usage
- Strong multilingual support
- Active community support

**Changes**:
- Added to mid_range.json as alternative
- Tested successful fine-tuning
```

## Troubleshooting

### Model Not Appearing

Check:
1. JSON syntax is valid
2. Profile name matches filename
3. Model ID is correct
4. Restart ModelForge

### Model Fails to Load

Check:
1. Model is publicly accessible
2. HuggingFace token has permissions
3. Model is compatible with Transformers
4. No gating issues

## Questions?

- See [Contributing Guide](contributing.md)
- Ask in [GitHub Discussions](https://github.com/forgeopus/modelforge/discussions)
- Check [FAQ](../troubleshooting/faq.md)

---

**Thank you for improving ModelForge's model recommendations!** 🤖
