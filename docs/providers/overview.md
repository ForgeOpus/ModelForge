# Provider Overview

ModelForge v3.0.0 introduces a modular provider system that allows you to choose different model loading and optimization backends.

## What Are Providers?

**Providers** are backends that handle:
- Loading models from model hubs or local storage
- Applying optimizations and patches
- Managing tokenizers
- Setting up device mappings

Different providers offer different trade-offs in terms of speed, memory usage, and compatibility.

## Available Providers

| Provider | Speed | Memory | Platform Support | Description |
|----------|-------|--------|------------------|-------------|
| **[HuggingFace](huggingface.md)** | 1x (baseline) | Standard | Windows/Linux | Standard HuggingFace Transformers |
| **[Unsloth](unsloth.md)** | 2x faster | 20% less | Linux/WSL/Docker only | Optimized training kernels |

## Choosing a Provider

### Use HuggingFace When:

✅ Running on native Windows  
✅ Maximum compatibility needed  
✅ Using models not supported by Unsloth  
✅ Debugging or development  
✅ First time using ModelForge  

### Use Unsloth When:

✅ Running on Linux or WSL  
✅ Training large models (7B+ parameters)  
✅ Need faster training times  
✅ Have limited VRAM  
✅ Using supported architectures (Llama, Mistral, etc.)  

## How to Specify a Provider

### In Configuration File

```json
{
  "provider": "huggingface",
  "model_name": "meta-llama/Llama-3.2-3B",
  "task": "text-generation",
  ...
}
```

### Via UI

1. Go to Training tab
2. Select **Provider** dropdown
3. Choose `huggingface` or `unsloth`

### Via API

```bash
curl -X POST http://localhost:8000/api/start_training \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "unsloth",
    "model_name": "meta-llama/Llama-3.2-3B",
    ...
  }'
```

## Provider Architecture

### How Providers Work

```
Training Request
       ↓
Provider Factory
       ↓
  Load Provider ← (HuggingFace / Unsloth / Custom)
       ↓
  Load Model
       ↓
  Load Tokenizer
       ↓
  Validate Access
       ↓
Return (model, tokenizer)
```

### Provider Protocol

All providers implement the same interface:

```python
class ModelProvider(Protocol):
    def load_model(self, model_id: str, model_class: str, 
                   quantization_config, device_map, **kwargs) -> Model
    
    def load_tokenizer(self, model_id: str, **kwargs) -> Tokenizer
    
    def validate_model_access(self, model_id: str, model_class: str) -> bool
    
    def get_provider_name(self) -> str
```

This ensures all providers work seamlessly with ModelForge.

## Provider Compatibility

### Model Architecture Support

| Architecture | HuggingFace | Unsloth |
|--------------|-------------|---------|
| Llama (1/2/3) | ✅ | ✅ |
| Mistral | ✅ | ✅ |
| Qwen | ✅ | ✅ |
| Gemma | ✅ | ✅ |
| Phi | ✅ | ✅ |
| BART | ✅ | ❌ |
| T5 | ✅ | ❌ |
| BERT-based | ✅ | ❌ |

### Task Support

| Task | HuggingFace | Unsloth |
|------|-------------|---------|
| Text Generation | ✅ | ✅ |
| Summarization | ✅ | ⚠️ Limited |
| Question Answering | ✅ | ⚠️ Limited |

### Strategy Support

| Strategy | HuggingFace | Unsloth |
|----------|-------------|---------|
| SFT | ✅ | ✅ |
| QLoRA | ✅ | ✅ |
| RLHF | ✅ | ⚠️ Experimental |
| DPO | ✅ | ⚠️ Experimental |

## Performance Comparison

### Training Speed

Based on Llama-3.2-3B with 1000 examples:

| Provider | Time | Speedup |
|----------|------|---------|
| HuggingFace | 45 min | 1x |
| Unsloth | 22 min | 2x |

### Memory Usage

Based on Llama-3.2-7B with batch_size=4:

| Provider | VRAM | Reduction |
|----------|------|-----------|
| HuggingFace | 16 GB | - |
| Unsloth | 12.8 GB | 20% |

> Results may vary based on hardware and configuration.

## Platform Requirements

### HuggingFace Provider

**Supported Platforms:**
- ✅ Windows 10/11
- ✅ Linux (all distributions)
- ✅ WSL 2
- ✅ Docker

**Requirements:**
- Python 3.11
- PyTorch 2.0+
- transformers 4.40+

### Unsloth Provider

**Supported Platforms:**
- ✅ Linux (Ubuntu 20.04+, Debian 11+)
- ✅ WSL 2 on Windows
- ✅ Docker with NVIDIA runtime
- ❌ Native Windows

**Requirements:**
- Python 3.11
- PyTorch 2.0+ with CUDA
- transformers 4.40+
- unsloth package
- NVIDIA GPU with CUDA 11.8+

## Troubleshooting

### Provider Not Found

**Error**: `ProviderError: Unknown provider 'xyz'`

**Solution**: Check spelling. Valid providers: `huggingface`, `unsloth`

### Unsloth Not Available on Windows

**Error**: `ProviderError: Unsloth is not installed`

**Solution**: Use WSL or Docker. See [Windows Installation](../installation/windows.md).

### Provider Import Error

**Error**: `ModuleNotFoundError: No module named 'unsloth'`

**Solution**: Install the provider:
```bash
pip install unsloth
```

## Adding Custom Providers

Want to add a new provider? See [Custom Providers Guide](custom-providers.md).

## Next Steps

- **[HuggingFace Provider](huggingface.md)** - Learn about the standard provider
- **[Unsloth Provider](unsloth.md)** - Enable 2x faster training
- **[Custom Providers](custom-providers.md)** - Add your own provider
- **[Training Strategies](../strategies/overview.md)** - Choose a training strategy

---

**Providers make ModelForge flexible and extensible!** Choose the right one for your needs.
