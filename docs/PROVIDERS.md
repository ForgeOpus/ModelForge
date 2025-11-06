# Provider System Documentation

ModelForge supports multiple fine-tuning providers, allowing you to choose the best backend for your specific needs. This document explains the provider system architecture and how to use it.

## Available Providers

### HuggingFace Provider (Default)

**Status:** Always available (built-in)

**Best for:**
- All task types (text-generation, summarization, extractive-question-answering)
- Maximum compatibility with HuggingFace models
- Full control over quantization (4-bit and 8-bit)
- Extensive hyperparameter customization

**Key Features:**
- Supports all three task types
- Battle-tested and widely adopted
- Compatible with all HuggingFace models
- Full quantization control (4-bit and 8-bit)
- Extensive hyperparameter options

### Unsloth Provider (Optional)

**Status:** Optional dependency (requires installation)

**Best for:**
- Text-generation tasks
- Resource-constrained environments
- Faster training with limited GPU memory

**Key Features:**
- 2x faster training speed
- 60% less memory consumption
- Optimized 4-bit quantization
- Efficient gradient checkpointing
- Focus on text-generation tasks

**Installation:**
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

Or:
```bash
pip install modelforge-finetuning[unsloth]
```

**Current Limitations:**
- Only supports text-generation tasks
- 8-bit quantization not implemented (uses 4-bit)
- Requires process restart after installation

## Using the Provider System

### 1. Provider Selection

When you start a new fine-tuning session, you'll first see the provider selection page where you can:
- View available providers
- See which providers are installed
- Review the benefits of each provider
- Select your preferred provider

### 2. Provider Flow

The workflow follows these steps:
1. **Select Provider** → Choose between HuggingFace or Unsloth
2. **Detect Hardware** → System analyzes your hardware capabilities
3. **Configure Settings** → Adjust hyperparameters
4. **Upload Dataset** → Provide training data
5. **Start Training** → Begin fine-tuning with selected provider

### 3. Provider State Management

The selected provider is preserved across all pages in the workflow:
- Provider choice stored in application state
- Automatically included in training configuration
- Persists through hardware detection and settings pages
- Saved in model configuration file for later reference

## Backend Architecture

### Provider Abstraction Layer

The provider system uses an abstract base class pattern:

```
ModelForge/utilities/finetuning/providers/
├── __init__.py              # Provider registration
├── base_provider.py         # Abstract base class
├── provider_registry.py     # Factory and registry
├── provider_adapter.py      # Router adapter
├── huggingface_provider.py  # HuggingFace implementation
└── unsloth_provider.py      # Unsloth implementation
```

### Key Components

1. **BaseProvider** - Abstract interface all providers must implement
2. **ProviderRegistry** - Manages provider registration and availability
3. **ProviderAdapter** - Bridges router and provider implementations
4. **Provider Implementations** - Concrete provider classes

### Provider Capabilities

Each provider must implement:
- `get_provider_name()` - Unique identifier
- `is_available()` - Dependency check
- `validate_hyperparameters()` - Parameter validation
- `format_dataset()` - Dataset preprocessing
- `load_model()` - Model loading
- `prepare_for_training()` - Apply LoRA/PEFT
- `train()` - Execute training
- `save_model()` - Save fine-tuned model
- `get_model_class_name()` - Model class for config

## Configuration Files

Models fine-tuned with the provider system include provider metadata in `modelforge_config.json`:

```json
{
  "model_class": "AutoPeftModelForCausalLM",
  "pipeline_task": "text-generation",
  "provider": "unsloth"
}
```

**Backward Compatibility:** If `provider` field is missing, it defaults to `"huggingface"` for legacy models.

## API Endpoints

### Get Available Providers

The `/finetune/detect` endpoint returns provider information:

```json
{
  "providers": [
    {
      "name": "huggingface",
      "available": true,
      "supported_tasks": ["text-generation", "summarization", "extractive-question-answering"]
    },
    {
      "name": "unsloth",
      "available": false,
      "supported_tasks": ["text-generation"]
    }
  ],
  "available_providers": ["huggingface"]
}
```

### Provider Validation

Settings include provider field with validation:

```python
class SettingsFormData(BaseModel):
    provider: str = "huggingface"  # Default for backward compatibility
    
    @field_validator("provider")
    def validate_provider(cls, provider):
        available = ProviderRegistry.list_available()
        if provider not in available:
            raise ValueError(f"Provider '{provider}' is not available")
        return provider
```

## Adding New Providers

To add a new provider:

1. **Create provider class** inheriting from `BaseProvider`
2. **Implement all abstract methods**
3. **Add soft dependency check** in `is_available()`
4. **Register in `__init__.py`**
5. **Update frontend** with provider info

Example:

```python
from .base_provider import BaseProvider

class NewProvider(BaseProvider):
    @staticmethod
    def get_provider_name() -> str:
        return "new_provider"
    
    @staticmethod
    def is_available() -> bool:
        try:
            import new_provider_lib
            return True
        except ImportError:
            return False
    
    # ... implement other methods
```

Register in `__init__.py`:
```python
try:
    from .new_provider import NewProvider
    ProviderRegistry.register(NewProvider)
except ImportError:
    print("NewProvider not available")
```

## Troubleshooting

### Provider Not Available

**Issue:** Unsloth shows as "Not Installed" in provider selection.

**Solution:** Install Unsloth and restart the application:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
modelforge run
```

### Provider Validation Error

**Issue:** Error when starting training: "Provider 'X' is not available"

**Solution:** Either install the provider or select a different available provider.

### Backward Compatibility

**Issue:** Older models don't have provider field.

**Solution:** System automatically defaults to "huggingface" for models without provider metadata.

## Best Practices

1. **Choose the right provider:**
   - HuggingFace for maximum compatibility
   - Unsloth for text-generation optimization

2. **Check availability:**
   - Install optional providers before using
   - Verify in provider selection page

3. **Restart after installation:**
   - Provider availability detected at startup
   - Restart app after installing new providers

4. **Document provider choice:**
   - Provider saved in model config
   - Helps with reproducibility

## Future Enhancements

Potential future improvements:
- Additional provider support (e.g., other optimization frameworks)
- Strategy layer for advanced training techniques
- Dynamic capability enumeration
- Enhanced dataset validation per provider
- Provider-specific UI optimizations
