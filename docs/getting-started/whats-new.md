# What's New in ModelForge

## What's New in v2.1.3

### Interactive CLI Wizard

ModelForge now ships with a terminal-based interactive wizard for users who prefer a headless or SSH workflow:

- **`modelforge cli`** — launches the step-by-step interactive wizard in the terminal
- **`modelforge-nb`** — notebook-friendly entry point for Jupyter environments
- **`modelforge`** (no args) — still starts the web UI as before

### Optional Quantization

bitsandbytes has been moved to an optional `[quantization]` extra. Install it with:

```bash
pip install modelforge-finetuning[quantization]
```

If not installed, ModelForge will gracefully fall back and disable quantization features. This makes the base install lighter and avoids build issues on systems where bitsandbytes is hard to compile.

### Schema Validation

New `@model_validator` rules enforce valid configuration combinations at startup:

- **DPO/RLHF** strategies require `"task": "text-generation"` — using them with summarization or QA tasks will raise a clear validation error
- **Unsloth** provider requires `"task": "text-generation"` — encoder-decoder models are not supported

This prevents cryptic training errors by catching incompatible configurations early.

### Legacy Tuner Removal

The following legacy tuners have been removed in v2.1:

- `CausalLLMTuner`
- `Finetuner`
- `QuestionAnsweringTuner`
- `Seq2SeqLMTuner`

Use the provider/strategy pattern instead (e.g., `"provider": "huggingface"`, `"strategy": "sft"`).

### RLHF Clarification

The RLHF strategy now clearly uses **DPO (Direct Preference Optimization)** internally — no PPO, no reward model needed. Both RLHF and DPO strategies use TRL's `DPOTrainer` with different default hyperparameters.

### Bug Fixes

- Fixed memory management and model offload directory handling
- Fixed TensorBoard auto-open on training start
- Fixed perplexity evaluation for text generation tasks
- Fixed dataset format validation errors
- Fixed incompatible wizard selections being passed to the trainer
- CLI subcommand handling improved with clearer usage instructions when invoked incorrectly

---

# What's New in ModelForge v2.0

ModelForge v2.0 represents a complete architectural overhaul that maintains 100% backward compatibility while introducing powerful new features and improvements.

## 🚀 Major New Features

### 1. Multiple Provider Support

**Before**: Only HuggingFace models were supported.

**Now**: Choose from multiple model providers:

- **HuggingFace** - Standard models from HuggingFace Hub
- **Unsloth** - 2x faster training with optimized kernels (NEW!)

```json
{
  "provider": "unsloth",
  "model_name": "meta-llama/Llama-3.2-3B",
  ...
}
```

> **Note**: Unsloth requires WSL or Docker on Windows. See [Windows Installation](../installation/windows.md).

### 2. Multiple Training Strategies

**Before**: Only standard supervised fine-tuning (SFT).

**Now**: Choose from advanced training strategies:

- **SFT** - Standard supervised fine-tuning with LoRA
- **QLoRA** - Memory-efficient quantized LoRA (NEW!)
- **RLHF** - Reinforcement Learning from Human Feedback (NEW!)
- **DPO** - Direct Preference Optimization (NEW!)

```json
{
  "strategy": "qlora",
  "model_name": "meta-llama/Llama-3.2-7B",
  ...
}
```

### 3. Evaluation System

**NEW**: Automatic train/validation splits with task-specific metrics!

- Automatic dataset splitting (configurable ratio)
- Real-time evaluation during training
- Task-specific metrics:
  - **Text Generation**: Perplexity, loss
  - **Summarization**: ROUGE-1, ROUGE-2, ROUGE-L
  - **Question Answering**: Exact Match, F1 score

```json
{
  "eval_split": 0.2,      // 20% for validation
  "eval_steps": 100,      // Evaluate every 100 steps
  ...
}
```

### 4. Better Error Handling

- Structured exception hierarchy
- Clear, actionable error messages
- Comprehensive logging throughout
- No more silent failures!

## 🏗️ Architecture Improvements

### Service Layer with Dependency Injection

**Before**: Global singleton pattern with tight coupling.

**Now**: Clean service layer with dependency injection:

- `TrainingService` - Training orchestration
- `ModelService` - Model management
- `HardwareService` - Hardware detection

**Benefits**:
- No global state
- Easy to test
- Clear separation of concerns

### Provider Abstraction Layer

**Before**: 15+ files needed modification to add a new provider.

**Now**: Just 2 files to add a provider!

- Protocol-based design
- Factory pattern
- Plug-and-play architecture

### Strategy Pattern for Training

**Before**: 10+ files needed modification to add a new strategy.

**Now**: Just 2 files to add a strategy!

- Clean strategy interface
- Isolated strategy logic
- Easy to extend

### Database with SQLAlchemy

**Before**: Direct SQLite access, opened/closed on every operation.

**Now**: SQLAlchemy ORM with connection pooling:

- 10 connections in pool
- 20 max overflow
- Proper session management
- Easy migration to PostgreSQL

## 📊 Performance Improvements

| Feature | Before | After | Improvement |
|---------|--------|-------|-------------|
| Database Ops | Open/close each | Pooling | ~10x faster |
| Training Speed | Standard | Unsloth option | 2x faster |
| Memory Usage | Standard | QLoRA option | 30-50% reduction |

## 🧹 Code Quality Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Code Duplication | 150+ lines | 0 lines | 100% reduction |
| Router LOC | 563 lines | ~250 lines | 56% reduction |
| Singleton Usage | 1 global | 0 | Eliminated |
| Test Coverage | 0% | Ready | Testable architecture |

## 🎯 User-Facing Changes

### No Breaking Changes!

All existing workflows continue to work exactly as before. The v2.0 features are **opt-in**.

### New Optional Configuration Fields

```json
{
  "provider": "huggingface",  // NEW: Choose provider (default: huggingface)
  "strategy": "sft",          // NEW: Choose strategy (default: sft)
  "eval_split": 0.2,          // NEW: Validation split ratio
  "eval_steps": 100,          // NEW: Evaluation frequency
  ... // All existing fields still work
}
```

### New API Endpoints

- `GET /api/info` - System information (providers, strategies, tasks)
- `GET /api/health` - Health check endpoint

## 📝 What Stayed the Same

- **All existing API endpoints** work exactly as before
- **Dataset format** remains the same
- **UI workflow** is unchanged
- **Model outputs** are identical
- **Installation process** is the same

## 🔧 For Developers

### Easier Extensibility

**Adding a New Provider** (e.g., vLLM):
```python
# 1. Create provider class (1 file)
class VLLMProvider:
    def load_model(self, ...): ...
    def get_provider_name(self): return "vllm"

# 2. Register in factory (1 line)
ProviderFactory._providers["vllm"] = VLLMProvider
```

**Adding a New Strategy** (e.g., ORPO):
```python
# 1. Create strategy class (1 file)
class ORPOStrategy:
    def prepare_model(self, ...): ...
    def get_strategy_name(self): return "orpo"

# 2. Register in factory (1 line)
StrategyFactory._strategies["orpo"] = ORPOStrategy
```

### Better Testing

With dependency injection, everything is now testable:

```python
def test_training_service():
    mock_db = MagicMock(spec=DatabaseManager)
    service = TrainingService(mock_db)
    assert service.get_training_status()["status"] == "idle"
```

## 🚦 Migration Guide

### For End Users

**No action required!** Continue using ModelForge as before, or:

**Try new features** (optional):
1. **Faster training**: Set `"provider": "unsloth"`
2. **Save memory**: Set `"strategy": "qlora"`
3. **Enable evaluation**: Set `"eval_split": 0.2`

### For Contributors

**Old pattern**:
```python
from globals_instance import global_manager
db = global_manager.db_manager
```

**New pattern**:
```python
from dependencies import get_db_manager
db = get_db_manager()
```

**In routers**:
```python
from fastapi import Depends
from dependencies import get_model_service

@router.get("/models")
async def get_models(service: ModelService = Depends(get_model_service)):
    return service.get_all_models()
```

## 🎓 Learn More

- **[Provider Documentation](../providers/overview.md)** - Deep dive into providers
- **[Strategy Documentation](../strategies/overview.md)** - Understanding strategies
- **[Architecture Overview](../contributing/architecture.md)** - How it all works
- **[Configuration Guide](../configuration/configuration-guide.md)** - All config options

## 🙏 Thank You

This refactoring was a massive undertaking that improves every aspect of ModelForge while maintaining the user experience. The architecture is now ready for scale and community-driven growth.

**ModelForge v2.0: Modular, Extensible, Maintainable** 🚀
