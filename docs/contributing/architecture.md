# ModelForge v3 Architecture

Technical overview of ModelForge's modular architecture.

## Architecture Overview

ModelForge v3 uses a clean, modular architecture based on SOLID principles:

```
┌─────────────────────────────────────────────────────┐
│                   Web Interface (React)              │
└────────────────────┬────────────────────────────────┘
                     │
                     │ HTTP/REST
                     ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Application Layer               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │  Finetuning  │  │    Models    │  │Playground│  │
│  │    Router    │  │    Router    │  │  Router  │  │
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  │
└─────────┼──────────────────┼───────────────┼────────┘
          │                  │               │
          │ Dependency Injection (FastAPI)   │
          ▼                  ▼               ▼
┌─────────────────────────────────────────────────────┐
│                  Service Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │   Training   │  │    Model     │  │ Hardware │  │
│  │   Service    │  │   Service    │  │ Service  │  │
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  │
└─────────┼──────────────────┼───────────────┼────────┘
          │                  │               │
          ▼                  ▼               ▼
┌─────────────────────────────────────────────────────┐
│             Business Logic Layer                     │
│                                                      │
│  ┌─────────────────┐      ┌────────────────────┐   │
│  │  Provider       │      │  Strategy          │   │
│  │  Factory        │      │  Factory           │   │
│  │                 │      │                    │   │
│  │ ┌─────────────┐ │      │ ┌────────────────┐ │   │
│  │ │ HuggingFace │ │      │ │      SFT       │ │   │
│  │ │  Provider   │ │      │ │   Strategy     │ │   │
│  │ └─────────────┘ │      │ └────────────────┘ │   │
│  │ ┌─────────────┐ │      │ ┌────────────────┐ │   │
│  │ │   Unsloth   │ │      │ │     QLoRA      │ │   │
│  │ │  Provider   │ │      │ │   Strategy     │ │   │
│  │ └─────────────┘ │      │ └────────────────┘ │   │
│  └─────────────────┘      └────────────────────┘   │
│                                                      │
│  ┌─────────────────┐      ┌────────────────────┐   │
│  │  Evaluation     │      │  Quantization      │   │
│  │  System         │      │  Factory           │   │
│  └─────────────────┘      └────────────────────┘   │
└──────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────────────────────────────────────┐
│                  Data Layer                          │
│  ┌──────────────────┐      ┌────────────────────┐  │
│  │  Database        │      │  File Manager      │  │
│  │  Manager         │      │                    │  │
│  │  (SQLAlchemy)    │      │  - Datasets        │  │
│  │                  │      │  - Checkpoints     │  │
│  │  - Models        │      │  - Logs            │  │
│  │  - Training      │      └────────────────────┘  │
│  └──────────────────┘                               │
└──────────────────────────────────────────────────────┘
```

## Core Components

### 1. Routers (API Layer)

**Location**: `ModelForge/routers/`

**Responsibility**: HTTP request handling

**Files**:
- `finetuning_router.py` - Training endpoints
- `models_router.py` - Model management
- `playground_router.py` - Inference testing
- `hub_management_router.py` - Model hub operations

**Pattern**: Thin controllers, delegate to services

### 2. Services (Business Logic)

**Location**: `ModelForge/services/`

**Responsibility**: Core business logic

**Files**:
- `training_service.py` - Training orchestration
- `model_service.py` - Model CRUD operations
- `hardware_service.py` - Hardware detection

**Pattern**: Service layer with dependency injection

### 3. Providers (Model Loading)

**Location**: `ModelForge/providers/`

**Responsibility**: Model and tokenizer loading

**Files**:
- `__init__.py` - Provider protocol
- `huggingface_provider.py` - HuggingFace implementation
- `unsloth_provider.py` - Unsloth implementation
- `provider_factory.py` - Provider creation

**Pattern**: Protocol + Factory

### 4. Strategies (Training Algorithms)

**Location**: `ModelForge/strategies/`

**Responsibility**: Training algorithm implementation

**Files**:
- `__init__.py` - Strategy protocol
- `sft_strategy.py` - Supervised fine-tuning
- `qlora_strategy.py` - Quantized LoRA
- `rlhf_strategy.py` - RLHF
- `dpo_strategy.py` - DPO
- `strategy_factory.py` - Strategy creation

**Pattern**: Strategy + Factory

### 5. Database Layer

**Location**: `ModelForge/database/`

**Responsibility**: Data persistence

**Files**:
- `models.py` - SQLAlchemy models
- `database_manager.py` - DB operations

**Pattern**: Repository with ORM

### 6. Evaluation System

**Location**: `ModelForge/evaluation/`

**Responsibility**: Training evaluation

**Files**:
- `metrics.py` - Task-specific metrics
- `dataset_validator.py` - Dataset validation

## Design Patterns

### Dependency Injection

**Implementation**: FastAPI's `Depends()`

**Example**:
```python
from fastapi import APIRouter, Depends
from ..dependencies import get_training_service

router = APIRouter()

@router.post("/start_training")
async def start_training(
    config: TrainingConfig,
    service: TrainingService = Depends(get_training_service),
):
    return service.train_model(config.model_dump())
```

### Factory Pattern

**Used for**: Providers and Strategies

**Example**:
```python
class ProviderFactory:
    _providers = {
        "huggingface": HuggingFaceProvider,
        "unsloth": UnslothProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_name: str):
        provider_class = cls._providers.get(provider_name)
        if not provider_class:
            raise ProviderError(f"Unknown provider: {provider_name}")
        return provider_class()
```

### Strategy Pattern

**Used for**: Training algorithms

**Example**:
```python
class TrainingStrategy(Protocol):
    def prepare_model(self, model, config): ...
    def prepare_dataset(self, dataset, tokenizer, config): ...
    def create_trainer(self, model, dataset, ...): ...
```

### Protocol (Interface)

**Used for**: Defining contracts

**Example**:
```python
from typing import Protocol

class ModelProvider(Protocol):
    def load_model(self, model_id: str, ...): ...
    def load_tokenizer(self, model_id: str, ...): ...
    def validate_model_access(self, model_id: str, ...): ...
    def get_provider_name(self) -> str: ...
```

## Data Flow

### Training Request Flow

1. **User** submits training request via UI
2. **React Frontend** sends POST to `/api/start_training`
3. **FastAPI Router** receives request, validates with Pydantic
4. **Router** injects `TrainingService` via dependency
5. **TrainingService** orchestrates:
   - Validates dataset
   - Creates provider from `ProviderFactory`
   - Loads model via provider
   - Creates strategy from `StrategyFactory`
   - Prepares model and dataset via strategy
   - Creates trainer and starts training
6. **Training** runs with callbacks for progress
7. **Results** saved to database and file system
8. **Response** returned to user

### Model Loading Flow

```
User Request
    ↓
ProviderFactory.create_provider(provider_name)
    ↓
Provider.load_model(model_id, config)
    ↓
Provider-specific implementation
    ↓
Return (model, tokenizer)
```

## Extension Points

### Adding a Provider

1. Create class implementing `ModelProvider` protocol
2. Register in `ProviderFactory._providers`
3. That's it! No other changes needed.

### Adding a Strategy

1. Create class implementing `TrainingStrategy` protocol
2. Register in `StrategyFactory._strategies`
3. That's it! No other changes needed.

### Adding a Task

1. Add task-specific formatter in `services/training_service.py`
2. Add metrics in `evaluation/metrics.py`
3. Update schema validation

## Error Handling

### Exception Hierarchy

```
ModelForgeException (base)
├── ProviderError
├── StrategyError
├── DatasetValidationError
├── TrainingError
├── ConfigurationError
├── HardwareError
└── DatabaseError
```

### Error Handler

All exceptions caught by FastAPI error handlers and converted to appropriate HTTP responses.

## Configuration Management

### Pydantic Schemas

**Location**: `ModelForge/schemas/`

Validation and serialization of configuration.

### Environment Variables

- `HUGGINGFACE_TOKEN` - HuggingFace API token
- `MODELFORGE_DB_PATH` - Custom database path
- `MODELFORGE_DISABLE_TENSORBOARD` - Disable TensorBoard

## Testing Strategy

### Unit Tests

Test individual components in isolation:
```python
def test_provider_factory():
    provider = ProviderFactory.create_provider("huggingface")
    assert provider.get_provider_name() == "huggingface"
```

### Integration Tests

Test component interactions:
```python
def test_training_flow():
    service = TrainingService(mock_db, mock_file_manager)
    result = service.train_model(config)
    assert result["status"] == "success"
```

### Mocking

Use dependency injection for easy mocking:
```python
mock_db = MagicMock(spec=DatabaseManager)
service = TrainingService(mock_db, file_manager)
```

## Performance Considerations

### Connection Pooling

SQLAlchemy connection pool:
- Pool size: 10
- Max overflow: 20
- Recycle: 3600 seconds

### Lazy Loading

Models and datasets loaded only when needed.

### Gradient Checkpointing

Reduces memory at cost of compute.

### Mixed Precision

bf16/fp16 for faster training on modern GPUs.

## Security

### Input Validation

All inputs validated via Pydantic schemas.

### SQL Injection Prevention

SQLAlchemy ORM prevents SQL injection.

### File Access

File paths validated and sandboxed.

### Token Security

HuggingFace tokens stored in environment, never in code.

## Scalability

### Current Limitations

- Single-GPU training
- Single-process server
- SQLite database

### Future Improvements

- Multi-GPU support (already structured for it)
- Distributed training
- PostgreSQL for production
- Redis for caching
- Kubernetes deployment

## Code Quality Metrics

| Metric | Value |
|--------|-------|
| Code Duplication | 0% |
| Cyclomatic Complexity | Low (< 10 per function) |
| Test Coverage | (To be added) |
| Type Hints | Extensive |
| Documentation | Comprehensive |

## Contributing

See [Contributing Guide](contributing.md) for:
- Code style guidelines
- Testing requirements
- PR process

---

**Understanding the architecture makes contributing easy!** Read the code in `ModelForge/` to see it in action.
