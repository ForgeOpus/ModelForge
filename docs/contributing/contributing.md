# Contributing to ModelForge

Thank you for your interest in contributing to ModelForge! This guide will help you get started.

## Ways to Contribute

- 🐛 **Report bugs** via GitHub Issues
- 💡 **Suggest features** via GitHub Discussions
- 📝 **Improve documentation**
- 🔧 **Submit code improvements**
- 🤖 **Add model recommendations**
- 🏗️ **Add new providers or strategies**

## Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ModelForge.git
cd ModelForge
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## Code Style

### Python

We follow PEP 8 with some modifications:

- Maximum line length: 100 characters
- Use Black for formatting
- Use type hints where possible

**Format code:**
```bash
black ModelForge/
```

**Lint code:**
```bash
flake8 ModelForge/ --max-line-length=100
```

**Type check:**
```bash
mypy ModelForge/
```

### React/JavaScript

- Use ESLint and Prettier
- Follow Airbnb style guide
- Functional components with hooks

## Commit Messages

Follow conventional commits:

```
feat: add QLoRA strategy support
fix: resolve CUDA memory leak
docs: update Windows installation guide
refactor: improve provider abstraction
test: add tests for training service
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Pull Request Process

### 1. Before Submitting

- ✅ Code follows style guidelines
- ✅ All tests pass
- ✅ Documentation updated
- ✅ Commits are clean and descriptive

### 2. Submit PR

1. Push to your fork
2. Create PR on GitHub
3. Fill out PR template
4. Link related issues

### 3. Code Review

- Respond to feedback promptly
- Make requested changes
- Keep PR focused and small

### 4. After Merge

- Delete your branch
- Pull latest changes
- Celebrate! 🎉

## Adding Model Recommendations

See [Model Configurations Guide](model-configs.md) for detailed instructions.

**Quick steps:**

1. Add JSON file to `ModelForge/model_configs/`
2. Follow schema:

```json
{
  "profile": "mid_range",
  "tasks": {
    "text-generation": {
      "primary": "meta-llama/Llama-3.2-3B",
      "alternatives": ["mistralai/Mistral-7B-v0.1"]
    }
  }
}
```

## Adding a Provider

### 1. Create Provider Class

Create `ModelForge/providers/your_provider.py`:

```python
from typing import Any, Tuple
from ..logging_config import logger

class YourProvider:
    def load_model(
        self,
        model_id: str,
        model_class: str,
        quantization_config=None,
        device_map=None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Load model and tokenizer."""
        logger.info(f"Loading model {model_id} with Your Provider")
        # Implementation here
        return model, tokenizer
    
    def load_tokenizer(self, model_id: str, **kwargs) -> Any:
        """Load tokenizer."""
        # Implementation here
        return tokenizer
    
    def validate_model_access(self, model_id: str, model_class: str) -> bool:
        """Check if model is accessible."""
        # Implementation here
        return True
    
    def get_provider_name(self) -> str:
        """Return provider name."""
        return "your_provider"
```

### 2. Register in Factory

Edit `ModelForge/providers/provider_factory.py`:

```python
from .your_provider import YourProvider

class ProviderFactory:
    _providers = {
        "huggingface": HuggingFaceProvider,
        "unsloth": UnslothProvider,
        "your_provider": YourProvider,  # Add here
    }
```

### 3. Add Tests

Create `tests/test_your_provider.py`:

```python
def test_your_provider_loads():
    provider = YourProvider()
    assert provider.get_provider_name() == "your_provider"
```

### 4. Update Documentation

Add docs to `docs/providers/your_provider.md`

## Adding a Strategy

### 1. Create Strategy Class

Create `ModelForge/strategies/your_strategy.py`:

```python
from typing import Any, Dict
from ..logging_config import logger

class YourStrategy:
    def get_strategy_name(self) -> str:
        return "your_strategy"
    
    def prepare_model(self, model: Any, config: Dict) -> Any:
        """Prepare model for training."""
        logger.info("Preparing model with Your Strategy")
        # Implementation here
        return model
    
    def prepare_dataset(self, dataset: Any, tokenizer: Any, config: Dict) -> Any:
        """Prepare dataset for strategy."""
        # Implementation here
        return dataset
    
    def create_trainer(
        self,
        model,
        train_dataset,
        eval_dataset,
        tokenizer,
        config,
        callbacks
    ):
        """Create trainer instance."""
        # Implementation here
        return trainer
    
    def get_required_dataset_fields(self):
        """Return required dataset fields."""
        return ["input", "output"]
```

### 2. Register in Factory

Edit `ModelForge/strategies/strategy_factory.py`:

```python
from .your_strategy import YourStrategy

class StrategyFactory:
    _strategies = {
        "sft": SFTStrategy,
        "qlora": QLoRAStrategy,
        "rlhf": RLHFStrategy,
        "dpo": DPOStrategy,
        "your_strategy": YourStrategy,  # Add here
    }
```

### 3. Add Tests and Documentation

Similar to providers above.

## Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test

```bash
pytest tests/test_providers.py
```

### Run with Coverage

```bash
pytest --cov=ModelForge --cov-report=html
```

### Manual Testing

```bash
# Start development server
modelforge cli

# Test in browser
open http://localhost:8000
```

## Documentation

### Building Docs Locally

Documentation is in Markdown. Preview with any Markdown viewer.

### Adding Documentation

1. Create `.md` file in appropriate `docs/` subdirectory
2. Add to index in `docs/README.md`
3. Use relative links: `[text](../other/file.md)`

### Documentation Style

- Use clear, simple language
- Include code examples
- Add links to related docs
- Use emoji sparingly (✅❌⚠️)

## Issue Guidelines

### Reporting Bugs

Include:
- ModelForge version
- Python version
- OS and platform
- GPU model and VRAM
- Steps to reproduce
- Error messages/logs
- Expected vs actual behavior

### Requesting Features

Include:
- Use case description
- Expected behavior
- Why it's useful
- Implementation ideas (optional)

## Community Guidelines

- Be respectful and professional
- Help others learn and grow
- Give constructive feedback
- Follow Code of Conduct

## Recognition

Contributors are recognized in:
- GitHub Contributors page
- Release notes
- README (for significant contributions)

## Questions?

- Check [FAQ](../troubleshooting/faq.md)
- Ask in [GitHub Discussions](https://github.com/forgeopus/modelforge/discussions)
- Read [Architecture Guide](architecture.md)

---

**Thank you for contributing to ModelForge!** 🎉
