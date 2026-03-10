# Development Setup

Guide for setting up a ModelForge development environment.

## Overview

This guide helps you set up a local development environment for contributing to ModelForge.

## Prerequisites

### Required Software

- **Python 3.11.x** (Python 3.12 not yet supported)
- **Git** for version control
- **NVIDIA GPU** with CUDA support (for testing)
- **CUDA Toolkit** 11.8 or 12.x
- **Node.js 18+** and **npm** (for frontend development)

### Recommended Tools

- **VS Code** or **PyCharm** for IDE
- **Docker** (optional, for isolated testing)
- **WSL 2** (Windows users)

---

## Initial Setup

### 1. Fork and Clone Repository

```bash
# Fork the repository on GitHub first

# Clone your fork
git clone https://github.com/YOUR-USERNAME/ModelForge.git
cd ModelForge

# Add upstream remote
git remote add upstream https://github.com/forgeopus/modelforge.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install ModelForge in editable mode with all extras
pip install -e ".[cli,quantization]"

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy pre-commit
```

### 4. Install Unsloth (Optional, Linux/WSL only)

```bash
pip install unsloth
```

### 5. Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files
```

---

## Frontend Development Setup

ModelForge has a React frontend in the `Frontend` directory.

### 1. Navigate to Frontend Directory

```bash
cd Frontend
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Run Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`.

### 4. Build for Production

```bash
npm run build
```

---

## Running ModelForge in Development

### Start Backend Server

```bash
# From repository root
modelforge          # Launch web UI
modelforge cli      # Launch CLI wizard (alternative)
```

Or run directly with Python:

```bash
python -m ModelForge.cli run
```

### Start Frontend Development Server

```bash
cd Frontend
npm run dev
```

### Access Application

- Backend API: `http://localhost:8000`
- Frontend Dev Server: `http://localhost:5173`
- API Documentation: `http://localhost:8000/docs`

---

## Project Structure

```
ModelForge/
├── ModelForge/                 # Backend Python package
│   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   ├── cli.py                  # CLI entry point
│   ├── providers/              # Model providers
│   │   ├── huggingface_provider.py
│   │   ├── unsloth_provider.py
│   │   └── provider_factory.py
│   ├── strategies/             # Training strategies
│   │   ├── sft_strategy.py
│   │   ├── qlora_strategy.py
│   │   ├── rlhf_strategy.py
│   │   ├── dpo_strategy.py
│   │   └── strategy_factory.py
│   ├── routers/                # API routers
│   │   ├── finetuning_router.py
│   │   ├── models_router.py
│   │   ├── playground_router.py
│   │   └── hub_management_router.py
│   ├── services/               # Business logic
│   │   ├── training_service.py
│   │   ├── model_service.py
│   │   └── hardware_service.py
│   ├── schemas/                # Pydantic schemas
│   │   └── training_schemas.py
│   ├── utilities/              # Utility functions
│   │   ├── hardware_detection/
│   │   └── configs/
│   ├── formatters/             # Dataset formatters
│   ├── evaluation/             # Evaluation metrics
│   ├── notebook_cli/           # Notebook/CLI wizard
│   └── model_configs/          # Hardware profile configs
├── Frontend/                   # React frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── docs/                       # Documentation
├── tests/                      # Test suite
├── pyproject.toml              # Python project config
├── README.md
└── requirements.txt
```

---

## Development Workflow

### 1. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Edit code following the coding standards (see below).

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ModelForge --cov-report=html

# Run specific test file
pytest tests/test_providers.py

# Run specific test
pytest tests/test_providers.py::test_huggingface_provider
```

### 4. Format Code

```bash
# Format with black
black ModelForge/ tests/

# Check with flake8
flake8 ModelForge/ tests/

# Type check with mypy
mypy ModelForge/
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add new feature"
```

**Commit Message Format**:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Tests
- `chore:` Maintenance

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** for function signatures
- Max line length: **88 characters** (Black default)
- Use **docstrings** for all public functions and classes

### Example

```python
from typing import Dict, Any

def load_model(
    model_id: str,
    device_map: str = "auto",
    **kwargs: Dict[str, Any]
) -> Any:
    """
    Load a model from HuggingFace Hub.
    
    Args:
        model_id: HuggingFace model ID
        device_map: Device mapping strategy
        **kwargs: Additional arguments
        
    Returns:
        Loaded model instance
        
    Raises:
        ModelAccessError: If model cannot be accessed
    """
    # Implementation
    pass
```

### JavaScript/React Style

- Use **ESLint** and **Prettier**
- Use **functional components** and **hooks**
- Use **TypeScript** for new components (preferred)

### Testing

- Write tests for new features
- Maintain >80% code coverage
- Use descriptive test names

```python
def test_huggingface_provider_loads_model_successfully():
    """Test that HuggingFace provider can load a model."""
    provider = HuggingFaceProvider()
    model = provider.load_model("meta-llama/Llama-3.2-3B")
    assert model is not None
```

---

## Testing

### Unit Tests

```bash
# Run unit tests
pytest tests/unit/

# Run with verbose output
pytest -v tests/unit/
```

### Integration Tests

```bash
# Run integration tests
pytest tests/integration/

# Skip slow tests
pytest -m "not slow" tests/
```

### End-to-End Tests

```bash
# Run E2E tests
pytest tests/e2e/
```

### Coverage Report

```bash
pytest --cov=ModelForge --cov-report=html
open htmlcov/index.html  # View coverage report
```

---

## Debugging

### Backend Debugging

#### VS Code

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: ModelForge",
      "type": "python",
      "request": "launch",
      "module": "ModelForge.cli",
      "args": ["run"],
      "console": "integratedTerminal"
    }
  ]
}
```

#### PyCharm

1. Right-click `ModelForge/cli.py`
2. Select "Debug 'cli'"
3. Add breakpoints as needed

### Frontend Debugging

#### Browser DevTools

1. Open application in browser
2. Press F12 to open DevTools
3. Use Console, Network, and React DevTools

#### VS Code

Install "Debugger for Chrome" extension and configure.

---

## Environment Variables

Create a `.env` file in the repository root:

```bash
# HuggingFace token
HUGGINGFACE_TOKEN=your_token_here

# Optional: Custom paths
MODELFORGE_DATA_DIR=/path/to/data
MODELFORGE_MODELS_DIR=/path/to/models

# Development settings
DEBUG=true
LOG_LEVEL=DEBUG
```

---

## Common Development Tasks

### Adding a New Provider

1. Create `ModelForge/providers/your_provider.py`
2. Implement provider interface
3. Register in `provider_factory.py`
4. Add to `VALID_PROVIDERS` in `training_schemas.py`
5. Write tests
6. Update documentation

See [Custom Providers Guide](../providers/custom-providers.md).

### Adding a New Strategy

1. Create `ModelForge/strategies/your_strategy.py`
2. Implement strategy interface
3. Register in `strategy_factory.py`
4. Add to `VALID_STRATEGIES` in `training_schemas.py`
5. Write tests
6. Update documentation

See [Custom Strategies Guide](../strategies/custom-strategies.md).

### Adding a New API Endpoint

1. Add endpoint in appropriate router (e.g., `routers/finetuning_router.py`)
2. Create/update Pydantic schemas in `schemas/`
3. Implement business logic in `services/`
4. Write tests
5. Update API documentation

### Adding a Frontend Component

1. Create component in `Frontend/src/components/`
2. Add to relevant page in `Frontend/src/pages/`
3. Update routing if needed
4. Style with Tailwind CSS
5. Test in browser

---

## Documentation

### Build Documentation (If Using MkDocs)

```bash
# Install MkDocs
pip install mkdocs mkdocs-material

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

### Documentation Standards

- Use Markdown format
- Include code examples
- Add screenshots for UI features
- Keep docs in sync with code

---

## Continuous Integration

### GitHub Actions

ModelForge uses GitHub Actions for CI/CD.

Workflow runs on:
- Pull requests
- Pushes to main branch

Checks:
- Linting (black, flake8)
- Type checking (mypy)
- Tests (pytest)
- Coverage

### Local CI Simulation

```bash
# Run all CI checks locally
black --check ModelForge/ tests/
flake8 ModelForge/ tests/
mypy ModelForge/
pytest --cov=ModelForge
```

---

## Release Process

### Version Bumping

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v2.1.0`
4. Push tag: `git push --tags`

### Publishing to PyPI

```bash
# Build distribution
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

---

## Getting Help

### Resources

- **GitHub Discussions**: Ask questions
- **GitHub Issues**: Report bugs or request features
- **Discord/Slack**: Real-time chat (if available)
- **Documentation**: Comprehensive guides

### Before Asking for Help

1. Check existing documentation
2. Search existing issues
3. Try to reproduce the issue
4. Gather relevant information (logs, error messages)

---

## Contributing Guidelines

### Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add entry to CHANGELOG.md
4. Request review from maintainers
5. Address review feedback
6. Squash commits if requested

### Code Review Checklist

- [ ] Code follows style guide
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or documented)
- [ ] Performance considerations addressed
- [ ] Security implications reviewed

---

## Troubleshooting Development Issues

### Import Errors

**Problem**: Cannot import ModelForge modules

**Solution**:
```bash
# Reinstall in editable mode
pip install -e .
```

### CUDA Errors

**Problem**: CUDA not available

**Solution**:
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### Frontend Build Errors

**Problem**: npm build fails

**Solution**:
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Test Failures

**Problem**: Tests fail locally

**Solution**:
```bash
# Clear pytest cache
pytest --cache-clear

# Run with verbose output
pytest -vv
```

---

## Best Practices

### Code Quality

1. ✅ Write self-documenting code
2. ✅ Use meaningful variable names
3. ✅ Keep functions small and focused
4. ✅ Add comments for complex logic
5. ✅ Handle errors gracefully

### Performance

1. ✅ Profile before optimizing
2. ✅ Use appropriate data structures
3. ✅ Avoid premature optimization
4. ✅ Cache expensive computations
5. ✅ Test with realistic data sizes

### Security

1. ✅ Never commit secrets
2. ✅ Validate all inputs
3. ✅ Use environment variables for config
4. ✅ Keep dependencies updated
5. ✅ Follow security best practices

---

## Next Steps

- **[Contributing Guide](contributing.md)** - Contribution guidelines
- **[Architecture Overview](architecture.md)** - System architecture
- **[Model Configurations](model-configs.md)** - Adding model configs

---

**Happy coding!** Thanks for contributing to ModelForge! 🚀
