# ModelForge Documentation

Welcome to the comprehensive documentation for **ModelForge v2.0** - A no-code toolkit for fine-tuning Large Language Models on your local GPU.

## 📚 Documentation Index

### Getting Started
- [Quick Start Guide](getting-started/quickstart.md) - Get up and running in 5 minutes
- [What's New in v2.0](getting-started/whats-new.md) - Major features and improvements

### Installation & Setup
- [Windows Installation](installation/windows.md) - Complete setup guide for Windows users
  - Native Windows setup and limitations
  - WSL setup for full feature support
  - Docker with NVIDIA Container Toolkit
- [Linux Installation](installation/linux.md) - Setup guide for Linux
- [macOS (Apple Silicon) Installation](installation/macos-mps.md) - Setup for M1/M2/M3/M4 Macs with MPS support
- [Post-Installation Setup](installation/post-installation.md) - Initial configuration steps

### Configuration & Usage
- [Configuration Guide](configuration/configuration-guide.md) - Detailed configuration options
- [Dataset Formats](configuration/dataset-formats.md) - Preparing your training data
- [Training Tasks](configuration/training-tasks.md) - Text generation, summarization, Q&A
- [Hardware Profiles](configuration/hardware-profiles.md) - Optimizing for your GPU

### Providers
- [Provider Overview](providers/overview.md) - Understanding model providers
- [HuggingFace Provider](providers/huggingface.md) - Standard HuggingFace models
- [Unsloth Provider](providers/unsloth.md) - 2x faster training with Unsloth
- [Adding Custom Providers](providers/custom-providers.md) - Extend with new providers

### Training Strategies
- [Strategy Overview](strategies/overview.md) - Understanding training strategies
- [Supervised Fine-Tuning (SFT)](strategies/sft.md) - Standard LoRA fine-tuning
- [QLoRA Strategy](strategies/qlora.md) - Memory-efficient quantized training
- [RLHF Strategy](strategies/rlhf.md) - Reinforcement Learning from Human Feedback
- [DPO Strategy](strategies/dpo.md) - Direct Preference Optimization
- [Adding Custom Strategies](strategies/custom-strategies.md) - Extend with new strategies

### API Reference
- [REST API Documentation](api-reference/rest-api.md) - Complete API endpoints
- [Training Configuration Schema](api-reference/training-config.md) - Configuration options
- [Response Formats](api-reference/responses.md) - API response structures

### Troubleshooting & FAQs
- [Common Issues](troubleshooting/common-issues.md) - Frequently encountered problems
- [Windows-Specific Issues](troubleshooting/windows-issues.md) - Windows troubleshooting
- [FAQ](troubleshooting/faq.md) - Frequently asked questions
- [Performance Optimization](troubleshooting/performance.md) - Getting the best performance

### Contributing
- [Contributing Guide](contributing/contributing.md) - How to contribute
- [Development Setup](contributing/development.md) - Setting up development environment
- [Architecture Overview](contributing/architecture.md) - Understanding the codebase
- [Model Configurations](contributing/model-configs.md) - Adding model recommendations

## 🔗 Quick Links

- **[Quick Start](getting-started/quickstart.md)** - Start here if you're new to ModelForge
- **[Windows Setup](installation/windows.md)** - Essential for Windows users
- **[Configuration Guide](configuration/configuration-guide.md)** - Learn all configuration options
- **[Troubleshooting](troubleshooting/common-issues.md)** - Having problems? Check here first

## 📖 External Resources

- [GitHub Repository](https://github.com/ForgeOpus/ModelForge)
- [PyPI Package](https://pypi.org/project/modelforge-finetuning/)
- [Issue Tracker](https://github.com/ForgeOpus/ModelForge/issues)
- [Discussions](https://github.com/ForgeOpus/ModelForge/discussions)

## 💡 Need Help?

- Check the [FAQ](troubleshooting/faq.md) for quick answers
- Search [existing issues](https://github.com/ForgeOpus/ModelForge/issues) on GitHub
- Ask a question in [Discussions](https://github.com/ForgeOpus/ModelForge/discussions)
- Report bugs via [GitHub Issues](https://github.com/ForgeOpus/ModelForge/issues/new)

---

**ModelForge v2.0** - Making LLM fine-tuning accessible to everyone.
