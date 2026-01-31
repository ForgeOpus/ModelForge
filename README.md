# ModelForge 🔧⚡

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/modelforge-finetuning?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/modelforge-finetuning)
[![License: BSD](https://img.shields.io/badge/License-BSD-yellow.svg)](https://opensource.org/licenses/BSD)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-2.0.1-blue)](https://github.com/ForgeOpus/ModelForge)

**Fine-tune LLMs on your laptop's GPU—no code, no PhD, no hassle.**

ModelForge v2.0 is a complete architectural overhaul bringing **2x faster training**, modular providers, advanced strategies, and production-ready code quality.

![logo](https://github.com/user-attachments/assets/12b3545d-0e8b-4460-9291-d0786c9cb0fa)

## ✨ What's New in v2.0

- 🚀 **2x Faster Training** with Unsloth provider
- 🧩 **Multiple Providers**: HuggingFace, Unsloth (more coming!)
- 🎯 **Advanced Strategies**: SFT, QLoRA, RLHF, DPO
- 📊 **Built-in Evaluation** with task-specific metrics
- 🏗️ **Modular Architecture** for easy extensibility
- 🔒 **Production-Ready** with proper error handling and logging

**[See What's New in v2.0 →](https://github.com/ForgeOpus/ModelForge/tree/main/docs/getting-started/whats-new.md)**

## 🚀 Features

- **GPU-Powered Fine-Tuning**: Optimized for NVIDIA GPUs (even 4GB VRAM)
- **One-Click Workflow**: Upload data → Configure → Train → Test
- **Hardware-Aware**: Auto-detects GPU and recommends optimal models
- **No-Code UI**: Beautiful React interface, no CLI or notebooks
- **Multiple Providers**: HuggingFace (standard) or Unsloth (2x faster)
- **Advanced Strategies**: SFT, QLoRA, RLHF, DPO support
- **Automatic Evaluation**: Built-in metrics for all tasks

## 📖 Supported Tasks

- **Text Generation**: Chatbots, instruction following, code generation, creative writing
- **Summarization**: Document condensing, article summarization, meeting notes
- **Question Answering**: RAG systems, document search, FAQ bots

## 🎯 Quick Start

### Prerequisites

- **Python 3.11.x** (Python 3.12 not yet supported)
- **GPU with 4GB+ VRAM**:
  - **NVIDIA GPU** (6GB+ recommended) for CUDA-accelerated training with Unsloth support
  - **OR Apple Silicon** (M1/M2/M3/M4) for MPS-accelerated training (HuggingFace provider only, experimental)
- **CUDA** (for NVIDIA GPUs) - [Installation Guide](https://developer.nvidia.com/cuda-downloads)
- **HuggingFace Account** with access token ([Get one here](https://huggingface.co/settings/tokens))
- **Linux, Windows, or macOS** operating system

> **Apple Silicon Users**: ModelForge now has **experimental support** for Apple Silicon Macs with MPS (Metal Performance Shaders). See [macOS MPS Installation Guide](https://github.com/ForgeOpus/ModelForge/tree/main/docs/installation/macos-mps.md) for setup instructions and limitations (HuggingFace provider only, no quantization support).
>
> **Windows Users**: See [Windows Installation Guide](https://github.com/ForgeOpus/ModelForge/tree/main/docs/installation/windows.md) for platform-specific instructions, especially for Unsloth support.

### Installation

```bash
# Install ModelForge
pip install modelforge-finetuning

# Install PyTorch with CUDA support
# Visit https://pytorch.org/get-started/locally/ for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Set HuggingFace Token

**Linux:**
```bash
export HUGGINGFACE_TOKEN=your_token_here
```

**Windows PowerShell:**
```powershell
$env:HUGGINGFACE_TOKEN="your_token_here"
```

**Or use .env file:**
```bash
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

### Run ModelForge

```bash
modelforge run
```

Open your browser to **http://localhost:8000** and start training!

**[Full Quick Start Guide →](https://github.com/ForgeOpus/ModelForge/tree/main/docs/getting-started/quickstart.md)**

## 📚 Documentation

### Getting Started
- **[Quick Start Guide](https://github.com/ForgeOpus/ModelForge/tree/main/docs/getting-started/quickstart.md)** - Get up and running in 5 minutes
- **[What's New in v2.0](https://github.com/ForgeOpus/ModelForge/tree/main/docs/getting-started/whats-new.md)** - Major features and improvements

### Installation
- **[Windows Installation](https://github.com/ForgeOpus/ModelForge/tree/main/docs/installation/windows.md)** - Complete Windows setup (including WSL and Docker)
- **[Linux Installation](https://github.com/ForgeOpus/ModelForge/tree/main/docs/installation/linux.md)** - Linux setup guide
- **[macOS (Apple Silicon) Installation](https://github.com/ForgeOpus/ModelForge/tree/main/docs/installation/macos-mps.md)** - Setup for M1/M2/M3 Macs with MPS support
- **[Post-Installation](https://github.com/ForgeOpus/ModelForge/tree/main/docs/installation/post-installation.md)** - Initial configuration

### Configuration & Usage
- **[Configuration Guide](https://github.com/ForgeOpus/ModelForge/tree/main/docs/configuration/configuration-guide.md)** - All configuration options
- **[Dataset Formats](https://github.com/ForgeOpus/ModelForge/tree/main/docs/configuration/dataset-formats.md)** - Preparing your training data
- **[Training Tasks](https://github.com/ForgeOpus/ModelForge/tree/main/docs/configuration/training-tasks.md)** - Understanding different tasks
- **[Hardware Profiles](https://github.com/ForgeOpus/ModelForge/tree/main/docs/configuration/hardware-profiles.md)** - Optimizing for your GPU

### Providers
- **[Provider Overview](https://github.com/ForgeOpus/ModelForge/tree/main/docs/providers/overview.md)** - Understanding providers
- **[HuggingFace Provider](https://github.com/ForgeOpus/ModelForge/tree/main/docs/providers/huggingface.md)** - Standard HuggingFace models
- **[Unsloth Provider](https://github.com/ForgeOpus/ModelForge/tree/main/docs/providers/unsloth.md)** - 2x faster training

### Training Strategies
- **[Strategy Overview](https://github.com/ForgeOpus/ModelForge/tree/main/docs/strategies/overview.md)** - Understanding strategies
- **[SFT Strategy](https://github.com/ForgeOpus/ModelForge/tree/main/docs/strategies/sft.md)** - Standard supervised fine-tuning
- **[QLoRA Strategy](https://github.com/ForgeOpus/ModelForge/tree/main/docs/strategies/qlora.md)** - Memory-efficient training
- **[RLHF Strategy](https://github.com/ForgeOpus/ModelForge/tree/main/docs/strategies/rlhf.md)** - Reinforcement learning
- **[DPO Strategy](https://github.com/ForgeOpus/ModelForge/tree/main/docs/strategies/dpo.md)** - Direct preference optimization

### API Reference
- **[REST API](https://github.com/ForgeOpus/ModelForge/tree/main/docs/api-reference/rest-api.md)** - Complete API documentation
- **[Training Config Schema](https://github.com/ForgeOpus/ModelForge/tree/main/docs/api-reference/training-config.md)** - Configuration options

### Troubleshooting
- **[Common Issues](https://github.com/ForgeOpus/ModelForge/tree/main/docs/troubleshooting/common-issues.md)** - Frequently encountered problems
- **[Windows Issues](https://github.com/ForgeOpus/ModelForge/tree/main/docs/troubleshooting/windows-issues.md)** - Windows-specific troubleshooting
- **[FAQ](https://github.com/ForgeOpus/ModelForge/tree/main/docs/troubleshooting/faq.md)** - Frequently asked questions

### Contributing
- **[Contributing Guide](https://github.com/ForgeOpus/ModelForge/tree/main/docs/contributing/contributing.md)** - How to contribute
- **[Architecture](https://github.com/ForgeOpus/ModelForge/tree/main/docs/contributing/architecture.md)** - Understanding the codebase
- **[Model Configurations](https://github.com/ForgeOpus/ModelForge/tree/main/docs/contributing/model-configs.md)** - Adding model recommendations

**[📖 Full Documentation Index →](https://github.com/ForgeOpus/ModelForge/tree/main/docs/README.md)**

## 🔧 Platform Support

| Platform | HuggingFace Provider | Unsloth Provider | Notes |
|----------|---------------------|------------------|-------|
| **Linux (Native)** | ✅ Full support | ✅ Full support | Recommended for best performance |
| **Windows (Native)** | ✅ Full support | ❌ Not supported | Use WSL or Docker for Unsloth |
| **WSL 2** | ✅ Full support | ✅ Full support | Recommended for Windows users |
| **Docker** | ✅ Full support | ✅ Full support | With NVIDIA runtime |
| **macOS (Apple MPS)** | ✅ Experimental | ❌ Not supported | Requires PyTorch MPS; no bitsandbytes / Unsloth; smaller models recommended |

**[Platform-Specific Installation Guides →](https://github.com/ForgeOpus/ModelForge/tree/main/docs/installation/)**

## ⚠️ Important Notes

### Windows Users

**Unsloth provider is NOT supported on native Windows.** For 2x faster training with Unsloth:

1. **Option 1: WSL (Recommended)** - [WSL Installation Guide](https://github.com/ForgeOpus/ModelForge/tree/main/docs/installation/windows.md#option-2-wsl-installation-recommended)
2. **Option 2: Docker** - [Docker Installation Guide](https://github.com/ForgeOpus/ModelForge/tree/main/docs/installation/windows.md#option-3-docker-installation)

The HuggingFace provider works perfectly on native Windows.

### Unsloth Constraints

When using Unsloth provider, you **MUST** specify a fixed `max_sequence_length`:

```json
{
  "provider": "unsloth",
  "max_seq_length": 2048  // ✅ Required - cannot be -1
}
```

Auto-inference (`max_seq_length: -1`) is **NOT supported** with Unsloth.

**[Learn more about Unsloth →](https://github.com/ForgeOpus/ModelForge/tree/main/docs/providers/unsloth.md)**

## 📂 Dataset Format

ModelForge uses JSONL format. Each task has specific fields:

**Text Generation:**
```jsonl
{"input": "What is AI?", "output": "AI stands for Artificial Intelligence..."}
{"input": "Explain ML", "output": "Machine Learning is a subset of AI..."}
```

**Summarization:**
```jsonl
{"input": "Long article text...", "output": "Short summary."}
```

**Question Answering:**
```jsonl
{"context": "Document text...", "question": "What is X?", "answer": "X is..."}
```

**[Complete Dataset Format Guide →](https://github.com/ForgeOpus/ModelForge/tree/main/docs/configuration/dataset-formats.md)**

## 🤝 Contributing

We welcome contributions! ModelForge v2.0's modular architecture makes it easy to:

- **Add new providers** - Just 2 files needed
- **Add new strategies** - Just 2 files needed
- **Add model recommendations** - Simple JSON configs
- **Improve documentation**
- **Fix bugs and add features**

**[Contributing Guide →](https://github.com/ForgeOpus/ModelForge/tree/main/docs/contributing/contributing.md)**

### Adding Model Recommendations

ModelForge uses modular configuration files for model recommendations. See the **[Model Configuration Guide](https://github.com/ForgeOpus/ModelForge/tree/main/docs/contributing/model-configs.md)** for instructions on adding new recommended models.

## 🛠 Tech Stack

- **Backend**: Python, FastAPI, SQLAlchemy
- **Frontend**: React.js
- **ML**: PyTorch, Transformers, PEFT, TRL
- **Training**: LoRA, QLoRA, bitsandbytes
- **Providers**: HuggingFace Hub, Unsloth

*Results on NVIDIA RTX 3090. Your results may vary.*

## 📜 License

BSD License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HuggingFace for Transformers and model hub
- Unsloth AI for optimized training kernels
- The open-source ML community

## 📧 Support

- **Documentation**: [https://github.com/ForgeOpus/ModelForge/tree/main/docs/](https://github.com/ForgeOpus/ModelForge/tree/main/docs/)
- **Issues**: [GitHub Issues](https://github.com/ForgeOpus/ModelForge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ForgeOpus/ModelForge/discussions)
- **PyPI**: [modelforge-finetuning](https://pypi.org/project/modelforge-finetuning/)

---

**ModelForge v2.0 - Making LLM fine-tuning accessible to everyone** 🚀

**[Get Started →](https://github.com/ForgeOpus/ModelForge/tree/main/docs/getting-started/quickstart.md)** | **[Documentation →](https://github.com/ForgeOpus/ModelForge/tree/main/docs/)** | **[GitHub →](https://github.com/ForgeOpus/ModelForge)**
