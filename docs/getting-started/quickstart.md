# Quick Start Guide

Get started with ModelForge in just a few minutes!

## Prerequisites

Before you begin, ensure you have:

- **Python 3.11.x** (Python 3.12 is not yet supported)
- **NVIDIA GPU** with at least 4GB VRAM (6GB+ recommended)
- **CUDA** installed and configured
- **HuggingFace Account** with an access token ([Create one here](https://huggingface.co/settings/tokens))

> **Windows Users**: See the [Windows Installation Guide](../installation/windows.md) for important platform-specific instructions, especially if you want to use the Unsloth provider.

## Installation

### 1. Install ModelForge

```bash
pip install modelforge-finetuning

# Optional extras
pip install modelforge-finetuning[cli]           # CLI wizard (required for `modelforge cli`)
pip install modelforge-finetuning[quantization]   # 4-bit/8-bit quantization support
```

### 2. Install PyTorch with CUDA Support

Visit the [PyTorch installation page](https://pytorch.org/get-started/locally/) and select your CUDA version.

For example, for CUDA 12.6:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Set Your HuggingFace Token

**Linux:**
```bash
export HUGGINGFACE_TOKEN=your_huggingface_token
```

**Windows PowerShell:**
```powershell
$env:HUGGINGFACE_TOKEN="your_huggingface_token"
```

**Windows CMD:**
```cmd
set HUGGINGFACE_TOKEN=your_huggingface_token
```

**Or use a .env file (all platforms):**
```bash
echo "HUGGINGFACE_TOKEN=your_huggingface_token" > .env
```

## Launch ModelForge

```bash
modelforge
```

Open your browser and navigate to:
```
http://localhost:8000
```

That's it! You should see the ModelForge web interface.

## Using the CLI Wizard

ModelForge also offers a terminal-based interactive wizard — useful for headless servers or SSH sessions where a browser isn't available:

```bash
pip install modelforge-finetuning[cli]   # Install CLI dependencies first
modelforge cli
```

This launches a step-by-step prompt that guides you through dataset selection, model configuration, and training — no browser required.

**In Jupyter notebooks**, you can also run the CLI wizard:

```python
from ModelForge.notebook_cli import run_cli
run_cli()
```

| Interface | Command | Best for |
|-----------|---------|----------|
| Web UI | `modelforge` | Local machines with a browser |
| CLI Wizard | `modelforge cli` | Headless servers, SSH, notebooks |

## Your First Fine-Tuning Job

### 1. Prepare Your Dataset

Create a JSONL file with your training data. The format depends on your task:

**Text Generation:**
```jsonl
{"input": "What is machine learning?", "output": "Machine learning is a subset of AI..."}
{"input": "Explain neural networks", "output": "Neural networks are computing systems..."}
```

**Summarization:**
```jsonl
{"input": "Long article text here...", "output": "Short summary here."}
```

**Question Answering:**
```jsonl
{"context": "Context paragraph...", "question": "What is X?", "answer": "X is..."}
```

See [Dataset Formats](../configuration/dataset-formats.md) for detailed information.

### 2. Upload Your Dataset

1. In the ModelForge UI, click **"Upload Dataset"**
2. Select your JSONL file
3. Wait for validation to complete

### 3. Configure Training

Select your training options:

- **Task**: Choose text-generation, summarization, or extractive-question-answering
- **Model**: Pick a recommended model for your GPU (or browse for others)
- **Provider**: Choose `huggingface` (standard) or `unsloth` (2x faster, requires WSL/Docker on Windows)
- **Strategy**: Choose `sft` (standard) or `qlora` (memory-efficient)
- **Hardware Profile**: Select based on your GPU VRAM

### 4. Start Training

Click **"Start Training"** and monitor progress in real-time!

### 5. Test Your Model

Once training completes:
1. Navigate to the **Playground** tab
2. Select your fine-tuned model
3. Enter a prompt and see the results!

## Next Steps

- **Learn about configuration options**: [Configuration Guide](../configuration/configuration-guide.md)
- **Optimize for your hardware**: [Hardware Profiles](../configuration/hardware-profiles.md)
- **Try different strategies**: [Training Strategies](../strategies/overview.md)
- **Use the API**: [REST API Documentation](../api-reference/rest-api.md)

## Stopping ModelForge

To stop the application and free up resources:

```bash
# Press Ctrl+C in the terminal running ModelForge
```

## Running ModelForge Again

Simply run:

```bash
modelforge
```

Your previous models and settings are preserved!

## Need Help?

- Check the [FAQ](../troubleshooting/faq.md) for common questions
- See [Troubleshooting](../troubleshooting/common-issues.md) for common issues
- Ask in [GitHub Discussions](https://github.com/forgeopus/modelforge/discussions)

---

**Congratulations! You're ready to start fine-tuning LLMs!** 🎉
