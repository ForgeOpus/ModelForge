# Frequently Asked Questions (FAQ)

Common questions and answers about ModelForge.

## General

### What is ModelForge?

ModelForge is a no-code toolkit for fine-tuning Large Language Models on your local GPU. It provides a web-based UI for training custom models without writing code.

### Is ModelForge free?

Yes! ModelForge is open-source under the MIT license. You can use it freely for personal or commercial projects.

### What's the difference between v2.x and v3?

v3 builds on the v2 architecture with:
- Apple Silicon (MPS) support — train on M1/M2/M3/M4/M5 Macs natively
- Interactive CLI wizard (`modelforge cli`) for headless/SSH environments
- Optional quantization — bitsandbytes moved to `[quantization]` extra for a lighter base install
- Schema validation — catches incompatible config combinations at startup

All existing v2 workflows remain compatible.

## System Requirements

### What are the minimum requirements?

**NVIDIA GPU (Windows/Linux):**
- Python 3.11.x
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.8 or 12.x
- 8GB RAM
- 10GB free disk space

**Apple Silicon Mac (macOS):**
- Python 3.11.x
- Apple Silicon Mac (M1 or later) with 8GB+ unified memory
- macOS 12.3 or later
- 10GB free disk space

### Can I run ModelForge without a GPU?

Yes, but training will be extremely slow (10-100x slower). Only recommended for testing with very small models.

### Can I use AMD or Intel GPUs?

Currently, only NVIDIA GPUs are supported due to CUDA requirements.

### Does ModelForge work on macOS?

**Yes — Apple Silicon Macs are supported via PyTorch's MPS backend (added in v3).**

Supported chips: M1, M2, M3, M4, M5. Intel Macs are not supported (no MPS).

**Limitations on MPS:**
- HuggingFace provider only (Unsloth requires NVIDIA CUDA)
- No 4-bit/8-bit quantization (bitsandbytes is CUDA-only)
- Smaller models recommended (1–7B depending on unified memory)

See the **[macOS Installation Guide](../installation/macos-mps.md)** for full setup instructions.

## Installation

### Why Python 3.11 specifically?

Some dependencies don't yet support Python 3.12. We'll add support when the ecosystem catches up.

### How do I check my CUDA version?

```bash
nvcc --version
```

### Can I install ModelForge in a conda environment?

Yes! Create environment with Python 3.11:
```bash
conda create -n modelforge python=3.11
conda activate modelforge
pip install modelforge-finetuning
```

### Installation fails with "No matching distribution"

Make sure you're using Python 3.11:
```bash
python --version  # Should show 3.11.x
```

## Windows-Specific

### Does ModelForge work on Windows?

Yes! The HuggingFace provider works perfectly on native Windows. For Unsloth support, use WSL or Docker.

### Why doesn't Unsloth work on Windows?

Unsloth requires Linux-specific libraries and compilation. Use WSL 2 or Docker for Unsloth support.

### How do I install WSL?

Open PowerShell as Administrator:
```powershell
wsl --install -d Ubuntu-22.04
```

See [Windows Installation Guide](../installation/windows.md) for details.

### Do I need to install NVIDIA drivers in WSL?

No! WSL uses your Windows NVIDIA drivers automatically. Only install CUDA Toolkit in WSL.

## Training

### How long does training take?

Depends on model size, dataset size, and hardware:

| Model Size | Dataset | GPU | Provider | Time |
|------------|---------|-----|----------|------|
| 1B | 500 examples | RTX 3060 | HuggingFace | 10 min |
| 1B | 500 examples | RTX 3060 | Unsloth | 5 min |
| 7B | 1000 examples | RTX 3090 | HuggingFace | 90 min |
| 7B | 1000 examples | RTX 3090 | Unsloth | 45 min |

### How much VRAM do I need?

| Model Size | Minimum VRAM | Recommended VRAM |
|------------|--------------|------------------|
| < 1B | 4GB | 6GB |
| 1-3B | 6GB | 8GB |
| 3-7B | 8GB | 12GB |
| 7-13B | 12GB | 16GB |
| 13B+ | 16GB | 24GB+ |

Use QLoRA strategy to reduce memory usage by 30-50%.

### Can I train multiple models simultaneously?

Not recommended. Training uses all available VRAM. Train one model at a time.

### Can I pause and resume training?

Currently not supported. Training must complete in one session. Use checkpointing to save progress.

### How do I know if training is working?

Monitor in the UI:
- Loss should decrease over time
- Accuracy/metrics should improve
- No errors in console

## Datasets

### What dataset format does ModelForge use?

JSONL (JSON Lines). Each line is one JSON object.

See [Dataset Formats](../configuration/dataset-formats.md) for details.

### How many examples do I need?

Minimum: 100 examples  
Recommended: 1,000+ examples  
Optimal: 5,000+ examples  

Quality > quantity!

### Can I use datasets from HuggingFace?

Yes! Convert them to JSONL format:

```python
from datasets import load_dataset
import json

dataset = load_dataset("your-dataset", split="train")

with open('dataset.jsonl', 'w') as f:
    for item in dataset:
        f.write(json.dumps({"input": item["text"], "output": item["label"]}) + '\n')
```

### Can I fine-tune on copyrighted data?

Check the license of your data. Fine-tuning on copyrighted material may violate copyright laws.

## Models

### What models are supported?

All HuggingFace models compatible with Transformers:
- Llama (1, 2, 3)
- Mistral
- Qwen
- Gemma
- Phi
- BART
- T5
- And many more!

### Can I use gated models (like Llama)?

Yes! You need:
1. Accept license on HuggingFace
2. Use access token with proper permissions

### Can I use local models?

Yes! Provide local path instead of HuggingFace ID:

```json
{
  "model_name": "/path/to/local/model"
}
```

### How do I download my trained model?

Navigate to Models tab in UI and click Download, or find checkpoints in:
- Linux: `~/.local/share/modelforge/model_checkpoints/`
- Windows: `C:\Users\<user>\AppData\Local\modelforge\model_checkpoints\`

## Providers

### What's the difference between HuggingFace and Unsloth?

| Feature | HuggingFace | Unsloth |
|---------|-------------|---------|
| Speed | 1x | 2x |
| Memory | Baseline | -20% |
| Platform | All | Linux/WSL/Docker |
| Compatibility | All models | Llama, Mistral, Qwen, Gemma, Phi |

### Should I use Unsloth?

Use Unsloth if:
- Running on Linux or WSL
- Using supported models
- Need faster training
- Have limited VRAM

Use HuggingFace if:
- On native Windows
- Using unsupported models
- Maximum compatibility needed

### Why does Unsloth require max_seq_length?

Unsloth pre-allocates memory for performance. Auto-inference (`-1`) is not supported.

## Strategies

### What's the difference between SFT and QLoRA?

**SFT**: Standard supervised fine-tuning with LoRA
**QLoRA**: Quantized LoRA using 4-bit quantization for lower memory

QLoRA uses 30-50% less memory with minimal quality loss.

### When should I use RLHF vs DPO?

In ModelForge, both RLHF and DPO use TRL's `DPOTrainer` internally. The difference is in default hyperparameters:

- **RLHF**: More conservative defaults (lr=1.41e-5, 1 epoch) — suited for careful alignment
- **DPO**: Standard defaults (lr=5e-7, 3 epochs) — good general-purpose preference learning

Both require preference data (prompt/chosen/rejected) and `"task": "text-generation"`. Start with SFT for most use cases.

### What is `modelforge cli`?

`modelforge cli` launches an 8-step interactive wizard in the terminal. It's ideal for headless servers, SSH sessions, or Jupyter notebooks where a browser isn't available. Install the CLI extra first:

```bash
pip install modelforge-finetuning[cli]
```

### Why do I get "bitsandbytes is required" error?

bitsandbytes is now an optional dependency. Install it with:

```bash
pip install modelforge-finetuning[quantization]
```

### What happened to legacy tuners (CausalLLMTuner, Finetuner, etc.)?

Legacy tuners were removed in v2.1. Use the provider/strategy pattern instead:

```json
{
  "provider": "huggingface",
  "strategy": "sft"
}
```

### What's the difference between RLHF and DPO in ModelForge?

Both use TRL's `DPOTrainer` internally — no PPO, no reward model needed. The only difference is default hyperparameters. RLHF uses more conservative settings (lower learning rate, fewer epochs) while DPO uses standard settings.

### Can I mix strategies?

No. Choose one strategy per training run.

## Errors

### "CUDA out of memory"

Solutions:
1. Reduce `per_device_train_batch_size`
2. Use QLoRA strategy
3. Reduce `max_seq_length`
4. Enable `gradient_checkpointing`
5. Use smaller model

### "Model not found"

Check:
1. Model ID is correct
2. HuggingFace token is set
3. You have access to gated models
4. Internet connection is working

### "Dataset validation failed"

Check:
1. File is valid JSONL
2. Required fields are present
3. All values are strings
4. At least 10 examples

### "Provider not found"

Check:
1. Provider is installed (`pip install unsloth`)
2. Provider name is correct (`huggingface` or `unsloth`)
3. On correct platform (Unsloth needs Linux/WSL)

## Performance

### Training is very slow

1. Use Unsloth provider (2x faster)
2. Use QLoRA strategy
3. Increase batch size if you have VRAM
4. Use fp16 or bf16
5. Enable gradient checkpointing

### How do I speed up training?

Best practices:
- Use Unsloth provider
- Larger batch size (if VRAM allows)
- Use bf16 on Ampere+ GPUs (RTX 30xx/40xx)
- Reduce gradient accumulation steps
- Use NVMe SSD for dataset storage

### How do I reduce memory usage?

Best practices:
- Use QLoRA strategy
- Enable 4-bit quantization
- Reduce batch size
- Reduce max_seq_length
- Enable gradient checkpointing

## API

### Can I use ModelForge via API?

Yes! ModelForge provides a REST API. See [API Documentation](../api-reference/rest-api.md).

### Can I integrate ModelForge into my app?

Yes! Use the API or import ModelForge as a library.

### Is there a Python SDK?

Not yet, but you can use the REST API with `requests`.

## Deployment

### Can I deploy ModelForge to production?

Yes! ModelForge is production-ready. Deploy with:
- Docker containers
- Systemd services
- Cloud platforms (AWS, GCP, Azure)

### How do I expose ModelForge to the internet?

Use reverse proxy (nginx/Apache) with SSL:

```nginx
server {
    listen 443 ssl;
    location / {
        proxy_pass http://localhost:8000;
    }
}
```

### Can I use PostgreSQL instead of SQLite?

Yes! ModelForge uses SQLAlchemy. Easy to switch to PostgreSQL.

## Contributing

### How can I contribute?

See [Contributing Guide](../contributing/contributing.md) for:
- Reporting bugs
- Suggesting features
- Submitting PRs
- Adding model configurations

### How do I add a new provider?

Create provider class and register in factory. See [Custom Providers](../providers/custom-providers.md).

### How do I add model recommendations?

Add JSON config file to `model_configs/`. See [Model Configurations](../contributing/model-configs.md).

## Still Have Questions?

- Check [Troubleshooting Guide](common-issues.md)
- Search [GitHub Issues](https://github.com/forgeopus/modelforge/issues)
- Ask in [GitHub Discussions](https://github.com/forgeopus/modelforge/discussions)
- Create new issue if bug

---

**Can't find your answer?** Ask in [GitHub Discussions](https://github.com/forgeopus/modelforge/discussions)!
