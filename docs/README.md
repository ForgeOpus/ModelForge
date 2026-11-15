# ModelForge Documentation

ModelForge fine-tunes open models from the Hugging Face Hub on the machine you run it on. It reads your GPUs, works out how large a model fits, picks a training recipe for your goal, and runs it while you watch the loss come down. This page explains what ModelForge does and where each part of the documentation lives.

!!! note "This is the documentation for ModelForge v4"
    v4 is a rewrite. Commands, the Python version, storage locations and several features changed from v3. If you are upgrading, read [What's New in v4](getting-started/whats-new.md) first.

## What it does

Fine-tuning normally means choosing a base model, a precision, an adapter configuration, a batch size and a parallelism scheme, then discovering at step 40 that the run is out of memory. ModelForge makes those choices from your hardware and shows its reasoning before anything starts.

- **Hardware-aware planning.** Total VRAM, the strategy's memory overhead and a fixed per-parameter cost give a ceiling on model size. The planner then picks a tier: full fine-tuning, LoRA, 8-bit QLoRA or 4-bit QLoRA. See [How Planning Works](configuration/configuration-guide.md) and [Hardware and Memory](configuration/hardware-profiles.md).
- **Model search.** The recommendation searches the Hugging Face Hub for the most-downloaded models of the right type that fit under the ceiling. You can also type in any Hub model ID.
- **A recipe per goal.** Supervised fine-tuning (SFT), DPO preference training and GRPO reinforcement training for causal language models, plus native trainers for summarization and extractive QA.
- **Multi-GPU.** Data parallel by default, with FSDP sharding when full fine-tuning needs more memory than one GPU has. See [Multi-GPU](strategies/multi-gpu.md).
- **Live runs.** Streamed training logs, checkpoints every 500 steps that you can resume from, and TensorBoard logging.
- **A playground.** Chat with a finished model or any of its checkpoints in the browser.

## Three ways to use it

The browser console, the terminal wizard and notebooks all call the same planning, dataset and training code, so a dataset that validates in one will train in the others.

| Interface | Start it with | Best for |
|---|---|---|
| Browser console | `modelforge run` | Most people. The docs refer to it unless noted. |
| Terminal wizard | `modelforge cli` | Headless servers over SSH. Needs the `[cli]` extra. |
| Notebook | `modelforge-nb` | Jupyter and Colab sessions. Plain `input()` prompts, no widgets. |

```bash
pip install modelforge-finetuning
modelforge run
```

ModelForge needs Python 3.13 or newer. Real training needs an NVIDIA GPU with CUDA. Apple Silicon (MPS) and CPU-only machines work with small models. Continue with the [Quick Start](getting-started/quickstart.md).

## Documentation index

### Getting started

- [Quick Start](getting-started/quickstart.md): install, start the server and run a first fine-tune.
- [What's New in v4](getting-started/whats-new.md): changes since v3 and how to migrate.
- [Fine-Tuning Concepts](getting-started/concepts.md): what fine-tuning is, full fine-tuning vs. PEFT, and the three objectives.

### Installation

- [Windows](installation/windows.md): native Windows with a CUDA build of PyTorch, or WSL2.
- [Linux](installation/linux.md): NVIDIA drivers, PyTorch and the optional Liger kernels.
- [macOS (Apple Silicon)](installation/macos-mps.md): MPS support and its limits.
- [Post-Installation Setup](installation/post-installation.md): Hugging Face login, the startup token, storage locations and environment variables.

### User guide

- [Browser Console](user-guide/browser-console.md): New run, Runs, Artifacts and Settings.
- [Playground](user-guide/playground.md): chat with trained models and checkpoints.
- [Terminal Wizard](user-guide/terminal-cli.md): `modelforge cli` step by step.
- [Notebooks](user-guide/notebooks.md): `modelforge-nb` in Jupyter and Colab.
- [Checkpoints and Resuming](user-guide/checkpoints.md): where checkpoints go and how to continue a stopped run.

### Configuration

- [How Planning Works](configuration/configuration-guide.md): from hardware to a training plan.
- [Dataset Formats](configuration/dataset-formats.md): file types, required columns and aliases.
- [Use Cases and Tasks](configuration/training-tasks.md): instruction following, text generation, summarization and extractive QA.
- [Hardware and Memory](configuration/hardware-profiles.md): VRAM budgets, tiers, CPU and Apple Silicon rules.

### Providers

- [Provider Overview](providers/overview.md): how models and tokenizers are loaded.
- [Hugging Face Provider](providers/huggingface.md): the built-in provider.
- [Adding a Provider](providers/custom-providers.md): write your own.

### Training strategies

- [Strategy Overview](strategies/overview.md): how objectives map to trainers.
- [Supervised Fine-Tuning (SFT)](strategies/sft.md)
- [Preference Training (DPO)](strategies/dpo.md)
- [Reinforcement Training (GRPO)](strategies/grpo.md)
- [Full Fine-Tuning](strategies/full-finetuning.md)
- [LoRA and QLoRA](strategies/qlora.md)
- [Multi-GPU (Data Parallel and FSDP)](strategies/multi-gpu.md)
- [Memory Optimizations](strategies/memory-optimizations.md): gradient checkpointing, packing, Liger kernels and activation offloading.
- [Adding a Strategy](strategies/custom-strategies.md)

### API reference

- [REST API](api-reference/rest-api.md): authentication and every endpoint.
- [Training Configuration](api-reference/training-config.md): the keys in a generated training config.
- [Responses and Events](api-reference/responses.md): recommendation fields, run and model shapes, WebSocket events.
- [CLI Commands](api-reference/cli.md): `modelforge` and `modelforge-nb`.

### Troubleshooting

- [Common Issues](troubleshooting/common-issues.md)
- [Windows-Specific Issues](troubleshooting/windows-issues.md)
- [FAQ](troubleshooting/faq.md)
- [Performance and Memory](troubleshooting/performance.md): speed and out-of-memory errors.

### Contributing

- [Contributing Guide](contributing/contributing.md)
- [Development Setup](contributing/development.md): source install, frontend build and tests.
- [Architecture Overview](contributing/architecture.md): registries, workflow service and training flow.
- [Adding a Use Case](contributing/use-cases.md)

## External resources

- [GitHub repository](https://github.com/ForgeOpus/ModelForge)
- [PyPI package](https://pypi.org/project/modelforge-finetuning/)
- [Issue tracker](https://github.com/ForgeOpus/ModelForge/issues)
