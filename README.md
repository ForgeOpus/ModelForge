# ModelForge

[![PyPI](https://img.shields.io/pypi/v/modelforge-finetuning)](https://pypi.org/project/modelforge-finetuning/)
[![Python](https://img.shields.io/pypi/pyversions/modelforge-finetuning)](https://pypi.org/project/modelforge-finetuning/)
[![Docs](https://readthedocs.org/projects/modelforge-finetuning/badge/?version=latest)](https://modelforge-finetuning.readthedocs.io)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](https://github.com/ForgeOpus/ModelForge/blob/main/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/modelforge-finetuning?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/modelforge-finetuning)

<a href="https://www.producthunt.com/products/forgeopus?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-forgeopus" target="_blank" rel="noopener noreferrer"><img alt="ForgeOpus - Where AI masterpieces are forged. Your work, your opus. | Product Hunt" width="250" height="54" src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1080450&amp;theme=light&amp;t=1771311433851"></a>

![ModelForge logo](https://github.com/user-attachments/assets/12b3545d-0e8b-4460-9291-d0786c9cb0fa)

Hardware-aware fine-tuning for open models from the Hugging Face Hub, with browser, terminal and notebook interfaces.

ModelForge reads your GPUs, works out how large a model fits, picks a training recipe for your goal, and runs it while you watch the loss come down. It shows its reasoning before anything starts, so you don't find out at step 40 that the run doesn't fit in memory.

## Features

- **Hardware-aware planning.** Total VRAM, the strategy's memory overhead and a per-parameter cost set a ceiling on model size. ModelForge picks full fine-tuning, LoRA or QLoRA (8-bit or 4-bit) to match.
- **A recipe for each goal.** Supervised fine-tuning (SFT), preference training (DPO) and reinforcement training (GRPO), plus native trainers for summarization and extractive QA.
- **Model recommendations.** Searches the Hub for popular models of the right type that fit your hardware, or validates any model ID you enter.
- **Multi-GPU.** Data parallel by default, switching to FSDP automatically when a full fine-tune is too large for one card.
- **Live runs.** Streamed training logs, checkpoints you can resume from, and TensorBoard logging.
- **Playground.** Chat with a finished model or any checkpoint in the browser, even while another run is training.

## Install

ModelForge needs Python 3.13 or newer. Real training needs an NVIDIA GPU with CUDA. Apple Silicon (MPS) and CPU-only machines work, but only with small models.

```bash
pip install modelforge-finetuning

# Optional extras: the terminal wizard, and Liger kernels for CUDA
pip install "modelforge-finetuning[cli,gpu]"
```

Per-OS instructions, including PyTorch with CUDA, are in the [installation guides](https://modelforge-finetuning.readthedocs.io/en/latest/installation/linux/).

## Quick start

```bash
modelforge run
```

This starts the server and opens the browser console. The link it prints includes a one-time token that connects your browser. On a remote machine, use `modelforge run --no-browser --host 127.0.0.1` and connect through an SSH tunnel.

For gated models such as Llama, accept the licence on the model's Hub page, then log in once with `hf auth login`.

| Interface | Command | Best for |
|---|---|---|
| Browser console | `modelforge run` | Most people |
| Terminal wizard | `modelforge cli` | Headless servers over SSH (needs `[cli]`) |
| Notebook | `modelforge-nb` | Jupyter and Colab |

The [Quick Start guide](https://modelforge-finetuning.readthedocs.io/en/latest/getting-started/quickstart/) walks through a first run.

## Documentation

Full documentation is at **[modelforge-finetuning.readthedocs.io](https://modelforge-finetuning.readthedocs.io)**. A shorter version ships inside the app at `/app/docs`.

- [What's new in v4 and migrating from v3](https://modelforge-finetuning.readthedocs.io/en/latest/getting-started/whats-new/)
- [Dataset formats](https://modelforge-finetuning.readthedocs.io/en/latest/configuration/dataset-formats/)
- [How planning works](https://modelforge-finetuning.readthedocs.io/en/latest/configuration/configuration-guide/)
- [Training strategies](https://modelforge-finetuning.readthedocs.io/en/latest/strategies/overview/)
- [REST API](https://modelforge-finetuning.readthedocs.io/en/latest/api-reference/rest-api/)
- [Troubleshooting](https://modelforge-finetuning.readthedocs.io/en/latest/troubleshooting/common-issues/)

## Running from source

Building or installing from a source checkout needs Node.js and npm, because the browser UI is compiled into `web/frontend_dist/`. That directory isn't committed.

```bash
git clone https://github.com/ForgeOpus/ModelForge.git
cd ModelForge
uv sync --extra dev
python scripts/build_frontend.py   # or, from web/frontend: npm ci && npm run build
uv run modelforge run --no-browser
```

`pip wheel .`, `python -m build` and `uv build` run the frontend build automatically, and fail if Node.js is missing, so a wheel is never published without a UI. Installing an already-built wheel doesn't need Node.

See the [development guide](https://modelforge-finetuning.readthedocs.io/en/latest/contributing/development/) and [DEVELOPER_GUIDE.md](https://github.com/ForgeOpus/ModelForge/blob/main/DEVELOPER_GUIDE.md) for the codebase tour, tests and the release process.

## Contributing

Contributions are welcome. Read the [contributing guide](https://modelforge-finetuning.readthedocs.io/en/latest/contributing/contributing/) and the [code of conduct](https://github.com/ForgeOpus/ModelForge/blob/main/CODE_OF_CONDUCT.md), and report bugs or request features through [GitHub issues](https://github.com/ForgeOpus/ModelForge/issues).

## License

ModelForge is released under the [BSD 3-Clause License](https://github.com/ForgeOpus/ModelForge/blob/main/LICENSE).
