# FAQ

Short answers to common questions about ModelForge v4, with links to the pages that go deeper. For error messages, see [Common issues](common-issues.md).

## General

### What is ModelForge?

A tool that fine-tunes open models from the Hugging Face Hub on your own machine. It reads your GPUs, works out how large a model fits, picks a training recipe for your goal, and runs it while you watch the loss. You can use it from a browser console (`modelforge run`), a terminal wizard (`modelforge cli`) or a notebook (`modelforge-nb`). All three share the same planning and training code. See [Concepts](../getting-started/concepts.md).

### Is it free?

Yes. ModelForge is open source under the BSD-3-Clause licence. The models you train keep their own licences, and gated models like Llama require you to accept theirs on the Hub.

### What changed from v3?

v4 is a rewrite. The planner now chooses the model, tier and training config from your hardware; there are no per-model JSON configs to edit. Training runs on the Hugging Face stack only (Transformers, TRL, PEFT, Accelerate); Unsloth and the separate RLHF strategy are gone. GRPO reinforcement training and native trainers for summarization and extractive QA were added, there's a new browser console with a Playground, and Python 3.13 is required. See [What's new](../getting-started/whats-new.md).

### Does it send my data anywhere?

No. Datasets, checkpoints and models stay on the machine running the server. It contacts the Hugging Face Hub to search and download models and Hub datasets, and only uploads when you press **Push** for a model. Playground conversations are not stored on the server.

## Requirements and installation

### What do I need?

- Python 3.13 or newer.
- An NVIDIA GPU with CUDA for practical training. More VRAM means larger models.
- Disk space for base models, checkpoints and outputs.

See [Hardware and memory](../configuration/hardware-profiles.md) for what fits where.

### Can I run it without a GPU?

Yes, but only for tiny models. CPU-only plans cap the model at 500M parameters (less on machines with little RAM) and train a LoRA adapter on full-precision weights. Expect it to be very slow.

### Does it work on a Mac?

On Apple Silicon, yes, with limits. MPS memory is counted as 8 GB, bitsandbytes quantization isn't available, and the plan picks full fine-tuning or LoRA bf16 of a small model. See [macOS (Apple Silicon)](../installation/macos-mps.md).

### AMD or Intel GPUs?

Not supported. GPU detection uses NVIDIA's NVML, so other GPUs aren't seen and the machine is planned as CPU-only.

### Does it work on Windows?

Natively, for single-GPU training, once you install a CUDA build of PyTorch. Liger kernels and multi-GPU training need Linux, so use WSL2 for those. See [Windows issues](windows-issues.md).

### Which extras should I install?

| Extra | Adds | Install when |
|---|---|---|
| `[cli]` | click, rich | You want `modelforge cli` or `modelforge tensorboard`. |
| `[gpu]` | liger-kernel | You train on CUDA under Linux and want lower memory use. |

```bash
pip install "modelforge-finetuning[cli,gpu]"
```

### Can I use conda or uv?

Yes. Any environment with Python 3.13 works. Install with `pip` inside the conda environment, or with `uv pip install` / `uv add`.

## Training

### Which objective should I use?

- **Supervised (SFT)**: you have examples of the input and the ideal output. Start here.
- **Preference (DPO)**: you have pairs of a better and a worse answer to the same prompt, and want to steer an already-capable model.
- **Reinforcement (GRPO)**: you have prompts with one exactly correct answer each, such as arithmetic or short factual answers.

See [Strategies overview](../strategies/overview.md).

### Which use cases are supported?

Instruction Following and Text Generation (supervised, preference, reinforcement), Summarization (supervised) and Extractive QA (supervised). See [Use cases and tasks](../configuration/training-tasks.md).

### How does it decide full fine-tuning vs LoRA vs QLoRA?

It computes a parameter ceiling for each tier from your VRAM, searches the Hub for popular models of the right type under the largest ceiling, picks the largest candidate, then uses the least quantized tier that fits that model. See [How planning works](../configuration/configuration-guide.md) and [LoRA and QLoRA](../strategies/qlora.md).

### Can I change the hyperparameters?

Not in the console, which shows the training config read-only. Through the API you can send an edited plan to `POST /api/runs`; it's used as sent when `model_id` is the plan's first candidate. See [Training config](../api-reference/training-config.md#overriding-the-plan).

### How long does training take?

It depends on the model size, dataset size, sequence length and GPU. The defaults train for 3 epochs. The run page shows progress and an estimated time remaining once steps are being logged. See [Performance](performance.md) for ways to speed it up.

### Can I train several models at once?

Not in one server: one run trains at a time, and starting another returns `409`. Running several ModelForge processes at once isn't supported either, because they share one run database and a starting process marks other processes' active runs as failed.

### Can I pause and resume?

There's no pause, but there is resume. Checkpoints are saved every 500 steps. Cancel a run, then press **Resume** beside a checkpoint to start a new run that continues from it. See [Checkpoints](../user-guide/checkpoints.md).

### What happens if the server stops mid-run?

The run ends with it. On the next start it's marked failed with `ModelForge stopped before this run reached a terminal state.` Its checkpoints are intact, so you can resume.

### How do I know training is working?

Watch the loss in the run's training log; it should trend down over the first few hundred steps. TensorBoard logs are written to `<output dir>/tensorboard` by default. View them with:

```bash
modelforge tensorboard ./modelforge_output/tensorboard
```

### Why did a browser tab with TensorBoard open on the server?

When a run starts with TensorBoard enabled, the trainer launches TensorBoard on `127.0.0.1:6006` on the training machine and asks the system to open it in a browser. On a headless server nothing opens. To view it from another machine, forward port 6006 over SSH.

### Can I use multiple GPUs?

On Linux, yes. Every detected GPU is used by default, with data parallel, or FSDP for full fine-tunes too large for one card. Use `gpu_indices` through the API to train on a subset. DeepSpeed and tensor parallelism are not supported. See [Multi-GPU training](../strategies/multi-gpu.md).

## Datasets

### What formats can I use?

Local `.jsonl`, `.json`, `.csv`, `.parquet`, `.arrow` and `.txt` files, a directory of shards, or a Hugging Face Hub dataset ID. Column names depend on the use case and objective; for example supervised Instruction Following needs `prompt` and `completion` (or `instruction` and `response`). See [Dataset formats](../configuration/dataset-formats.md).

### Can I use a chat `messages` column?

Not through the built-in use cases. Their validation requires prompt/completion columns (or their aliases) for supervised training. Convert multi-turn chats into prompt/completion pairs.

### How many examples do I need?

There's no fixed minimum. A few hundred clean, consistent examples are often enough to change a model's format or style; teaching new knowledge or skills takes far more. Quality matters more than quantity.

### Where should the dataset be?

On the machine running the server. Local paths are resolved there, relative to the folder `modelforge run` was started in. Use **Validate** to check the path and columns before you train.

### Can I use private or gated Hub datasets?

Yes, after `hf auth login` on the server machine with an account that has access.

## Models

### Which base models can I use?

Any Hub model with the right pipeline tag for the use case, a published safetensors parameter count, and a size that fits your hardware. The planner suggests popular ones; you can type any other ID. Models that need `trust_remote_code` aren't loaded by the Playground's one-shot path. See [Hugging Face provider](../providers/huggingface.md).

### Can I use gated models like Llama?

Yes. Accept the licence on the model's Hub page, then run `hf auth login` on the server machine.

### Can I fine-tune a model from a local folder?

Not from the console. Planning and model validation look the model up on the Hub, so a local path is refused there. Through the API, a full `recommendation` whose first `recommended_models` entry is the local path skips that lookup and the folder is loaded with `from_pretrained`, but this path isn't tested and nothing checks the model fits.

### What do I get at the end?

The output directory you chose contains the trained model (the full model for full fine-tuning, just the adapter for LoRA and QLoRA), the tokenizer, every checkpoint, TensorBoard logs, and `modelforge_config.json`. Adapters aren't merged into the base weights.

### How do I use the trained model elsewhere?

Load it with Transformers (`from_pretrained` on the folder), or with PEFT on top of the base model named in `modelforge_config.json` for adapter runs. Or push it to the Hub from **Artifacts**.

### Can I chat with the model before training finishes?

Yes. The Playground can load any registered checkpoint, including from a run that is still training. During training it runs on CPU and disk so it doesn't take the trainer's VRAM. See [Playground](../user-guide/playground.md).

## Strategies and providers

### What's the difference between LoRA and QLoRA?

Both freeze the base model and train small adapter matrices. QLoRA additionally stores the frozen weights in 8-bit or 4-bit form, so larger models fit, at some cost in speed and quality. See [LoRA and QLoRA](../strategies/qlora.md).

### Can I write my own reward function for GRPO?

No. GRPO uses one built-in reward: 1.0 when the completion, trimmed of surrounding whitespace, exactly equals the row's `answer`, otherwise 0.0. Executable reward functions are deliberately not accepted. See [GRPO](../strategies/grpo.md).

### Is Unsloth supported?

No. v4 has one provider, Hugging Face. Providers are pluggable in code; see [Custom providers](../providers/custom-providers.md).

### Can I add my own strategy or use case?

Yes, in code. See [Custom strategies](../strategies/custom-strategies.md) and [Adding a use case](../contributing/use-cases.md).

## API and deployment

### Can I script ModelForge?

Yes. Everything the console does goes through a token-authenticated JSON API on the same server. Start the server with `--token` so the token stays fixed. See [REST API](../api-reference/rest-api.md).

### Is there a Python SDK?

There's no separate client package. Use any HTTP client against the REST API, or call the Python modules directly (`workflow.service.WorkflowService`, `workflow.jobs.TrainingJobManager`), which is what the notebook interface does. See [Architecture](../contributing/architecture.md).

### Is there a Docker image?

No official image is published.

### Can I expose ModelForge on the internet?

It isn't designed for that. The only protection is the startup token, there's no TLS and no user accounts, and anyone with the token can start runs and push models with your Hub login. Bind to `127.0.0.1` and use an SSH tunnel. If you must expose it, put it behind a reverse proxy with TLS and its own authentication.

### Where is run history stored?

In a SQLite database, `modelforge.db`, in your user data folder: `~/.local/share/ModelForge` on Linux, `~/Library/Application Support/ModelForge` on macOS, `%LOCALAPPDATA%\ForgeOpus\ModelForge` on Windows. Deleting it clears run history and the model registry, but not the output folders.

## Contributing

### How can I contribute?

Bug reports, documentation fixes and pull requests are welcome at [github.com/ForgeOpus/ModelForge](https://github.com/ForgeOpus/ModelForge). See [Contributing](../contributing/contributing.md) and [Development setup](../contributing/development.md).

## See also

- [Common issues](common-issues.md)
- [Performance and out-of-memory](performance.md)
- [Windows issues](windows-issues.md)
- [Quickstart](../getting-started/quickstart.md)
