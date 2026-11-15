# What's New in v4

ModelForge v4 is a rewrite of the v3 codebase. The package name on PyPI is still `modelforge-finetuning`, but the Python version, the commands, the way a training plan is made and where data is stored have all changed. This page lists what is new, what was removed, and how to move from v3.

## What's new in v4

### A hardware-aware planner instead of fixed profiles

v3 picked models from three hand-written JSON profiles (`low_end`, `mid_range`, `high_end`) and let you edit hyperparameters in the UI. v4 computes the plan from the hardware it detects:

- It sums the VRAM of the selected GPUs (Apple MPS counts as 8 GB), adjusts it for the strategy's extra model copies and batch size, and computes a parameter ceiling for each tier: `full_finetune`, `lora_bf16`, `qlora_8bit` and `qlora_4bit`.
- It searches the Hugging Face Hub for the most-downloaded models with the right pipeline tag that fit under the ceiling, and builds a training config for the chosen model.
- The plan shows its reasoning: a VRAM estimate, the PEFT decision, the parallelism decision and warnings. The training config is read-only in the console.

See [How Planning Works](../configuration/configuration-guide.md) and [Hardware and Memory](../configuration/hardware-profiles.md).

### Objectives and trainers

Training is described by a use case (Instruction Following, Text Generation, Summarization, Extractive QA) and an objective (Supervised, Preference, Reinforcement).

| Objective | Trainer | Status in v4 |
|---|---|---|
| Supervised | TRL `SFTTrainer` (`sft`) | Rewritten. Packs short examples, uses gradient checkpointing. |
| Preference | TRL `DPOTrainer` (`dpo`) | Rewritten. DPO existed in v3; v4 budgets its frozen reference model in the planner. |
| Reinforcement | TRL `GRPOTrainer` (`grpo`) | New. Four sampled completions per prompt, exact-match reward. |
| Summarization | Native encoder-decoder trainer | Rewritten. Uses a sequence-to-sequence trainer instead of a chat-style one. |
| Extractive QA | QA trainer with start/end labels (`qa`) | Rewritten. Long passages are split into overlapping windows (stride 128 tokens). |

Dataset column aliases (for example `instruction`/`response` for `prompt`/`completion`) are applied the same way in the browser, terminal and notebook. See [Dataset Formats](../configuration/dataset-formats.md) and [Strategy Overview](../strategies/overview.md).

### Multi-GPU with FSDP


With two or more CUDA GPUs, the planner chooses data parallel by default and FSDP (fully sharded data parallel) when full fine-tuning needs more memory than the smallest GPU has. FSDP is only used with full fine-tuning. See [Multi-GPU](../strategies/multi-gpu.md).

### Memory optimizations

Generated configs turn on gradient checkpointing and packing, enable Liger kernels on CUDA when the optional `[gpu]` extra is installed, and turn on activation offloading for SFT on GPUs with less than 12 GB. See [Memory Optimizations](../strategies/memory-optimizations.md).

### Checkpoints and resume

Every run saves a checkpoint every 500 steps to `<output dir>/checkpoint-<step>`. Failed or cancelled runs show a **Resume** button per checkpoint, which starts a new run with the same plan from that step. If the server stops mid-run, the run is marked failed on the next start and can be resumed. See [Checkpoints and Resuming](../user-guide/checkpoints.md).

### A new browser console

The browser UI is a new React and Vite app with **New run**, **Runs**, **Playground**, **Artifacts**, **Settings** and built-in documentation pages. Run pages stream training logs over a WebSocket. See [Browser Console](../user-guide/browser-console.md).

### An in-browser Playground

v3 opened a chat session in a separate terminal window. v4 has a Playground page in the console that chats with a completed run's final model or any checkpoint, with a choice of context window and Markdown export. See [Playground](../user-guide/playground.md).

### Token authentication

`modelforge run` generates a random startup token and prints a link that carries it. Every API endpoint except `/api/health`, and every WebSocket, requires the token. Use `--token` to fix it across restarts. See [Post-Installation Setup](../installation/post-installation.md#the-startup-token).

### Terminal wizard and notebook entry point

`modelforge cli` (terminal wizard) and `modelforge-nb` (notebook prompts) existed in v3 and were rewritten on top of the same workflow code as the browser. The notebook entry point now uses plain `input()` prompts with no widgets. See [Terminal Wizard](../user-guide/terminal-cli.md) and [Notebooks](../user-guide/notebooks.md).

### New CLI utilities

| Command | Does |
|---|---|
| `modelforge recommend` | Prints a recommendation without training. |
| `modelforge list-tasks` | Lists the available use cases and their objectives. |
| `modelforge history` | Prints run history. `--legacy` shows terminal-wizard runs. |
| `modelforge tensorboard [LOGDIR]` | Launches TensorBoard (default `./modelforge_output`, `127.0.0.1:6006`). |

See [CLI Commands](../api-reference/cli.md).

## Removed in v4

| v3 feature | v4 status |
|---|---|
| Unsloth provider and the `[unsloth]` extra | Removed. Hugging Face is the only built-in provider. See [Provider Overview](../providers/overview.md). |
| `rlhf` strategy | Removed. In v3 it was DPO with different defaults. Use the Preference objective. |
| `qlora` as a separate strategy | Removed. QLoRA is now a PEFT tier (`qlora_8bit`, `qlora_4bit`) the planner picks for any strategy that supports adapters. See [LoRA and QLoRA](../strategies/qlora.md). |
| `model_configs/*.json` hardware profiles | Removed. Replaced by the planner and Hub search. To change which models are offered, see [Adding a Use Case](../contributing/use-cases.md). |
| Editable hyperparameters in the UI | Removed. The console shows the generated config read-only. |
| `[quantization]` extra | Removed. bitsandbytes is now a core dependency. |
| Built-in evaluation metrics module (perplexity, ROUGE and similar) | Removed. |
| `HUGGINGFACE_TOKEN` in a `.env` file, `CORS_ORIGINS` | No longer read. Use `hf auth login` or `HF_TOKEN`. |
| Docker instructions | Not provided for v4. There is no official image. |

## Migrating from v3

### 1. Use Python 3.13

v3 required Python 3.11 exactly (`>=3.11,<3.12`). v4 requires Python 3.13 or newer (`requires-python = ">=3.13"`, and the repository pins 3.13 in `.python-version`). A v3 virtual environment can't be upgraded in place. Create a new one:

```bash
python3.13 -m venv .venv-modelforge4
source .venv-modelforge4/bin/activate
pip install --upgrade modelforge-finetuning
```

On Windows, install a CUDA build of PyTorch into the new environment first. See [Windows](../installation/windows.md).

### 2. Update your extras

| v3 | v4 |
|---|---|
| `[cli]` (questionary, rich, ipywidgets, ipython) | `[cli]` (click, rich) |
| `[quantization]` (bitsandbytes) | Not needed. bitsandbytes is installed by default. |
| `[unsloth]` | Removed. |
| none | `[gpu]` (liger-kernel, CUDA on Linux only) |

### 3. Update your commands

| v3 | v4 |
|---|---|
| `modelforge` starts the server | `modelforge run` starts the server. `modelforge` alone prints help. |
| Server at `http://localhost:8000`, bound to `127.0.0.1` | Binds `0.0.0.0:8000` by default. Open the printed link with the token, or pass `--host 127.0.0.1`. |
| Hugging Face login required at startup | Not required. Log in only for gated models or Hub pushes. The terminal wizard offers a login if none is cached. |
| `modelforge cli` | Unchanged name. Needs the `[cli]` extra. |
| `modelforge-nb` | Unchanged name. |

Scripts that called the v3 REST API need rewriting: v4 has a new set of endpoints under `/api/`, and every request must carry the token. See [REST API](../api-reference/rest-api.md).

### 4. Prepare datasets for the new column names

v3 configs named a task (`text-generation`, `summarization`, `extractive-question-answering`). v4 uses a use case plus an objective, each with required columns. For example, supervised instruction following expects `prompt` and `completion` (or `instruction` and `response`). A dataset that contains both an alias and its canonical column is rejected as ambiguous. Check your files against [Dataset Formats](../configuration/dataset-formats.md) and use **Validate** before training.

### 5. Know where your data now lives

v3 and v4 store data in different places, and v4 does not read the v3 database.

| | v3 | v4 |
|---|---|---|
| Database | `<v3 data dir>/database/modelforge.sqlite` (SQLAlchemy) | `<v4 data dir>/modelforge.db` (SQLite) |
| Trained models | `<v3 data dir>/models/` | The output directory you choose per run (default `./modelforge_output`) |
| Checkpoints | `<v3 data dir>/model_checkpoints/` | `<output dir>/checkpoint-<step>` |
| TensorBoard logs | `./training_logs` in the directory you started from | `<output dir>/tensorboard/` |

The data directories, from `platformdirs`:

=== "Linux"

    v3 and v4 both use `~/.local/share/ModelForge`. The v4 database is a new file (`modelforge.db`) next to v3's `database/` folder; neither overwrites the other.

=== "macOS"

    v3 and v4 both use `~/Library/Application Support/ModelForge`, with separate database files as on Linux.

=== "Windows"

    v3: `%LOCALAPPDATA%\ModelForge\ModelForge`. v4: `%LOCALAPPDATA%\ForgeOpus\ModelForge`, because v4 passes `ForgeOpus` as the app author.

!!! warning "v3 run history does not carry over"
    v4 starts with an empty run list. There is no import tool for v3 runs or models. Models you trained with v3 are still on disk in the v3 data directory and can be loaded directly with `transformers` (or with PEFT for adapters), but they won't appear under **Runs**, **Artifacts** or the Playground. Delete the v3 data directory when you no longer need it.

### 6. For contributors

The v3 `ModelForge/` package is gone. v4 code is split into top-level packages: `cli/`, `notebook_cli/`, `web/`, `workflow/`, `providers/`, `finetuning_strategies/`, `recommendation_engine/`, `utilities/` and `dataset_manager/`. Providers, strategies, tasks and use cases register themselves by subclassing. A source install builds the browser UI with Node.js and npm. See [Architecture Overview](../contributing/architecture.md) and [Development Setup](../contributing/development.md).

## See also

- [Quick Start](quickstart.md)
- [Post-Installation Setup](../installation/post-installation.md)
- [FAQ](../troubleshooting/faq.md)
