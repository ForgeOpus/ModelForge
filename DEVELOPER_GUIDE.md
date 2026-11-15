# ModelForge Developer Guide

Welcome. This guide is for people who are new to this codebase and want to understand how it works before contributing. No prior knowledge of the repo is assumed — just a working knowledge of Python.

---

## What is ModelForge?

ModelForge is a CLI tool that helps you fine-tune large language models (LLMs) without needing to manually figure out which model fits your GPU, which training technique to use, or how to configure it. You tell it what you want to do (e.g., "I want to train a summarization model"), and it figures out the rest based on your hardware.

The key idea: **hardware-aware recommendations + automated training pipeline.**

---

## How to Run It

Install the project with `uv` or `pip` (see `pyproject.toml` for deps). Building or installing from a source checkout requires Node.js and npm so the browser UI can be packaged (see below). Then:

```bash
modelforge run --no-browser  # browser UI at http://127.0.0.1:8000/
modelforge cli               # optional terminal wizard
modelforge recommend         # show recommendations without training
modelforge list-tasks        # show available use cases
modelforge history           # view past training runs
```

The `modelforge` command maps to `cli/main.py:cli` (defined in `pyproject.toml`).

---

## Browser UI build and packaging

`modelforge run` serves the SPA from `web/frontend_dist/`. That directory is generated, not committed. `/`, `/app`, and deep links such as `/app/history` all return `web/frontend_dist/index.html`. Hashed files under `assets/` and files copied from `web/frontend/public/` are served next to it.

### Local development

Node.js and npm must be on `PATH`. The frontend build is locked to `web/frontend/package-lock.json`.

```bash
python scripts/build_frontend.py
# equivalent:
cd web/frontend && npm ci && npm run build

uv sync --extra dev          # or: pip install -e ".[dev]"
modelforge run --no-browser
```

`scripts/build_frontend.py` runs `npm ci` and then `npm run build`. Vite writes the bundle to `web/frontend_dist/`. An editable install runs the same script, so `uv sync` / `pip install -e .` also produces the bundle.

For UI iteration, start the API with `modelforge run --no-browser` and, in another shell, `cd web/frontend && npm run dev`. Vite serves the app on port 5173 and proxies `/api` to `http://127.0.0.1:8000`.

### Wheels and releases

`pip wheel .`, `python -m build`, and `uv build` run the locked frontend build before collecting package data. The wheel contains `web/frontend_dist/index.html`, every file that page references, and the public assets from `web/frontend/public/`.

If Node.js or npm is missing, or if `npm ci` / `npm run build` fails, the package build exits with an error and does not produce a wheel. Installing a wheel that was already built does not require Node. Installing from a source checkout or sdist does, because that path builds the wheel.

```bash
python -m build --wheel
pip install dist/modelforge_finetuning-*.whl
modelforge run --no-browser
```

`tests/unit_tests/frontend_packaging_test.py` builds a wheel from a clean tree that has no prebuilt `frontend_dist`, installs that wheel in a fresh virtualenv, and requests `/`, `/app`, and `/app/history`. It fails if the bundle or a referenced asset is missing. The test needs Python 3.13, Node.js, npm, and network access for `npm ci` and the wheel's Python dependencies.

---

## Directory Map

Here is the top-level layout and what each folder is responsible for:

```
ModelForge/
├── cli/                        # Everything the user sees and interacts with
├── providers/                  # Loads models and tokenizers (HuggingFace, etc.)
├── finetuning_strategies/      # The actual training logic
│   ├── strategies/             # How to train (SFT, future: DPO, GRPO)
│   └── tasks/                  # What kind of model to train (CausalLM, Seq2Seq, etc.)
├── recommendation_engine/      # Maps user goals + hardware → training plan
│   ├── language_use_cases/     # Translates human goals into technical parameters
│   └── validation_models/      # Pydantic output types for recommendations
├── utilities/                  # Shared helpers: hardware detection, DB, enums, parallelism config
├── dataset_manager/            # Loads datasets from disk or HuggingFace Hub
├── web/                        # Browser UI server and Vite frontend
│   ├── frontend/               # React source; build with npm ci && npm run build
│   └── frontend_dist/          # Generated bundle served by modelforge run
├── scripts/build_frontend.py   # Locked frontend build used by packaging
└── tests/unit_tests/           # All tests live here
```

---

## The Big Picture: How a Training Run Works

Before diving into files, here's the flow from the moment a user runs `modelforge run`:

```
User answers questions in the wizard
        ↓
Hardware is detected (RAM, VRAM, CPU)
        ↓
RecommendationEngine picks: model + strategy + PEFT config
        ↓
User reviews and confirms
        ↓
assemble_and_train() kicks off training
        ↓
Model is saved + run is logged to SQLite
```

Let's walk through each piece.

---

## Part 1: The CLI (`cli/`)

The CLI has four files. Each has a clear job.

### `cli/main.py` — Entry Point

This is where the `modelforge` commands live. It uses [Click](https://click.palletsprojects.com/) to define commands.

```python
@cli.command()
def run():
    ...
```

The `run` command drives the full training workflow. It calls out to `wizard.py` for user input, `recommendation_engine` for the plan, and `runner.py` to execute.

### `cli/wizard.py` — Interactive Prompts

This file handles all the back-and-forth with the user: which GPUs to use, which use case, which objective, which model, which dataset, where to save output.

Key functions:
- `check_hf_login()` — confirms the user is logged in to HuggingFace
- `select_gpus(hardware)` — lets user pick which GPUs to use
- `select_use_case()` — shows available use cases, returns slug + objective
- `select_model(recommendation)` — shows recommended models or prompts for custom ID
- `validate_and_load_dataset(data_format_spec)` — loads and validates the dataset

### `cli/runner.py` — Orchestrates Training

Once the user has confirmed everything, `assemble_and_train()` in this file takes over. It:
1. Gets the right strategy class from the registry
2. Gets the right provider from the registry
3. Builds the LoRA config from the recommendation
4. Calls `strategy.finetune(...)` to kick off training

This is the bridge between "user made their choices" and "training is actually happening."

### `cli/display.py` — Pretty Output

Uses [Rich](https://github.com/Textualize/rich) to format tables, panels, and progress bars. Also contains `RichTrainingCallback`, which hooks into the HuggingFace trainer to show live training stats.

---

## Part 2: The Registry Pattern

This is the most important architectural pattern in the codebase. **Almost every extensible concept uses it.**

The idea: when you define a subclass, it automatically registers itself. No manual registration needed.

Here's a simplified version of how it works:

```python
class FinetuningStrategy:
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # called automatically when a subclass is defined
        FinetuningStrategy._registry[cls.slug] = cls
```

So when Python loads `sft.py` and sees `class SFT(FinetuningStrategy)`, it immediately runs `__init_subclass__`, which stores `SFT` in the registry under its slug. No one has to do anything else.

**Five things use this pattern:**

| Concept | Base Class | File |
|---|---|---|
| Providers | `Provider` | `providers/provider.py` |
| Tasks | `FinetuningTask` | `finetuning_strategies/tasks/task.py` |
| Strategies | `FinetuningStrategy` | `finetuning_strategies/strategies/strategy.py` |
| Use Cases | `LanguageUseCase` | `recommendation_engine/language_use_cases/use_case.py` |

Factory classes (`ProviderFactory`, `TaskModels`, `FinetuningStrategies`) provide clean methods to look things up from these registries.

**Why this matters for contributing:** if you want to add a new strategy (e.g., DPO), you subclass `FinetuningStrategy`, define the required class attributes, implement `finetune()`, and it's automatically available everywhere — no registration needed.

---

## Part 3: Use Cases (`recommendation_engine/language_use_cases/`)

A **use case** represents a human-level goal like "I want to train a model to summarize text." It knows:
- Which task it maps to (e.g., summarization → seq2seq)
- Which training objectives are compatible
- What the data should look like for each objective

### The Base Class: `use_case.py`

`LanguageUseCase` is the abstract base. Every use case subclass must define:

```python
class SummarizationUseCase(LanguageUseCase):
    name = "Summarization"
    slug = "summarization"
    description = "..."
    task_slug = "seq2seq"                         # which task to use
    hub_task_filter = "summarization"             # HF Hub filter string
    compatible_objectives = [TrainingObjective.SUPERVISED]
    data_formats = {
        TrainingObjective.SUPERVISED: DataFormatSpec(...)
    }
```

`DataFormatSpec` describes what columns the dataset needs. It's surfaced to the user in the wizard to help them understand what format their data should be in.

### Current Use Cases

| File | Use Case | Task |
|---|---|---|
| `text_generation.py` | Open-ended text generation | `causal_lm` |
| `instruction_following.py` | Instruction/chat fine-tuning | `causal_lm` |
| `summarization.py` | Document summarization | `seq2seq` |
| `extractive_qa.py` | Question answering | `causal_lm` |

---

## Part 4: Tasks (`finetuning_strategies/tasks/`)

A **task** maps to a HuggingFace model class (e.g., `AutoModelForCausalLM`). It's a thin wrapper that tells the system which model architecture to load.

```python
class CausalLM(FinetuningTask):
    name = "Causal Language Modeling"
    slug = "causal_lm"
    description = "..."
    model_class = AutoModelForCausalLM
```

When the provider loads a model, it uses `task.model_class` to know what kind of model to instantiate.

Current tasks: `causal_lm`, `seq2seq`, `token_classification`.

---

## Part 5: Strategies (`finetuning_strategies/strategies/`)

A **strategy** is how you train. Right now there's only `SFT` (supervised fine-tuning). DPO and GRPO are planned.

### The Base Class: `strategy.py`

`FinetuningStrategy` defines what every strategy must provide:

- `slug` — identifier
- `extra_model_copies` — how many extra full weight copies training needs (SFT = 0; DPO needs a reference model = 1)
- `batch_multiplier` — memory multiplier vs. SFT baseline
- `supported_tasks` — list of task slugs this strategy works with
- `finetune(model_id, dataset, task_slug, provider, peft_config, callbacks)` — runs training
- `build_recommended_config(hardware, peft_tier)` — returns a hardware-tuned config

### `sft.py` — The SFT Strategy

`SFT` uses TRL's `SFTTrainer`. Key methods:

- `build_sft_config(hardware, peft_tier)` → builds an `SFTConfig` with sensible defaults for the hardware
- `finetune(...)` → loads model + tokenizer via provider, wraps with LoRA, trains with `SFTTrainer`
- `validate_data_format(dataset, data_format_spec)` → checks that the dataset has the required columns

`SFTRecommendedConfig` and `LoraRecommendedConfig` are Pydantic models that hold the typed config output.

---

## Part 6: The Recommendation Engine (`utilities/recommendation_engine.py`)

This is the brain of the system. Given:
- A use case slug (e.g., `"summarization"`)
- A training objective (e.g., `TrainingObjective.SUPERVISED`)
- Hardware info (GPU VRAM, RAM, etc.)

It returns a `Recommendation` object that tells you exactly which models to try, which strategy to use, which PEFT tier fits your hardware, and why.

### The Pipeline

`RecommendationEngine.recommend()` runs these steps internally:

1. **Compute raw VRAM** — sum across all selected GPUs (or RAM for CPU-only/MPS)
2. **Resolve use case and strategy** — look up in registries by slug
3. **Compute adjusted VRAM** — account for `extra_model_copies` and `batch_multiplier` of the strategy
4. **Select PEFT tier** — pick from `full_finetune`, `lora_bf16`, `qlora_8bit`, `qlora_4bit` based on VRAM
5. **Query HuggingFace Hub** — find models matching the use case's `hub_task_filter`
6. **Select provider** — currently always `huggingface`
7. **Build strategy config** — hardware-tuned `SFTRecommendedConfig`
8. **Select parallelism strategy** — `DATA_PARALLEL` vs `FSDP`, based on GPU count, the smallest GPU's VRAM, and the chosen model size (see Part 7)
9. **Build rationale** — human-readable explanation of every decision (including a `parallelism_decision` line)

### PEFT Tier Logic

| Tier | Memory per Param | Triggered When |
|---|---|---|
| `full_finetune` | 18 bytes | Very high VRAM (for small models) |
| `lora_bf16` | ~2 bytes (frozen) | 16+ GB adjusted VRAM |
| `qlora_8bit` | ~1 byte (frozen) | 8–16 GB adjusted VRAM |
| `qlora_4bit` | ~0.5 bytes (frozen) | Below 8 GB adjusted VRAM |

The full breakdown is documented in `memory_strategy_guide.md` at the project root.

### Output: `Recommendation`

Defined in `recommendation_engine/validation_models/recommendation.py`. It contains:

```python
class Recommendation(BaseModel):
    use_case: LanguageUseCase
    strategy_slug: str
    peft_tier: str
    provider_slug: str
    recommended_models: list[RecommendedModel]
    strategy_config: SFTRecommendedConfig
    peft_config: LoraRecommendedConfig
    parallelism_strategy: ParallelismStrategy  # DATA_PARALLEL (default) or FSDP
    rationale: Rationale
```

---

## Part 7: Distributed Training (Parallelism Strategies)

When training spans multiple GPUs, ModelForge supports two parallelism strategies. The recommendation engine picks one automatically (Part 6, step 8), and the user can override it.

| Strategy | What it does | When it's chosen |
|---|---|---|
| `DATA_PARALLEL` | Each GPU holds a full model replica; gradients are synced. The default. | Single GPU, any PEFT tier, or multi-GPU when a full replica fits on the smallest GPU. |
| `FSDP` | Fully Sharded Data Parallelism — shards weights, gradients, and optimizer state across GPUs, so models too big for one GPU can be trained. | ≥2 CUDA GPUs **and** full fine-tuning **and** a full replica does *not* fit on the smallest GPU. |

Tensor parallelism and DeepSpeed/ZeRO are deliberately not implemented — PyTorch-native FSDP covers fine-tuning needs without a heavy extra dependency.

### The three modules

- `utilities/parallelism_strategy.py` — the `ParallelismStrategy` enum (`DATA_PARALLEL`, `FSDP`).
- `utilities/fsdp_config.py` — `build_fsdp_config()` returns HuggingFace `TrainingArguments` overrides (`fsdp`, `fsdp_config`, …) that switch FSDP on.
- `utilities/parallelism_validator.py` — `validate_parallelism_compatibility()` enforces the one hard rule below.

### The one hard rule: FSDP requires full fine-tuning

FSDP cannot shard LoRA/QLoRA adapter parameters, so it is only valid with `peft_approach == "full_finetune"`. `validate_parallelism_compatibility()` raises `ValueError` for any other combination. It runs in two places:

- `WorkflowService.recommend()` — defense-in-depth, right after the engine produces a recommendation.
- `TrainingJobManager.start_run()` — after applying any user override, before a training thread is spawned.

### How FSDP is actually turned on

FSDP **must** be driven through the Trainer's own `fsdp` / `fsdp_config` arguments — *not* by mutating `AcceleratorState`. The TRL/transformers Trainer builds its own `Accelerator` from those arguments (setting `ACCELERATE_USE_FSDP` internally); a plugin stashed on the shared state beforehand is silently overwritten, leaving the run quietly data-parallel.

So `assemble_and_train()` (in `workflow/training.py`) merges `build_fsdp_config()` into the strategy config dict **before** `strategy.set_config(...)` — but only when the recommendation selected FSDP *and* the run actually has ≥2 processes (a stored or resumed recommendation may have been made against a different GPU set). The merged overrides:

- enable `fsdp="full_shard auto_wrap"` with a size-based auto-wrap policy,
- swap `gradient_checkpointing` for FSDP `activation_checkpointing` (enabling both raises in the Trainer and adds a redundant all-gather),
- disable FSDP-incompatible options (`use_liger_kernel`, `activation_offloading`).

Mixed precision is **not** set here — the Trainer derives it from the config's existing `bf16` / `fp16` flags.

### Overriding the choice

`TrainingRunRequest.parallelism_strategy_override` (`ParallelismStrategy | None`) lets a user force a strategy. When set, `start_run()` applies it to the recommendation via `model_copy(...)` and then re-validates, so an invalid combo (e.g. FSDP + LoRA) fails fast before any thread starts.

---

## Part 8: Providers (`providers/`)

A **provider** knows how to load a model and tokenizer. The only current provider is `HuggingFaceProvider`.

```python
provider = ProviderFactory.get_provider("huggingface")()
model = provider.load_model(model_id, task.model_class)
tokenizer = provider.load_tokenizer(model_id)
```

Adding a new provider (e.g., for a private model hub) means subclassing `Provider`, defining `slug`, and implementing `load_model()` and `load_tokenizer()`.

---

## Part 9: Hardware Detection (`utilities/hardware_detector.py`)

`HardwareDetector.detect_hardware()` returns a `HardwareSpecs` Pydantic model with:

- `gpus` — list of GPU info (name, VRAM in bytes)
- `total_ram_bytes`
- `available_ram_bytes`
- `cpu_count`
- `disk_space_bytes`
- `has_mps` — for Apple Silicon

It uses `psutil` for CPU/RAM/disk and `pynvml` (nvidia-ml-py) for GPU VRAM. Apple Silicon MPS is handled separately in `get_device_info_mps()`.

---

## Part 10: Dataset Loading (`dataset_manager/dataset_factory.py`)

`DatasetFactory.load(path_or_id)` handles all dataset loading in one place:

- CSV, JSON, Parquet, Arrow files from disk
- Directories of text files
- HuggingFace Hub dataset IDs (e.g., `"tatsu-lab/alpaca"`)

It auto-detects the format and returns a HuggingFace `Dataset` object.

---

## Part 11: Database Tracking (`utilities/database_manager.py`)

Two singletons track state in an SQLite database:

- **`DatabaseManager`** — stores `ModelMetadata` (model ID, task, provider, creation time)
- **`FinetuneLedger`** — logs completed finetune runs (model, dataset, strategy, output path, timestamp)

Both are singletons — only one instance exists per process. The `history` command reads from `FinetuneLedger` to show past runs.

---

## Part 12: Tests (`tests/unit_tests/`)

Tests use `pytest`. Each module covers one system. Run them with:

```bash
pytest tests/
```

Key things to know:

- **Registry isolation**: Each test that touches registries may need to avoid polluting other tests. Use `clear_registry()` if a factory provides one, or use a fresh import.
- **DB singleton**: `DatabaseManager` and `FinetuneLedger` are singletons, so tests share state unless you reset them.
- **SFT CPU config**: Some SFT tests configure `bf16=False` to run on CPU.
- **DatasetFactory mocking**: Tests that call `DatasetFactory.load()` often mock `load_dataset` from HuggingFace.

Test files:

| File | What it tests |
|---|---|
| `database_manager_test.py` | SQLite ledger and model metadata |
| `dataset_factory_test.py` | All dataset loading paths |
| `finetuning_strategy_test.py` | Strategy registry and attribute validation |
| `finetuning_task_test.py` | Task registry and auto-registration |
| `hardware_detector_test.py` | HardwareSpecs validation and detection logic |
| `model_factory_test.py` | `TaskModels` factory methods |
| `provider_test.py` | Provider registry and HuggingFace provider |
| `recommendation_engine_test.py` | Full engine pipeline end-to-end, incl. parallelism selection |
| `sft_test.py` | SFT config building and data validation |
| `strategy_factory_test.py` | `FinetuningStrategies` factory methods |
| `fsdp_config_test.py` | `build_fsdp_config()` overrides and `SFTConfig` acceptance |
| `parallelism_validator_test.py` | FSDP + PEFT compatibility validation |
| `workflow_jobs_test.py` | Job manager lifecycle, parallelism override + validation |

---

## How to Add New Things

### Add a new use case

1. Create a file in `recommendation_engine/language_use_cases/`
2. Subclass `LanguageUseCase`
3. Define: `name`, `slug`, `description`, `task_slug`, `hub_task_filter`, `compatible_objectives`, `data_formats`
4. Import it in `recommendation_engine/language_use_cases/__init__.py` so it gets registered

### Add a new strategy (e.g., DPO)

1. Create `finetuning_strategies/strategies/dpo.py`
2. Subclass `FinetuningStrategy`
3. Define: `name`, `slug`, `extra_model_copies`, `batch_multiplier`, `supported_tasks`
4. Implement: `finetune()`, `build_recommended_config()`
5. Import it in `finetuning_strategies/strategies/__init__.py`

### Add a new parallelism strategy (e.g., tensor parallelism)

Parallelism strategies are a small enum, not a registry. To add one:

1. Add a member to `ParallelismStrategy` in `utilities/parallelism_strategy.py`
2. Add a config builder (like `build_fsdp_config()`) in `utilities/fsdp_config.py` if it needs trainer-arg overrides
3. Extend `RecommendationEngine._select_parallelism_strategy()` (and `_build_parallelism_decision()`) with the conditions that pick it
4. Wire it into `assemble_and_train()` in `workflow/training.py`
5. Add any compatibility rules to `utilities/parallelism_validator.py`

### Add a new task

1. Create a file in `finetuning_strategies/tasks/`
2. Subclass `FinetuningTask`
3. Define: `name`, `slug`, `description`, `model_class` (a HuggingFace AutoModel class)
4. Import it in `finetuning_strategies/tasks/__init__.py`

### Add a new provider

1. Create a file in `providers/`
2. Subclass `Provider`
3. Define: `slug`
4. Implement: `load_model(name, model_class, **kwargs)`, `load_tokenizer(name)`
5. Import it in `providers/__init__.py`

---

## Key Dependencies at a Glance

| Library | What it does here |
|---|---|
| `transformers` | Loads models and tokenizers via `AutoModel*` / `AutoTokenizer` |
| `trl` | `SFTTrainer` and `SFTConfig` for supervised fine-tuning |
| `peft` | `LoraConfig` for parameter-efficient fine-tuning |
| `accelerate` | Distributed training (data parallel + FSDP), device management, `notebook_launcher` |
| `bitsandbytes` | 4-bit and 8-bit quantization for low-VRAM hardware |
| `huggingface_hub` | HF Hub API queries, login checks, model discovery |
| `pydantic` | Validation for hardware specs, configs, recommendations |
| `click` | CLI framework for `modelforge` commands |
| `rich` | Terminal formatting: tables, panels, progress bars |
| `psutil` | CPU, RAM, disk detection |
| `nvidia-ml-py` | GPU VRAM detection via `pynvml` |

---

## File Quick-Reference

| If you want to... | Look in... |
|---|---|
| Add a CLI command | `cli/main.py` |
| Change wizard prompts | `cli/wizard.py` |
| Change how training is orchestrated | `cli/runner.py` |
| Change terminal output formatting | `cli/display.py` |
| Add a new training strategy | `finetuning_strategies/strategies/` |
| Add a new model task type | `finetuning_strategies/tasks/` |
| Add a new use case | `recommendation_engine/language_use_cases/` |
| Change recommendation logic | `utilities/recommendation_engine.py` |
| Change parallelism (data parallel / FSDP) behavior | `utilities/parallelism_strategy.py`, `utilities/fsdp_config.py` |
| Change FSDP/PEFT compatibility rules | `utilities/parallelism_validator.py` |
| Change how models/tokenizers load | `providers/` |
| Change dataset loading | `dataset_manager/dataset_factory.py` |
| Change hardware detection | `utilities/hardware_detector.py` |
| Change DB tracking | `utilities/database_manager.py` |
| Understand VRAM math | `memory_strategy_guide.md` |
| Understand recommendation design | `recommendation_engine_spec.md` |

---

That's the full picture. The codebase is designed so that adding new things (strategies, tasks, use cases, providers) means subclassing and defining attributes — not modifying core logic. When in doubt, look at an existing implementation of the thing you're adding and follow the same pattern.
