# Architecture

This page describes how ModelForge v4 is put together: the packages in the repository, how a training request travels from the browser to the trainer and back, the registry pattern behind every extensible concept, the recommendation pipeline, storage, hardware detection and the playground. Read it before changing anything outside a single strategy or use case.

## Module map

All packages sit at the repository root. There is no `ModelForge/` package any more.

| Path | Responsibility |
|---|---|
| `cli/main.py` | `modelforge` entry point (argparse). Subcommands `run`, `cli`, `recommend`, `history`, `list-tasks`, `tensorboard`. |
| `cli/terminal.py`, `cli/wizard.py`, `cli/display.py` | Optional terminal wizard (`modelforge cli`), built on `click` and `rich`. |
| `cli/runner.py` | Re-exports `workflow.training.assemble_and_train` for old imports. |
| `notebook_cli/main.py` | `modelforge-nb` entry point. Plain `input()` prompts, no widgets. |
| `web/app.py` | FastAPI app: REST endpoints, WebSockets, and the SPA from `web/frontend_dist/`. |
| `web/auth.py` | `TokenAuth`: the startup token checked on every `/api` call. |
| `web/playground.py` | Single-shot inference endpoint (`POST /api/playground/infer`). |
| `web/frontend/` | Vite + React source. The app is `src/main.tsx`; in-app docs are `src/docs/`. |
| `workflow/service.py` | `WorkflowService`: UI-independent operations (hardware, use cases, recommendations, dataset and model validation). |
| `workflow/jobs.py` | `TrainingJobManager`: background training threads, events, cancellation, resume, artifact registration. |
| `workflow/training.py` | `assemble_and_train()`: the one execution boundary every interface uses to start training. |
| `workflow/playground.py` | `PlaygroundManager`: streaming chat sessions over WebSocket. |
| `workflow/models.py` | Pydantic request and event models (`RecommendationRequest`, `TrainingRunRequest`, `TrainingEvent`, ...). |
| `workflow/registries.py` | `load_core_registries()`: imports every built-in use case, task, strategy and provider. |
| `recommendation_engine/language_use_cases/` | Use cases (`LanguageUseCase` subclasses) and `DataFormatSpec`. |
| `recommendation_engine/validation_models/` | `Recommendation`, `RecommendedModel`, `Rationale`, `VramEstimate`. |
| `utilities/recommendation_engine.py` | `RecommendationEngine.recommend()`: the planning pipeline. |
| `utilities/hardware_detector.py` | `HardwareDetector`: CPU, RAM, disk and GPU detection. |
| `utilities/database_manager.py` | SQLite ledgers. |
| `utilities/parallelism_*.py`, `utilities/fsdp_config.py` | Data parallel vs FSDP selection, validation and trainer overrides. |
| `utilities/precision.py`, `chat_format.py`, `qa_inference.py`, `tensorboard.py`, `training_objective.py` | Shared helpers. |
| `finetuning_strategies/strategies/` | `FinetuningStrategy` and the `sft`, `dpo`, `grpo` and `qa` strategies. |
| `finetuning_strategies/tasks/` | `FinetuningTask`: which `AutoModel*` class to load. |
| `providers/` | `Provider` and `HuggingFaceProvider`: load models and tokenizers. |
| `dataset_manager/dataset_factory.py` | `DatasetFactory`: load local files or Hub datasets, rename alias columns. |
| `scripts/` | Frontend build, distribution build and smoke test, training-flow harness. |
| `tests/unit_tests/` | pytest suite. |

## Layers

Three interfaces share one workflow layer. Nothing below the interfaces imports from them.

```text
  Browser SPA            modelforge cli           modelforge-nb
  (web/frontend)         (cli/terminal.py)        (notebook_cli/main.py)
       |  HTTP + WS              |                        |
       v                         |                        |
  web/app.py  ------------------+------------------------+
       |                         |                        |
       v                         v                        v
  workflow/  WorkflowService, TrainingJobManager*, assemble_and_train()
       |
       v
  utilities/recommendation_engine.py   dataset_manager/   utilities/database_manager.py
       |
       v
  recommendation_engine/ (use cases)  finetuning_strategies/ (strategies, tasks)  providers/
       |
       v
  transformers / trl / peft / accelerate / bitsandbytes / huggingface_hub

  * The terminal wizard calls RecommendationEngine and assemble_and_train()
    directly. It does not use TrainingJobManager and records runs in the
    legacy FinetuneLedger table.
```

## Request flow: browser to trainer and back

This is what happens when you click **Start training** in the browser console.

```text
Browser (main.tsx)
  1. GET  /api/system                  -> use cases, strategies, providers
  2. POST /api/recommendations         -> Recommendation (plan)
  3. POST /api/datasets/validate       -> columns, row_count, error
  4. POST /api/runs  {..., recommendation}
         |
         v
web/app.py  create_run()
  - TokenAuth.dependency checks ?token= or X-ModelForge-Token
         |
         v
workflow/jobs.py  TrainingJobManager.start_run()
  - refuses if playground inference holds resource_gate
  - refuses if a run is already active (max_active_jobs = 1)
  - rebuilds or re-validates the Recommendation
  - applies parallelism_strategy_override, validate_parallelism_compatibility()
  - WebRunLedger.create_run()  -> run id, status "queued"
  - starts a daemon thread: _run_training()
         |
         v   (background thread)
  - status "running", emit("status")
  - DatasetFactory.load(dataset_path)
  - workflow/training.py  assemble_and_train()
        - load_core_registries()
        - set CUDA_VISIBLE_DEVICES from the selected GPUs
        - DatasetFactory.normalize_columns()  alias -> canonical columns
        - strategy.set_config(strategy_config + TensorBoard + FSDP overrides)
        - build LoraConfig unless peft_approach == "full_finetune"
        - accelerate.notebook_launcher(strategy.finetune, num_processes = GPU count)
              -> provider loads model/tokenizer, TRL or HF trainer runs
              -> JobEventCallback.on_log / on_save / on_step_end
  - status "completed" | "cancelled" | "failed"
  - on success: write <output_dir>/modelforge_config.json, WebRunLedger.record_model()
  - archive all events to web_run_events
         |
         v
TrainingJobManager.emit(run_id, type, message, payload)
  - appends to a per-run replay buffer (5000 events)
  - pushes to every subscriber queue
         |
         v
WS /api/runs/{run_id}/events?token=...
  - one queue per connection, so several tabs can watch one run
  - sends {"type": "heartbeat"} after 1 s with no events
Browser merges GET /api/runs/{run_id}/events (history) with live events
```

Points worth knowing:

- Training runs in a **thread inside the server process**, not a separate service. `notebook_launcher` runs the function in-process for one device and starts one worker process per GPU when more than one is selected. Stopping `modelforge run` stops the run.
- On startup, `TrainingJobManager` calls `WebRunLedger.mark_interrupted_runs()`, which marks any run left `queued` or `running` as `failed`.
- Cancellation sets a flag. `JobEventCallback.on_step_end` sees it and sets `control.should_training_stop`, so the run stops after the current step.
- Event types emitted are `status`, `log` (payload `step`, `max_steps`, `logs`), `checkpoint` (payload `path`, `step`), `cancel`, `completed`, `cancelled` and `error`. The WebSocket adds `heartbeat`. `ts` is Unix seconds. See [Responses](../api-reference/responses.md).
- `assemble_and_train()` is the only place training starts. The terminal wizard, the notebook UI and `scripts/run_training_flow_tests.py` all call it, so data normalization, TensorBoard setup and FSDP overrides behave the same everywhere.
- The notebook UI uses `WorkflowService` and `TrainingJobManager` directly, without HTTP, so its runs appear in the browser's run history.

## The registry pattern

Use cases, strategies, tasks and providers each have a base class with a class-level `_registry` dict. `__init_subclass__` validates the subclass's class attributes and stores it under its `slug` the moment the class is defined. A duplicate slug raises `ValueError`.

| Concept | Base class | File | Built-in slugs |
|---|---|---|---|
| Use case | `LanguageUseCase` | `recommendation_engine/language_use_cases/use_case.py` | `text_generation`, `instruction_following`, `summarization`, `extractive_qa` |
| Strategy | `FinetuningStrategy` | `finetuning_strategies/strategies/strategy.py` | `sft`, `dpo`, `grpo`, `qa` |
| Task | `FinetuningTask` | `finetuning_strategies/tasks/task.py` | `causal_language_modeling`, `seq2seq_modeling`, `token_classification`, `question_answering` |
| Provider | `Provider` | `providers/provider.py` | `huggingface` |

Registration happens on import, so a new class must be imported somewhere. Each package's `__init__.py` imports its built-ins, and `workflow/registries.py:load_core_registries()` imports every package. `WorkflowService()` and `assemble_and_train()` both call it.

Lookups go through the registries or thin factories: `LanguageUseCase.get_use_case_by_slug()`, `FinetuningStrategies.get_strategy_class()` (`finetuning_strategies/strategy_factory.py`), `TaskModels.get_task_class()` (`finetuning_strategies/model_factory.py`) and `ProviderFactory.get_provider()` (`providers/provider_factory.py`).

The pieces connect like this:

- A use case names one `task_slug` and a list of `compatible_objectives`.
- An objective maps to a strategy through `OBJECTIVE_STRATEGY_MAP`: `supervised` to `sft`, `preference` to `dpo`, `reinforcement` to `grpo`. A use case can override this with a `strategy_slug` attribute; `extractive_qa` sets `strategy_slug = "qa"`.
- The strategy must list the task in `supported_tasks` and the provider in `supported_providers`.
- `WorkflowService.list_use_cases()` hides objectives with no registered, task-compatible strategy, and hides a use case entirely if none of its objectives can run.

Parallelism is not a registry. `ParallelismStrategy` is a two-member enum (`data_parallel`, `fsdp`) in `utilities/parallelism_strategy.py`.

See [Adding a use case](use-cases.md), [Custom strategies](../strategies/custom-strategies.md) and [Custom providers](../providers/custom-providers.md).

## Recommendation pipeline

`RecommendationEngine.recommend(use_case_slug, training_objective, hardware, n=5, selected_model=None)` in `utilities/recommendation_engine.py` returns a `Recommendation`. `WorkflowService.recommend()` wraps it: it detects hardware, filters to the selected GPUs, validates a typed-in model ID against the Hub, and re-checks parallelism compatibility.

```text
1. Raw memory        sum VRAM of selected GPUs
                     Apple MPS -> 8 GB effective; no GPU -> CPU-only
2. Resolve           use case -> task_slug; objective (or strategy_slug) -> strategy
                     check objective is compatible and strategy supports the task
3. Adjusted memory   (raw - (batch_multiplier - 1) * 2 GB) / (1 + extra_model_copies)
4. Tier capacities   P_max per tier, reserving 2 GB activations (+0.1 GB adapter):
                       full_finetune  18 B/param
                       lora_bf16       2 B/param
                       qlora_8bit      1 B/param
                       qlora_4bit    0.5 B/param
                     adapter tiers are capped by the smallest GPU
                     CPU-only: qlora_4bit only, min(500M, RAM/3) params
                     MPS: no 8-bit or 4-bit
5. Candidates        HfApi.list_models(pipeline_tag = hub_task_filter, sort = downloads)
                     skip tiny/random/dummy and internal-testing repos
                     skip models without safetensors parameter counts
                     instruction_following keeps only chat/instruct models
                     if none fit, retry once with a 50% larger bound
                     (a typed-in model skips the search)
6. Pick              largest shortlisted model, then least-quantized tier that fits it
7. Config            strategy_cls.build_recommended_config(tier, hardware, model)
                     -> strategy_config, peft_config
8. Parallelism       FSDP only if >= 2 CUDA GPUs, full_finetune, and a full
                     replica (18 B/param + 2 GB) does not fit the smallest GPU;
                     otherwise data_parallel
9. Rationale         vram_summary, use_case_resolution, peft_decision,
                     model_filter, parallelism_decision, warnings
```

The output also carries `data_format` (the use case's `DataFormatSpec` for the objective) and `vram_estimate`. The browser sends the whole recommendation back with `POST /api/runs`; the job manager checks that it still matches the use case and objective, and re-plans if you picked a different model. Field-by-field detail is in [Responses](../api-reference/responses.md); the reasoning is in [How planning works](../configuration/configuration-guide.md) and [Hardware and memory](../configuration/hardware-profiles.md).

## Storage

`utilities/database_manager.py` keeps everything in one SQLite file, `modelforge.db`, in `platformdirs.user_data_dir("ModelForge", "ForgeOpus")`. On Linux that is `~/.local/share/ModelForge/`.

| Class | Tables | Used by |
|---|---|---|
| `WebRunLedger` | `web_runs`, `web_checkpoints`, `web_run_events`, `web_models` | Browser and notebook (through `TrainingJobManager`) |
| `FinetuneLedger` | `finetunes` | Terminal wizard; shown by `modelforge history --legacy` |
| `DatabaseManager` | `models` | Older model-metadata store; not used by the v4 interfaces |

- `web_runs` holds status, the full recommendation as `config_json`, `launch_json` (GPU indices, remote flag), TensorBoard directory and error.
- `web_checkpoints` is filled by `JobEventCallback.on_save`. Resume only accepts a checkpoint registered here for that run.
- `web_run_events` stores each run's event timeline as JSON once the run finishes, so a run's history survives a restart.
- `web_models` records finished models, with size on disk and Hub push status.

`WebRunLedger` opens a new connection per call and accepts a `db_path`, which tests use. `FinetuneLedger` and `DatabaseManager` are per-process singletons. Schema changes are made in place in `_initialize()` with `ALTER TABLE`, so existing databases keep working.

Each finished run also writes `modelforge_config.json` into its output directory (base model, task, use case, objective, strategy, PEFT approach, `AutoModel` class, pipeline task). The playground needs it to reload an adapter with the right head.

## Hardware detection

`HardwareDetector.detect_hardware()` in `utilities/hardware_detector.py` returns a `HardwareSpecs` model:

| Field | Source |
|---|---|
| `physical_core_count` | `psutil.cpu_count(logical=False)` |
| `ram_gb` | `psutil.virtual_memory().total` |
| `disk_storage_gb`, `available_disk_storage_gb` | `psutil.disk_usage("/")` |
| `gpus` | list of `{"name", "total_memory_gb"}` |

On CUDA machines GPUs come from NVML (`nvidia-ml-py`). If NVML fails the list is empty and the machine is treated as CPU-only. When `torch.backends.mps.is_available()` is true, a single entry `Apple Silicon GPU` with `total_memory_gb: "N/A"` is returned; the engine treats that as 8 GB. Detection runs on every recommendation and run start, never cached. See [Hardware and memory](../configuration/hardware-profiles.md).

## Playground

There are two implementations.

- **Streaming sessions** (`workflow/playground.py`, used by the Playground page). `POST /api/playground/sessions` loads one model from an output directory; `WS /api/playground/{session_id}/chat` streams tokens back as `start`, `token`, `done` or `error` messages and accepts `{"type": "stop"}`. `GET /api/playground/status` reports availability. One session is kept at a time; opening a different model closes the previous one. Sessions are refused while a training run is active, and an open session holds `TrainingJobManager.resource_gate`, so a run cannot start until the session is closed.
- **Single-shot inference** (`web/playground.py`). `POST /api/playground/infer` takes a run ID (and optionally a registered checkpoint), estimates memory, plans placement with `accelerate.infer_auto_device_map`, answers once, and frees memory. It supports all four tasks, including extractive QA and token classification. While training is active it runs on CPU (or disk offload, under `MODELFORGE_OFFLOAD_DIR` or the system temp directory) and refuses any GPU placement.

Both pick the model class from the trained task: sessions read `task_slug` from `modelforge_config.json`, single-shot inference reads it from the run record. PEFT adapters are loaded onto their base model. See [Playground](../user-guide/playground.md).

## Authentication

`modelforge run` creates a `TokenAuth` with a random token (or the `--token` value) and prints `http://127.0.0.1:8000/app?token=...`. Every `/api` route except `/api/health` requires the token as a `token` query parameter or `X-ModelForge-Token` header; WebSockets take it as `?token=` and close with code 1008 if it is wrong. CORS allows all origins, so the token is the only access control. See [REST API](../api-reference/rest-api.md).

## See also

- [Development setup](development.md)
- [Adding a use case](use-cases.md)
- [Custom strategies](../strategies/custom-strategies.md)
- [Custom providers](../providers/custom-providers.md)
- [REST API](../api-reference/rest-api.md)
