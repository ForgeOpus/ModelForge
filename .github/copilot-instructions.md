# Review Instructions
**Our code reviews are focused, constructive, and respect development velocity. Only flag issues that are critical or significantly impact security, maintainability, or user experience. Do _not_ nitpick style or minor, non-blocking conventions.**

## Project Context (ModelForge v4)
ModelForge (`modelforge-finetuning` on PyPI) is a hardware-aware fine-tuning tool with three interfaces over one shared workflow layer. Python 3.13, Node 22 (`.node-version`), dependencies locked in `uv.lock` and `web/frontend/package-lock.json`. All packages live at the repository root; there is no `ModelForge/` package.

| Path | Role |
|---|---|
| `cli/main.py` | `modelforge` entry point (argparse): `run`, `cli`, `recommend`, `history`, `list-tasks`, `tensorboard` |
| `cli/terminal.py`, `cli/wizard.py`, `cli/display.py` | Optional terminal wizard (`click`, `rich`; `[cli]` extra) |
| `notebook_cli/main.py` | `modelforge-nb` entry point |
| `web/app.py`, `web/auth.py`, `web/playground.py` | FastAPI app, startup-token auth, single-shot inference |
| `web/frontend/` | Vite + React 19 + TypeScript SPA. The app is the single file `src/main.tsx`; in-app docs are `src/docs/` |
| `workflow/` | `WorkflowService` (service.py), `TrainingJobManager` (jobs.py), `assemble_and_train()` (training.py), playground sessions, Pydantic request/event models, `load_core_registries()` |
| `utilities/recommendation_engine.py` | Planning pipeline: memory, PEFT tier, Hub search, config, parallelism |
| `recommendation_engine/language_use_cases/` | `LanguageUseCase` subclasses and `DataFormatSpec` |
| `finetuning_strategies/strategies/`, `finetuning_strategies/tasks/` | `sft`, `dpo`, `grpo`, `qa` strategies; `AutoModel*` tasks |
| `providers/` | `HuggingFaceProvider` |
| `dataset_manager/dataset_factory.py` | Dataset loading and alias-column normalization |
| `utilities/database_manager.py` | SQLite ledgers (`WebRunLedger`, `FinetuneLedger`) |
| `tests/unit_tests/`, `web/frontend/tests/` | pytest suite, Vitest suite |
| `docs/` | Read the Docs site (MkDocs Material, `mkdocs.yml`) |

Use cases, strategies, tasks and providers self-register via `__init_subclass__` on import; new classes must be imported from their package `__init__.py` (see `workflow/registries.py`).

CI (`.github/workflows/ci.yml`) runs `uv sync --frozen --extra dev && uv run pytest`, `npm ci && npm run build && npm test` in `web/frontend`, and the wheel smoke test (`scripts/build_distribution.sh` + `scripts/distribution_smoke.py`). CI is CPU-only and offline (`HF_HUB_OFFLINE=1`, no CUDA devices).

## Review Criteria

### 1. Compatibility & Non-Breaking Changes
- Ensure that proposed changes do **not break existing features or workflows** across all three interfaces (browser, `modelforge cli`, `modelforge-nb`). They share `workflow/`, so a change there affects all of them.
- Detect and flag any breaking changes in `/api/*` endpoints, WebSocket event shapes (`TrainingEvent`, playground `start`/`token`/`done`/`error`), `Recommendation` fields, CLI flags, or the SQLite schema unless explicitly approved as a major version update.
- Database schema changes must migrate existing `modelforge.db` files in place (as `WebRunLedger._initialize()` does with `ALTER TABLE`).
- New request fields must be optional with a default.

### 2. Best Practices & Code Quality
- Only highlight **substantial violations** of best practices, such as blocking calls inside `async` FastAPI handlers (use `asyncio.to_thread`), unsafe shared state across the training thread and request handlers, or inefficient algorithms.
- Training must start through `workflow.training.assemble_and_train()`; flag code that builds trainers from an interface layer directly.
- New use cases, strategies, tasks and providers should follow the registry pattern rather than adding `if slug == ...` branches.
- Keep the frontend app in `web/frontend/src/main.tsx`; flag changes that split it into a new component tree or reintroduce a separate redesign.
- Discourage overengineering and reject premature optimization.
- Highlight any obvious anti-patterns or technical debt introduced.

### 3. Security
- Identify and flag **potential security vulnerabilities**, such as:
  - `/api` routes or WebSockets added without the `TokenAuth` dependency or `?token=` check (only `/api/health` is public).
  - Logging or returning the startup token or Hugging Face tokens.
  - Path traversal in dataset paths, output directories, checkpoint paths or `/api/models/{id}` deletion with `delete_files`.
  - `trust_remote_code=True` when loading models.
  - Use of deprecated/insecure libraries or APIs, hard-coded secrets, inadequate input validation.

### 4. Error Handling & Robustness
- Require **robust error handling** for all external calls, file operations, database actions, and integrations—including the Hugging Face Hub, NVML hardware detection and dataset loading.
- Errors in the training thread must end with a `failed` run status and an `error` event, never a silent hang.
- Flag any missing or ineffective error handling that can lead to silent failures, crashes, or confusing UX.

### 5. Deprecations & Maintenance
- **Identify usage of deprecated functions, methods, or APIs** in `transformers`, `trl`, `peft`, `accelerate` and FastAPI.
- Request migration when dependencies release critical/major/breaking changes or mark features as deprecated.

### 6. Documentation & Clarity
- Ensure that **public functions, classes, and components** are documented where business logic or complex interfaces exist.
- Any user-facing change (flag, default, endpoint, column name, recommendation behavior, error the user must act on) must update **both** the Read the Docs page in `docs/` **and** the matching in-app page in `web/frontend/src/docs/content/*.tsx`. Flag PRs that update only one. Adding or renaming an in-app page also requires updating `web/frontend/src/docs/rtd.ts` and the mapping table in `docs/contributing/contributing.md`.

### 7. Performance & Resource Usage
- Warn about **major performance issues** affecting training, inference, or serving large models—especially VRAM use on local GPUs, loading a second model beside an active training run, or unbounded event buffers.
- Suggest optimizations only for issues that have _clear, measurable impact_.

### 8. Test Coverage
- Check for corresponding tests in `tests/unit_tests/` (pytest) or `web/frontend/tests/` (Vitest) when features are core or bug fixes are non-trivial.
- Tests must run on CPU without network access; flag tests that download from the Hub or require a GPU.
- Request regression tests if changes fix previously reported issues.

***

**Do not:**
- Nitpick variable names, whitespace, or stylistic choices.
- Repeat feedback already caught by `tsc` strict mode, the Vite build, or the test suites.
