# Contributing to ModelForge

This page explains how to get a change into ModelForge: the fork-and-pull-request workflow, what a pull request must include, and the rule that keeps the Read the Docs site and the in-app docs in step. For setting up a working checkout, see [Development setup](development.md).

## Ways to contribute

- Report a bug or request a feature using the GitHub issue templates.
- Fix a bug or improve error messages.
- Add a use case, strategy, task or provider. Each one is a subclass that registers itself. See [Adding a use case](use-cases.md), [Custom strategies](../strategies/custom-strategies.md) and [Custom providers](../providers/custom-providers.md).
- Improve the documentation, both here and in the app.
- Add tests for code paths that have none.

## Code of conduct

Everyone taking part in the project is expected to follow the [Code of Conduct](https://github.com/ForgeOpus/ModelForge/blob/main/CODE_OF_CONDUCT.md). Report unacceptable behaviour to the maintainers as described there.

## License

ModelForge is released under the [BSD 3-Clause License](https://github.com/ForgeOpus/ModelForge/blob/main/LICENSE). By opening a pull request you agree that your contribution is licensed under the same terms.

## Workflow

The upstream repository is [ForgeOpus/ModelForge](https://github.com/ForgeOpus/ModelForge). All pull requests target its `main` branch.

1. Fork the repository on GitHub, then clone your fork and add upstream:

    ```bash
    git clone https://github.com/<your-user>/ModelForge.git
    cd ModelForge
    git remote add upstream https://github.com/ForgeOpus/ModelForge.git
    ```

2. Set up the environment as described in [Development setup](development.md). The short version:

    ```bash
    uv sync --extra dev
    ```

3. Create a branch from an up-to-date `main`:

    ```bash
    git fetch upstream
    git switch -c fix/short-description upstream/main
    ```

4. Make the change, with tests and documentation (see below).

5. Run the same checks CI runs:

    ```bash
    uv run pytest
    cd web/frontend && npm ci && npm run build && npm test
    ```

6. Push the branch to your fork and open a pull request against `ForgeOpus/ModelForge:main`. Describe what changed, why, and how you tested it. Link the issue it fixes, if there is one.

7. CI runs three jobs on every pull request: Python tests, the frontend type-check, build and tests, and the wheel smoke test. All three must pass before review. See [Continuous integration](development.md#continuous-integration).

## What a pull request must include

### Tests

Every behaviour change needs a test. Bug fixes need a regression test that fails without the fix.

- Python tests live in `tests/unit_tests/` and run with `pytest`. Test files are named `<subject>_test.py`.
- Frontend tests live in `web/frontend/tests/` and run with Vitest (`npm test`).
- Tests must run on CPU and offline. CI hides all CUDA devices and sets `HF_HUB_OFFLINE=1`, so a test that downloads from the Hugging Face Hub or needs a GPU will fail. Mock `HfApi` or `RecommendationEngine._query_hub`, and build tiny local models or datasets instead.
- Changes to training flows should also pass `python scripts/run_training_flow_tests.py`, which trains tiny local models through every use case and objective. It is not part of CI. See [Training-flow harness](development.md#training-flow-harness).

### Compatibility

- Don't break the REST API, the WebSocket event shapes, the `modelforge` and `modelforge-nb` commands, or the SQLite schema without saying so in the pull request. The browser UI, the terminal wizard and the notebook all depend on them.
- New request fields should be optional with a default.
- Database changes must migrate existing databases in place, as `WebRunLedger._initialize()` does with `ALTER TABLE`.

### Style

There is no enforced formatter or linter. Follow the style of the file you are editing.

- Python: type hints on public functions, Pydantic models for request and response shapes, short docstrings where the reason for the code is not obvious.
- TypeScript: the build runs `tsc` in strict mode, so type errors fail CI. The app UI is the single file `web/frontend/src/main.tsx`; keep it that way rather than splitting it into a new component tree.
- No secrets, tokens or absolute local paths in code, tests or logs.

### Commit messages

Write a short imperative subject line ("Fix GRPO reward on whitespace answers"), a blank line, then the reason for the change if it isn't obvious. Keep unrelated changes in separate commits.

## Documentation

ModelForge has two sets of user docs:

- **Read the Docs** (this site): Markdown in `docs/`, built with MkDocs Material from `mkdocs.yml`. Published at <https://modelforge-finetuning.readthedocs.io>.
- **In-app docs**: React pages in `web/frontend/src/docs/content/*.tsx`, shown in the browser console at `/app/docs/<slug>`. Each page has a `slug`, and each links to its longer Read the Docs counterpart.

!!! warning "Any user-facing change updates both"
    If a change alters something a user sees or relies on (a flag, a default, an endpoint, a column name, a number in a recommendation, an error message they need to act on), the pull request must update **both** the Read the Docs page in `docs/` **and** the matching in-app page in `web/frontend/src/docs/content/*.tsx`. A pull request that updates only one of them is incomplete.

The in-app pages are short. The Read the Docs pages may go deeper, but must not contradict them. Facts in both must match the code.

### Keeping the two doc sets in sync

Each in-app page maps to one Read the Docs page. The mapping lives in `web/frontend/src/docs/rtd.ts` (`RTD_PAGES`), which the app uses for the "Full guide on Read the Docs" link on each page. The Vitest test `web/frontend/tests/docs-rtd.test.ts` fails if an in-app page has no mapping or if a mapping points at a slug that no longer exists.

When you add or rename an in-app page, update `rtd.ts` and the table below in the same pull request.

| In-app slug | In-app file | Read the Docs page |
|---|---|---|
| `introduction` | `getStarted.tsx` | [Home](../README.md) |
| `install-and-first-run` | `getStarted.tsx` | [Quick start](../getting-started/quickstart.md) |
| `tour-of-the-console` | `getStarted.tsx` | [Browser console](../user-guide/browser-console.md) |
| `using-the-playground` | `getStarted.tsx` | [Playground](../user-guide/playground.md) |
| `what-fine-tuning-is` | `basics.tsx` | [Fine-tuning concepts](../getting-started/concepts.md) |
| `objectives` | `basics.tsx` | [Strategies overview](../strategies/overview.md) |
| `use-cases-and-tasks` | `basics.tsx` | [Use cases and tasks](../configuration/training-tasks.md) |
| `preparing-a-dataset` | `basics.tsx` | [Dataset formats](../configuration/dataset-formats.md) |
| `where-vram-goes` | `intermediate.tsx` | [Hardware and memory](../configuration/hardware-profiles.md) |
| `full-fine-tuning` | `intermediate.tsx` | [Full fine-tuning](../strategies/full-finetuning.md) |
| `lora-and-qlora` | `intermediate.tsx` | [LoRA and QLoRA](../strategies/qlora.md) |
| `how-a-tier-is-chosen` | `intermediate.tsx` | [How planning works](../configuration/configuration-guide.md) |
| `picking-a-base-model` | `intermediate.tsx` | [How planning works](../configuration/configuration-guide.md) |
| `data-parallel-vs-fsdp` | `advanced.tsx` | [Multi-GPU](../strategies/multi-gpu.md) |
| `gradient-checkpointing-and-packing` | `advanced.tsx` | [Memory optimizations](../strategies/memory-optimizations.md) |
| `liger-kernels` | `advanced.tsx` | [Memory optimizations](../strategies/memory-optimizations.md) |
| `resuming-from-checkpoints` | `advanced.tsx` | [Checkpoints](../user-guide/checkpoints.md) |
| `fixing-out-of-memory-errors` | `advanced.tsx` | [Performance and out-of-memory](../troubleshooting/performance.md) |
| `recommendation-fields` | `reference.tsx` | [Responses](../api-reference/responses.md) |
| `training-config-keys` | `reference.tsx` | [Training config](../api-reference/training-config.md) |
| `cli-commands` | `reference.tsx` | [CLI reference](../api-reference/cli.md) |
| `http-api` | `reference.tsx` | [REST API](../api-reference/rest-api.md) |

Read the Docs pages that are not in this table (installation, providers, custom strategies, troubleshooting, contributing and others) have no in-app counterpart. Changes to those touch `docs/` only.

### Building the docs

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Open <http://127.0.0.1:8000>. Read the Docs builds with `mkdocs build --strict`, so a broken relative link fails the build. Run it before pushing documentation changes. Note that `mkdocs serve` and `modelforge run` both default to port 8000; pass `-a 127.0.0.1:8001` to one of them if you run both.

### Documentation style

- One H1 per page, followed by a one to three sentence lead.
- Short sentences, concrete numbers, no marketing language, no emojis.
- Check every flag, default, endpoint and column name against the code before writing it down.
- Link between pages with relative `.md` paths.

## Reporting issues

Use the templates under **New issue** on GitHub.

- **Bug report**: include the ModelForge version (`pip show modelforge-finetuning`), Python version, OS, GPU and driver, which interface you used (browser, `modelforge cli` or `modelforge-nb`), the steps to reproduce, and the full error or the run's event log.
- **Feature request**: describe the problem first, then the change you'd like. Say which interface it affects.

Don't post Hugging Face tokens or the ModelForge startup token in an issue.

## See also

- [Development setup](development.md)
- [Architecture](architecture.md)
- [Adding a use case](use-cases.md)
- [Custom strategies](../strategies/custom-strategies.md)
- [Custom providers](../providers/custom-providers.md)
