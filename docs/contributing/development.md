# Development setup

This page covers working on ModelForge from a source checkout: the Python and Node toolchains, running the backend and the Vite dev server, the test suites, what CI runs, packaging, building these docs, and cutting a release.

## Prerequisites

| Tool | Version | Where it is pinned |
|---|---|---|
| Python | 3.13 | `.python-version`, `requires-python = ">=3.13"` in `pyproject.toml` |
| [uv](https://docs.astral.sh/uv/) | recent | `uv.lock` |
| Node.js and npm | 22.23.5 | `.node-version` |
| Git | any | |

Node is needed even if you only work on Python. Installing from a source checkout builds the browser UI, and the build stops with an error if `npm` is not on `PATH`. Vite needs Node `^20.19.0` or `>=22.12.0`; use the pinned version.

A CUDA GPU is optional. Everything in CI runs on CPU.

## Set up the checkout

```bash
git clone https://github.com/<your-user>/ModelForge.git
cd ModelForge
uv sync --extra dev
```

`uv sync` creates `.venv/`, installs the locked dependencies plus the `dev` extra (`click`, `rich`, `httpx`, `pytest`), and installs the project in editable mode. The editable install runs `scripts/build_frontend.py`, so it also produces `web/frontend_dist/`.

Use `uv sync --frozen --extra dev` to fail instead of updating `uv.lock`; that is what CI does.

Without uv:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Optional extras

| Extra | Installs | Use |
|---|---|---|
| `dev` | `click`, `rich`, `httpx`, `pytest` | Tests and the terminal wizard |
| `cli` | `click`, `rich` | The `modelforge cli` terminal wizard only |
| `gpu` | `liger-kernel` | Liger fused kernels. CUDA only, no CPU or Windows wheels. Strategies fall back when it is absent. |
| `notebook` | nothing | Placeholder; the notebook UI needs no extra packages |

```bash
uv sync --extra dev --extra gpu
```

## Run the app

```bash
uv run modelforge run --no-browser
```

The server binds `0.0.0.0:8000` by default and prints a URL of the form `http://127.0.0.1:8000/app?token=...`. The token is required for every `/api` request and WebSocket. Pass `--token <value>` to keep the same token across restarts while you develop. See the [CLI reference](../api-reference/cli.md) for all flags.

The other entry points:

```bash
uv run modelforge cli            # terminal wizard
uv run modelforge recommend --use-case instruction_following
uv run modelforge list-tasks
uv run modelforge-nb             # notebook prompts, also works in a plain shell
```

## Frontend development

The browser UI is a Vite + React 19 + TypeScript app in `web/frontend/`. The app itself is the single file `src/main.tsx`; the in-app docs are under `src/docs/`.

For UI work, run the API and the Vite dev server side by side:

```bash
# shell 1
uv run modelforge run --no-browser --token dev

# shell 2
cd web/frontend
npm ci
npm run dev
```

Vite serves on port 5173 (bound to `0.0.0.0`) and proxies `/api`, including WebSockets, to `http://127.0.0.1:8000`. Open `http://127.0.0.1:5173/app?token=dev`. Edits reload in the browser without rebuilding.

| Script | Command | Does |
|---|---|---|
| `npm run dev` | `vite --host 0.0.0.0` | Dev server with hot reload |
| `npm run build` | `tsc && vite build` | Strict type-check, then production bundle into `web/frontend_dist/` |
| `npm run preview` | `vite preview` | Serve the production bundle |
| `npm test` | `vitest run` | Frontend tests in `web/frontend/tests/` |

### Building `frontend_dist`

`modelforge run` serves the SPA from `web/frontend_dist/`. The directory is generated and not committed. `/`, `/app` and deep links such as `/app/history` all return `index.html`; hashed files under `assets/` are served next to it.

```bash
python scripts/build_frontend.py
# equivalent:
cd web/frontend && npm ci && npm run build
```

`scripts/build_frontend.py` runs `npm ci` and `npm run build` against `package-lock.json`, then checks that `index.html`, every asset it references, and the files from `web/frontend/public/` are present. It aborts if npm is missing or the lockfile is absent. If you change Python code only, you don't need to rebuild; rebuild after pulling frontend changes.

## Tests

### Python

```bash
uv run pytest
uv run pytest tests/unit_tests/recommendation_engine_test.py
uv run pytest -k grpo
```

`pyproject.toml` sets `testpaths = ["tests"]` and puts the repository root on `sys.path`. All tests are in `tests/unit_tests/`, one file per subsystem, named `<subject>_test.py`.

Things to know when writing tests:

- **No network, no GPU.** CI sets `CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`. Set the same locally to catch accidental downloads. Patch `RecommendationEngine._query_hub` or `utilities.recommendation_engine.HfApi` instead of calling the Hub.
- **Registries are global.** Use cases, strategies, tasks and providers register at import time and reject duplicate slugs. A test that defines a throwaway subclass must remove it from `_registry` afterwards. `ProviderFactory.clear_registry()` exists for providers.
- **Singletons.** `DatabaseManager` and `FinetuneLedger` are singletons; reset `_instance` between tests. `WebRunLedger` takes a `db_path`, so pass a `tmp_path` file.
- **Fake hardware.** Build `HardwareDetector.HardwareSpecs(...)` directly rather than detecting the test machine.
- `tests/unit_tests/frontend_packaging_test.py` builds a wheel from a clean copy of the tree, installs it in a fresh virtualenv and requests `/`, `/app` and `/app/history`. It needs Node, npm and network access for `npm ci`.

### Frontend

```bash
cd web/frontend
npm ci
npm test
```

Vitest runs in jsdom. The current suites are `routes.test.tsx` (routes and auth), `streaming.test.tsx`, `progress.test.ts` (progress and ETA from Unix-second timestamps) and `docs-rtd.test.ts` (every in-app docs page maps to a Read the Docs page). There is no Playwright or other browser end-to-end suite.

### Training-flow harness

```bash
python scripts/run_training_flow_tests.py --list
python scripts/run_training_flow_tests.py
python scripts/run_training_flow_tests.py --flow instruction-following-supervised
```

This is an integration harness, not part of pytest or CI. It builds tiny local models and datasets, calls `workflow.training.assemble_and_train` for every use case and objective, and checks that each run produces a model. It needs no Hub access or GPU but does real training steps, so it takes a few minutes. Results go to `test-results/training-flows.json` (`--output` to change). `--keep-artifacts` keeps the models under `test-results/training-flow-artifacts/`. Run it when you touch strategies, data normalization or `workflow/training.py`.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and every push to `main`. The three jobs are independent, install from the lockfiles, stay CPU-only and offline, and pin third-party actions to commit SHAs. `tests/unit_tests/ci_workflow_test.py` asserts those properties, so edit the workflow and that test together.

| Job | Name | Runs |
|---|---|---|
| `python` | Python tests | `uv sync --frozen --extra dev`, then `uv run pytest`. Node is installed for the packaging test. |
| `frontend` | Frontend type-check and build | `npm ci`, `npm run build`, `npm test`, then asserts `web/frontend_dist/index.html` and a non-empty JS bundle exist |
| `distribution` | Wheel smoke test | `scripts/build_distribution.sh`, installs the wheel into a clean environment, runs `scripts/distribution_smoke.py` |

To reproduce the Python and frontend jobs locally:

```bash
uv sync --frozen --extra dev
uv run pytest
cd web/frontend && npm ci && npm run build && npm test
```

## Packaging

`pip wheel .`, `python -m build` and `uv build` all run the locked frontend build through `setup.py` before collecting package data, so the wheel contains `web/frontend_dist/`. If Node or npm is missing, or the frontend build fails, no wheel is produced. Installing a wheel that is already built does not need Node; installing from a source checkout or sdist does.

`scripts/build_distribution.sh` is the release path and the CI wrapper. It checks for Node, npm and uv, checks the Node version, builds the frontend, runs `uv build` into `dist/`, then runs `scripts/distribution_smoke.py --archives dist` to check that both the sdist and the wheel contain the bundle.

### Smoke-testing the installed wheel

This is the `distribution` CI job, run from the repository root:

```bash
bash scripts/build_distribution.sh
req="$(mktemp)"
uv export --frozen --no-dev --no-emit-project --output-file "$req"
uv venv --python "$(tr -d '[:space:]' < .python-version)" /tmp/modelforge-wheel
uv pip install --python /tmp/modelforge-wheel/bin/python --requirement "$req"
uv pip install --python /tmp/modelforge-wheel/bin/python --no-deps dist/*.whl
root="$PWD"
( cd /tmp && /tmp/modelforge-wheel/bin/python "$root/scripts/distribution_smoke.py" )
```

Start the last command outside the checkout so Python imports the installed package rather than the source tree. The script checks the `modelforge` and `modelforge-nb` entry points, confirms the installed package contains the frontend bundle, starts `modelforge run`, and requests `/`, `/app` and an authenticated `/api/use-cases`.

## Building the docs

These pages are MkDocs Material, configured by `mkdocs.yml` at the repository root.

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

`mkdocs serve` defaults to `127.0.0.1:8000`, the same port as `modelforge run`. Use `mkdocs serve -a 127.0.0.1:8001` if both are running. Before pushing, run `mkdocs build --strict`; Read the Docs fails on broken links and missing nav entries the same way.

Remember that user-facing changes also update the in-app docs. See [Keeping the two doc sets in sync](contributing.md#keeping-the-two-doc-sets-in-sync).

## Releasing

Releases are published by `.github/workflows/workflow.yaml` when a version tag is pushed.

1. Bump `version` in `pyproject.toml`. The string `4.0.0` is also hard-coded in `workflow/service.py` (`system_info`), `workflow/jobs.py` (`modelforge_config.json`), `web/app.py` (FastAPI `version`) and `web/frontend/package.json`; update those too.
2. Merge to `main`.
3. Tag the merge commit with the bare version and push the tag:

    ```bash
    git tag 4.0.1
    git push upstream 4.0.1
    ```

    Accepted tag forms are `X.Y.Z`, `X.Y.ZaN`, `X.Y.ZbN` and `X.Y.ZrcN`. There is no `v` prefix.

The workflow then:

| Job | Does |
|---|---|
| `details` | Reads the version from the tag |
| `check_pypi` | Fails unless the tag version is greater than the latest `modelforge-finetuning` release on PyPI |
| `build_package` | Fails unless the tag equals `version` in `pyproject.toml`, then sets up Python from `.python-version`, Node from `.node-version` and uv, runs `scripts/build_distribution.sh` (locked frontend build, sdist and wheel, archive check) and uploads `dist/` |
| `pypi_publish` | Publishes to PyPI with trusted publishing (OIDC, `id-token: write`) in the `release` environment. No API token is stored. |
| `github_release` | Creates a GitHub release named after the tag, with the built files and generated notes |

Bump the version before tagging. A tag that doesn't match `pyproject.toml` fails in `build_package` before anything is published.

## See also

- [Contributing](contributing.md)
- [Architecture](architecture.md)
- [Adding a use case](use-cases.md)
- [CLI reference](../api-reference/cli.md)
