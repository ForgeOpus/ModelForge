# Common issues

This page covers the failures people hit most often with ModelForge v4, grouped by where they show up: connecting to the server, installing, planning, validating data, and running. Error messages are quoted exactly as ModelForge prints them, so you can search this page for the text you see.

Out-of-memory errors have their own page: [Performance and out-of-memory](performance.md). Native Windows problems are in [Windows issues](windows-issues.md).

## Connecting to the console

### "Connection unavailable" in the sidebar, or 401 errors

The sidebar shows **Connection unavailable** when the console's API calls fail. Almost always the token is missing or out of date. The API answers with:

```json
{"detail": "Invalid or missing ModelForge startup token."}
```

A new token is generated **every time the server starts**, so a bookmarked link or a token saved in another browser stops working after a restart.

Fix it one of these ways:

- Open the **Open:** link printed by the most recent `modelforge run`. The console stores the token and removes it from the address bar.
- Paste the token under **Settings → Connection → API token**, press **Save token**, and reload.
- Start the server with a fixed token so links survive restarts:

    ```bash
    modelforge run --token my-long-random-token
    ```

To check that the server is up at all, `GET /api/health` needs no token:

```bash
curl http://127.0.0.1:8000/api/health
```

If that fails too, the server isn't running, or isn't reachable on that host and port.

### Live logs or the Playground don't connect

WebSockets only take the token from the query string (`?token=...`). With a missing or wrong token the server closes the socket with code `1008` before accepting it. The console handles this for you once the token is saved; in your own clients, add `?token=` to the WebSocket URL. A `Bearer` header is not enough for WebSockets.

If you reach the server through a reverse proxy, the proxy must forward WebSocket upgrades for `/api/runs/{id}/events` and `/api/playground/{id}/chat`.

### The console can't be reached from another machine

`modelforge run` binds `0.0.0.0` by default, so it listens on every interface, but a firewall may still block port 8000. The printed **Open:** link always says `127.0.0.1`, which only works on the server itself. From another machine, replace it with the server's address, or better, use an SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 user@gpu-box
# then open http://127.0.0.1:8000/app?token=<token> locally
```

Start the server with `--no-browser` on remote machines so it doesn't try to open a browser there.

## Starting the server

### Port already in use

If something else holds the port, uvicorn exits right after the banner with an error like:

```text
ERROR:    [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
```

(`[Errno 48]` on macOS, `[WinError 10048]` on Windows.) Either stop the other process, often an earlier `modelforge run` still running in another terminal, or pick another port:

```bash
modelforge run --port 8080
```

### The page shows "Internal Server Error" (source checkouts)

Opening `/` or `/app` returns `500 Internal Server Error`, and the server log ends with:

```text
RuntimeError: File at path .../web/frontend_dist/index.html does not exist.
```

The browser UI is a built bundle in `web/frontend_dist/`. It ships inside the PyPI wheel, but it is **not committed** to the repository. A source checkout has to build it once, which needs Node.js and npm:

```bash
python scripts/build_frontend.py
# or
cd web/frontend && npm ci && npm run build
```

An editable install (`pip install -e .` or `uv sync`) runs the same build automatically and fails with this message if npm isn't on `PATH`:

```text
Node.js and npm are required to package the ModelForge browser UI. Install Node.js (npm is included), ensure npm is on PATH, and retry the package build. Refusing to build a wheel without web/frontend_dist/index.html.
```

The API keeps working without the bundle; only the console pages fail. Installing the released wheel with `pip install modelforge-finetuning` does not need Node. See [Development setup](../contributing/development.md).

### Wrong Python version

ModelForge v4 requires **Python 3.13 or newer** (`requires-python = ">=3.13"`). On older interpreters pip skips v4 and reports something like:

```text
ERROR: Ignored the following versions that require a different python version: 4.0.0 Requires-Python >=3.13
ERROR: No matching distribution found for modelforge-finetuning
```

or silently installs an old 3.x release that has a different command set. Check with `python --version` and create a 3.13 environment:

```bash
uv venv --python 3.13
# or
python3.13 -m venv .venv
```

### `modelforge cli` says optional dependencies are missing

```text
The terminal UI requires optional dependencies. Install with: pip install "modelforge-finetuning[cli]"
```

The terminal wizard and `modelforge tensorboard` need the `[cli]` extra (click and rich). The browser console doesn't.

## Hugging Face login and gated models

Public models and datasets need no login. Gated models (Llama, Gemma and others) and private repositories need two things:

1. Accept the licence on the model's Hub page while signed in.
2. Log in once on the machine running the server:

    ```bash
    hf auth login
    ```

    Or set `HF_TOKEN` in the environment that starts `modelforge run`.

!!! note "`huggingface-cli login` no longer works"
    ModelForge v4 depends on `huggingface-hub` 1.14 or newer, where the old `huggingface-cli` command only prints `Warning: huggingface-cli is deprecated and no longer works. Use hf instead.` Use `hf auth login` instead.

The server reads the cached login whenever it talks to the Hub, so log in as the same user that runs `modelforge run`. Restarting the server is not required.

Without access, model validation reports:

```text
No model with that ID is visible on the Hub. Check the spelling, or run `hf auth login` if it is private or gated.
```

The Hub answers "unauthorized" rather than "not found" for unknown repos when you aren't signed in, so this message covers both a typo and a missing login.

## Planning and model refusals

### Recommendation errors

`POST /api/recommendations` (the console's **Generate recommendation**) returns `422` with these messages:

| Message | Cause and fix |
|---|---|
| `Unknown use case slug '<slug>'. Registered: [...]` | Use a slug from `modelforge list-tasks` or `GET /api/use-cases`. |
| `Use case '<slug>' does not support objective '<objective>'. Compatible objectives: [...]` | Summarization and Extractive QA are supervised only. |
| `GPU indices must be between 0 and N.` | A `gpu_indices` entry is out of range. On a machine with no detected GPU any index fails. |
| `Strategy '<slug>' requires more VRAM than is available on this hardware for any viable model size. Adjusted VRAM: X GB.` | The strategy's overhead exceeds your VRAM. DPO and GRPO need the most. |
| `Selected model exceeds the available training memory budget.` | The model you named is larger than even the 4-bit ceiling. Pick a smaller model or add GPUs. |
| `Cannot configure this model without parameter metadata.` | See below. |

A Hub outage or no network gives a `500` instead, because the model search itself failed.

### "No models found on the Hub"

The plan comes back with an empty model list, `strategy_config: null`, and the warning `No models found on the Hub matching task '...' with at most XB parameters. No strategy config was generated.` Nothing popular under your ceiling has the use case's pipeline tag and publishes a parameter count. On CPU-only machines the ceiling is at most 500M parameters, which leaves few candidates. Enter a model ID by hand, or plan on a machine with more VRAM.

### Hand-entered model IDs that are refused

A model ID typed into **Or enter a different model ID** (or sent as `model_id`) is checked against the Hub before training. The refusals:

| Message | Cause |
|---|---|
| `Enter a model ID, for example org/model.` | Empty field. |
| ``No model with that ID is visible on the Hub. Check the spelling, or run `hf auth login` if it is private or gated.`` | Typo, private repo or gated model. See [login](#hugging-face-login-and-gated-models). |
| `This model is tagged '<tag>', which is incompatible with task '<task>'. Expected one of: <tags>.` | For example a `text-generation` model for Summarization, which needs `summarization`, `translation` or `text2text-generation`. Extractive QA needs `question-answering`. |
| `The Hub returned an error (<status>).` | Any other Hub HTTP error. Retry later. |
| `Could not reach the Hugging Face Hub. Check your connection.` | No network, proxy problems. |
| `Cannot configure this model without parameter metadata.` / `Cannot safely configure this model without valid parameter metadata.` | The repo has no safetensors parameter count (common for `.bin`-only or very old repos), so ModelForge can't size it. Choose a model that publishes safetensors weights. |
| `Selected model exceeds the available training memory budget.` | Too large for this hardware at any tier. |

### `POST /api/runs` returns 400 with a recommendation you sent

| Message | Fix |
|---|---|
| `Recommendation does not match the selected use case and objective.` | `task_slug` / `strategy_slug` in the recommendation must come from the same use case and objective as the request. Re-fetch the plan. |
| `Recommendation provider is not supported by this strategy.` | `provider_slug` must be `huggingface`. |
| `Unknown fine-tuning approach.` | `peft_approach` must be `full_finetune`, `lora_bf16`, `qlora_8bit` or `qlora_4bit`. |
| `FSDP parallelism is not compatible with PEFT approach '...'...` | FSDP only works with full fine-tuning. |
| A pydantic validation message | The recommendation object is incomplete. Send the whole object returned by `/api/recommendations`. |

Edits to a recommendation are discarded unless `model_id` is its first candidate. See [Training config: overriding the plan](../api-reference/training-config.md#overriding-the-plan).

## Dataset validation errors

**Validate** in the console (`POST /api/datasets/validate`) loads the data the same way training does. Its `error` field carries one of these:

| Error | Cause and fix |
|---|---|
| `Dataset path not found: <path>` | Local paths are resolved **on the server**, relative to the directory `modelforge run` was started in. Use an absolute path, or switch to **Hub** for a Hub dataset ID. |
| `Dataset is missing required column(s): <names>.` | Required columns are absent. The message lists the use case's **alias** names; the trainer's names work too. For example `instruction, response` is satisfied by `prompt` and `completion`. See [Dataset formats](../configuration/dataset-formats.md). |
| `Dataset contains both an alias and its canonical training column: 'document' -> 'prompt'. Remove or rename one explicitly.` | You have both names for the same column, so ModelForge won't guess which to train on. Drop one. |
| `Dataset must contain the following columns: {'chosen', 'rejected'}. Found columns: {...}` | DPO pairs are missing. |
| `GRPO datasets must contain 'prompt' and 'answer' columns. 'answer' is used by ModelForge's built-in exact-match reward. Found columns: [...]` | Reinforcement data needs an exact expected answer per prompt. |
| `Extractive QA datasets require context, question, and answers columns. Missing: ...` | QA data needs SQuAD-style rows. |
| `No supported dataset files found in directory: <path>` | A directory must contain `.parquet`, `.arrow`, `.jsonl`, `.json`, `.csv` or `.txt` files. Only files with the first extension found in that priority order are loaded. |
| `Invalid JSON structure for dataset loading: ...` | The JSON file is neither JSON Lines, a top-level array, nor an object with a list field. Convert it to JSON Lines. |
| `This objective is not executable.` | The objective isn't available for that use case (for example reinforcement for Summarization). |
| A Hugging Face `datasets` error | For Hub datasets: wrong ID, a gated dataset without login, or a dataset that needs a config name. ModelForge loads Hub datasets with their default config. |

## Running training

### "A training run is already active" (409)

```text
A training run is already active. Cancel or wait for it before starting another.
```

Each server trains one run at a time. Wait, or cancel the active run first. Queuing is not supported; script it by polling `GET /api/runs/{id}` until the run finishes.

### "Playground inference is using model resources" (409)

```text
Playground inference is using model resources. Wait for it to finish before starting training.
```

A Playground request is in progress, or a **streaming session is still open**. An open session keeps its model loaded and blocks training until it's closed. Press **Close session** in the Playground (or `DELETE /api/playground/sessions/{id}`), then start the run.

### Runs marked failed with "ModelForge stopped before this run reached a terminal state."

When a server starts, it marks every run still recorded as `queued` or `running` as failed with this message, because their training thread died with the previous process. Checkpoints are untouched, so [resume](../user-guide/checkpoints.md) from the latest one.

### Cancel returns 404

```json
{"detail": "No active job found for run 7."}
```

Only runs that are actively training in this server process can be cancelled. Finished runs, and runs from a previous server process, can't.

### A run fails when it starts with "unexpected keyword argument"

```text
SFTConfig.__init__() got an unexpected keyword argument 'my_key'
```

An edited recommendation added a key the trainer config doesn't accept. The key names in `strategy_config` must be valid TRL / Transformers training arguments. See [Training config](../api-reference/training-config.md#how-keys-reach-the-trainer).

### Two runs overwrite each other

Every run defaults to `./modelforge_output`. Runs that share an output directory overwrite each other's weights and checkpoints, and the model registry keeps one entry per folder. Give each run its own output directory.

### Out of memory during training

`CUDA out of memory` (or `torch.OutOfMemoryError`) ends the run with `status: failed`. ModelForge doesn't retry. See [Performance and out-of-memory](performance.md).

## GPU detection and installation

### A GPU is listed but training runs on the CPU, or precision is fp32

**Settings → Hardware** reads GPUs through NVIDIA's NVML, which only needs the driver. Training needs a **CUDA build of PyTorch**. If PyTorch can't see CUDA, the plan sets `bf16: false` and `fp16: false`, and the trainer runs on the CPU. Check in the same environment as the server:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `False`, install a CUDA build of PyTorch from the [PyTorch install selector](https://pytorch.org/get-started/locally/) for your driver, then restart the server and generate the plan again. On Windows, the default PyPI `torch` wheel is CPU-only; see [Windows issues](windows-issues.md).

### No GPU detected at all

The plan says `CPU-only (no GPU)`. Check `nvidia-smi` works; if not, install or repair the NVIDIA driver. AMD and Intel GPUs are not detected. Apple Silicon is detected as `Apple Silicon GPU` with VRAM `N/A` and planned as 8 GB.

### bitsandbytes errors

`bitsandbytes` is a required dependency (version 0.49.2 or newer), with wheels for Linux x86_64 and aarch64, Windows x86_64 and macOS arm64. It is only used by the QLoRA tiers (`qlora_8bit`, `qlora_4bit`), which need CUDA. On CPU and Apple MPS the planner never uses those tiers and strips quantization from the config.

If a QLoRA run fails inside bitsandbytes, the usual cause is a PyTorch build without CUDA or a driver too old for it. Fix PyTorch as above. As a workaround, force `lora_bf16` by sending an edited recommendation with `peft_approach: "lora_bf16"` and `model_init_kwargs: {}`, if the model fits.

### Liger kernels: install fails or `use_liger_kernel` stays false

```bash
pip install "modelforge-finetuning[gpu]"
```

`liger-kernel` depends on Triton, which only publishes Linux wheels. On Windows and macOS this extra fails to install. ModelForge doesn't need it; plans simply set `use_liger_kernel: false`.

On Linux, if the plan still shows `use_liger_kernel: false` with a CUDA GPU, `liger-kernel` isn't importable from the environment running the server. Install it there and generate the plan again. If a run fails inside a Liger kernel, uninstall `liger-kernel` and re-plan. GRPO, QA and Summarization never use Liger. See [Memory optimizations](../strategies/memory-optimizations.md).

## Models and the Hub

### Push fails

| Response | Fix |
|---|---|
| `502` `Upload failed: <Hub error>` | Not logged in on the server, or the token has no write access to that namespace. Run `hf auth login` with a write token. |
| `409` `The files for this model are no longer at <path>.` | The output folder was moved or deleted. |

Push uploads the entire output folder, checkpoints and TensorBoard logs included.

### Delete is refused

```text
Refusing to remove a filesystem or workspace root.
```

The model's recorded folder is `/`, your home directory, or the server's working directory. Remove the registry entry without files (`delete_files=false`) and clean up by hand.

## Playground

| Message | Fix |
|---|---|
| `A training run is using the GPU. Loading a model beside it would run the card out of memory, so the playground stays closed until the run finishes.` | Streaming sessions are closed while training. Wait, or use a one-shot reply, which runs on CPU and disk during training. |
| `Model resources are busy. Try again when the current request finishes.` | Another Playground request or session holds the model lock. |
| `Choose a completed run or a registered checkpoint.` | The run hasn't completed. Pick one of its checkpoints instead. |
| `This conversation needs N tokens, exceeding the M-token limit. Start a new chat or choose a larger supported context.` | Start a new chat or raise the context window. Earlier messages are never dropped silently. |
| `Send JSON with nonempty "question" and "context" strings.` | Extractive QA models expect `{"question": "...", "context": "..."}`. |
| `Not enough available RAM for safe inference.` / `Not enough memory to execute the model, even with disk offload.` / `Not enough disk space for safe model offload.` | See [Playground out-of-memory](performance.md#out-of-memory-in-the-playground). |

## Getting more detail

- The terminal running `modelforge run` shows full tracebacks for `500` errors and failed runs.
- `GET /api/runs/{id}` has the run's `error`; `GET /api/runs/{id}/events` has the full timeline.
- `GET /api/hardware` and `GET /api/system` show what the server detected.

When opening an issue at [github.com/ForgeOpus/ModelForge/issues](https://github.com/ForgeOpus/ModelForge/issues), include the ModelForge version, `python --version`, the output of the PyTorch check above, the plan's `rationale`, and the run's `error`.

## See also

- [Performance and out-of-memory](performance.md)
- [Windows issues](windows-issues.md)
- [FAQ](faq.md)
- [REST API](../api-reference/rest-api.md)
