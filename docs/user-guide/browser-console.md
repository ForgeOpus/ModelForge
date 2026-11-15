# Browser Console

The browser console is the main way to use ModelForge. This page walks through every page in it: planning a run on **New run**, watching it on the run detail page, finding finished models under **Artifacts**, and connecting a browser under **Settings**.

## Starting the console

```bash
modelforge run
```

The server prints the address it is listening on and a link that carries a startup token:

```text
$ modelforge run
ModelForge v4 browser UI
Serving on: http://0.0.0.0:8000
Open:       http://127.0.0.1:8000/app?token=…
The startup token is required for API and WebSocket access.
```

Your browser opens the **Open** link automatically unless you pass `--no-browser`. The console saves the token in the browser's local storage, removes it from the address bar, and sends it with every API request and WebSocket connection. A new random token is generated each time the server starts, unless you pass `--token`. All flags are listed in the [CLI reference](../api-reference/cli.md#modelforge-run).

!!! warning "The server listens on every interface by default"
    `--host` defaults to `0.0.0.0`. Anyone on your network who has the token can start runs and read files the server can read. Use `--host 127.0.0.1` on shared networks, and reach a remote machine through an SSH tunnel:

    ```bash
    ssh -L 8000:127.0.0.1:8000 user@gpu-box
    ```

The console itself is served at `/app`. The root path `/` shows a landing page.

## Layout

The sidebar has six pages:

| Page | Path | Purpose |
|---|---|---|
| New run | `/app` | Plan and start a fine-tune. |
| Runs | `/app/history` | Every run and its status. The badge shows how many runs are listed. |
| Playground | `/app/playground` | Chat with a finished model or a checkpoint. See [Playground](playground.md). |
| Artifacts | `/app/artifacts` | Output folders of completed runs. |
| Docs | `/app/docs` | The built-in documentation. |
| Settings | `/app/settings` | Theme, API token, detected hardware and environment. |

Below the navigation the sidebar shows:

- **Recent runs**: the two most recently updated runs, with status and date.
- A hardware card: the first GPU's name and VRAM (or **CPU** when no GPU is found), total RAM and free disk.
- A connection indicator: **Connected** plus the host, or **Connection unavailable** when API calls fail. The run list refreshes every 10 seconds.

The top bar has a theme toggle and a **New run** button. On narrow screens the sidebar collapses behind a menu button.

## New run

The **New run** page has a four-step form on the left and the **Plan** panel on the right.

### 1. Task

- **Use case**: one tile per use case the server can train, showing its name and task type. The use cases are Instruction Following, Text Generation, Summarization and Extractive QA. See [Use cases and tasks](../configuration/training-tasks.md).
- **Objective**: **Supervised**, **Preference** or **Reinforcement**. Objectives the selected use case cannot train are disabled. Summarization and Extractive QA are supervised only.

Changing the use case or objective clears any recommendation and dataset validation result, because both depend on the pair.

### 2. Dataset

- **Dataset path or Hub ID**: a local file or folder path (resolved on the machine running the server, not your browser), or a Hugging Face dataset ID such as `org/dataset`.
- **Local** / **Hub**: tells the server which kind of value you entered.
- **Validate**: loads the dataset, renames known column aliases, and checks the columns against the selected use case and objective. A valid dataset shows `Valid · columns: …`. An invalid one shows the error.

Validation is optional, but it runs the same checks that training runs, so it catches format problems before any GPU time is spent. Column requirements are in [Dataset formats](../configuration/dataset-formats.md).

### 3. Base model

Press **Generate recommendation**. The server reads the hardware, chooses a fine-tuning tier, and searches the Hub for up to five models that fit. They are listed with their parameter count and download count; the first is selected.

To use another model, pick a different row or type any Hub model ID into **Or enter a different model ID**. Whenever the selected model changes, the console asks the server to re-plan for that exact model (after a 250 ms pause in typing). While this runs the Plan shows **Updating plan…**, and **Start training** stays disabled until the plan matches the selected model. A typed ID must exist on the Hub, report a parameter count, and have a pipeline tag compatible with the task; otherwise the error is shown above the form. See [How planning works](../configuration/configuration-guide.md).

### 4. Output

- **Output directory**: where weights, checkpoints and logs are written, on the server's filesystem. Defaults to `./modelforge_output`, relative to the directory where `modelforge run` was started.
- **Write TensorBoard logs**: on by default. Logs go to `<output dir>/tensorboard/run-<id>`.
- **Start training**: enabled once there is a recommendation, a model and a dataset path. On success you are taken to the new run's detail page.

!!! warning "Give each run its own output directory"
    Every run defaults to `./modelforge_output`. Two runs writing to the same folder overwrite each other's weights and checkpoints.

The console always trains on every GPU the server detects. Choosing a subset of GPUs is possible in the [terminal wizard](terminal-cli.md) and through the [REST API](../api-reference/rest-api.md) (`gpu_indices`), not in the console.

Only one run trains at a time. If a run is already active, starting another fails with *A training run is already active. Cancel or wait for it before starting another.* Starting is also refused while the Playground is using model resources (see [Playground and training](playground.md#playground-and-training)).

### The Plan panel

The Plan fills in once a recommendation exists and matches the selected model. It shows:

| Item | Content |
|---|---|
| Heading | The tier (for example `qlora 4bit`) and the model's short name. |
| Subtitle | The strategy slug (`sft`, `dpo`, `grpo` or `qa`) and the parallelism strategy. |
| **VRAM estimate** | How total VRAM was turned into a model size ceiling. |
| **PEFT decision** | Why that tier was chosen. |
| **Training config** | Every `strategy_config` key and value the trainer will receive. |
| Warnings | One notice per warning, such as low disk space. |

The training config is read-only in the console. What each key means is in the [Training configuration](../api-reference/training-config.md) reference; how the tier is chosen is in [How planning works](../configuration/configuration-guide.md) and [Hardware and memory](../configuration/hardware-profiles.md).

## Runs

**Runs** lists up to 100 runs, most recently updated first, with the run number and model, the status, the use case, and the creation time. The list refreshes every 10 seconds.

| Status | Meaning |
|---|---|
| `queued` | Created, not yet started. |
| `running` | Training. |
| `completed` | Finished normally. The output is registered as an artifact. |
| `cancelled` | Stopped with **Cancel run**. |
| `failed` | Raised an error, or the server stopped while the run was active. |

Runs started from a [notebook](notebooks.md) are stored in the same database and appear here too. Runs from the [terminal wizard](terminal-cli.md) are stored separately and do not.

## Run detail

Open a run from **Runs**, from **Recent runs** in the sidebar, or by starting one. The page polls the run every 5 seconds and its checkpoints every 10 seconds.

- **Header**: run number, status, model ID and use case. While the run is `queued` or `running`, a **Cancel run** button asks for confirmation, then stops training after the current step.
- **Progress**: percentage, `Step N of M`, an estimate of the minutes remaining while active (from the observed step rate), and the latest loss. Before the trainer reports a step count it shows *Waiting for training metrics*.
- **Training log**: the raw events for the run, one JSON object per line. The page first loads every event already emitted (so reloading mid-run loses nothing), then streams new ones over a WebSocket while the run is active. Finished runs show the timeline saved when they ended. Event types are listed in [Responses and events](../api-reference/responses.md).
- **Checkpoints**: each saved checkpoint path and step, newest first. On runs that are not active and not completed, each has a **Resume** button. See [Checkpoints and resuming](checkpoints.md).
- **Run details**: objective, dataset, output directory and creation time.
- An error notice with the stored error message, if the run failed.

The console has no TensorBoard view. When a run with TensorBoard logging starts, the trainer launches a TensorBoard server inside the ModelForge process at `http://127.0.0.1:6006`, pointed at `<output dir>/tensorboard`, and tries to open it in a browser on the machine running training. Run records from the API carry this address as `tensorboard_url`. On a remote machine, forward the port:

```bash
ssh -L 6006:127.0.0.1:6006 user@gpu-box
```

To view logs later, after the server has stopped, start TensorBoard yourself on the server machine. The `modelforge tensorboard` helper needs the `[cli]` extra; plain `tensorboard --logdir …` works too:

```bash
modelforge tensorboard ./modelforge_output/tensorboard
```

Each run writes to its own `run-<id>` subfolder, so pointing TensorBoard at the `tensorboard` folder shows a run and its resumed continuations side by side.

## Artifacts

**Artifacts** shows one card per completed run, with the model's short name, the run number, the use case, the output directory and a **View run** link.

A completed output directory contains:

- The trained weights, or only the LoRA adapter for LoRA and QLoRA runs.
- The `checkpoint-<step>` folders saved during training.
- The `tensorboard/` logs, if enabled.
- `modelforge_config.json`, which records the base model, task, use case, objective, strategy, PEFT approach, the Transformers model class, the pipeline task and the ModelForge version. The Playground uses it to reload the model correctly.

When a run completes, ModelForge also adds the output folder to a model registry, named `<model short name>-<run id>`, with its size on disk.

### Pushing to the Hugging Face Hub

The console has no push button. Pushing is available through the REST API, using the registry ID from `GET /api/models` (not the run ID):

```bash
TOKEN=...   # the startup token
curl -s -H "X-ModelForge-Token: $TOKEN" http://127.0.0.1:8000/api/models
curl -s -X POST -H "X-ModelForge-Token: $TOKEN" -H "Content-Type: application/json" \
  -d '{"repo_id": "yourname/my-model", "private": true}' \
  http://127.0.0.1:8000/api/models/1/push
```

The server creates the repository if it does not exist (`private` defaults to `false`), uploads the whole output folder with the server's cached Hugging Face login, and marks the model as pushed. It returns the repository URL. The login must have write access; run `hf auth login` on the server machine first (or set `HF_TOKEN` in the server's environment).

!!! warning "Pushing uploads the whole folder"
    That includes every checkpoint and the TensorBoard logs. Delete what you do not want to publish before pushing.

`DELETE /api/models/{id}` removes a registry entry; add `?delete_files=true` to also delete the output folder. Both endpoints are described in the [REST API reference](../api-reference/rest-api.md).

## Settings

Settings has five sections:

| Section | Content |
|---|---|
| **Appearance** | Theme: **Light**, **Dark** or **System** (follows the OS). Stored in this browser only. |
| **Connection** | **API token**: paste the token printed by `modelforge run`, then **Save token** and reload. **Clear** removes it. |
| **Hardware** | GPUs with VRAM (or *No GPU detected*), total memory and free disk, as detected on the server. |
| **Environment** | ModelForge version and the registered providers. |
| **About** | Version blurb. |

If the hardware shown here is wrong, the plan will be wrong too. See [Hardware and memory](../configuration/hardware-profiles.md).

### Connecting another browser

Open the printed link in that browser, or paste the token under **Settings → Connection → API token** and reload. Error notices that mention the token include an **Open settings** link. Because the token changes on every restart unless you pass `--token`, a browser that was connected before a restart shows **Connection unavailable** until it gets the new token.

## See also

- [Quick start](../getting-started/quickstart.md)
- [Playground](playground.md)
- [Checkpoints and resuming](checkpoints.md)
- [REST API](../api-reference/rest-api.md)
- [Common issues](../troubleshooting/common-issues.md)
