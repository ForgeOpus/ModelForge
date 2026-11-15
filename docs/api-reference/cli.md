# CLI Commands

ModelForge installs two commands: `modelforge`, which starts the browser console and provides terminal utilities, and `modelforge-nb`, the notebook interface. This page lists every subcommand, argument and default.

```text
$ modelforge --help
usage: modelforge [-h] {run,cli,recommend,history,list-tasks,tensorboard} ...

positional arguments:
  {run,cli,recommend,history,list-tasks,tensorboard}
    run                 Start the browser UI.
    cli                 Start the optional terminal wizard.
    recommend           Print hardware-aware model recommendations.
    history             Print run history.
    list-tasks          List available use cases.
    tensorboard         Launch TensorBoard for a log directory.

options:
  -h, --help            show this help message and exit
```

Running `modelforge` with no subcommand prints this help. Every subcommand accepts `-h` / `--help`. There is no `--version` flag; the version is shown under **Settings → Environment** in the console and by `GET /api/health`.

## Summary

| Command | Needs | Purpose |
|---|---|---|
| [`modelforge run`](#modelforge-run) | base package | Start the server and browser console. |
| [`modelforge cli`](#modelforge-cli) | `[cli]` extra | Interactive terminal wizard. |
| [`modelforge recommend`](#modelforge-recommend) | base package | Print a recommendation without training. |
| [`modelforge history`](#modelforge-history) | base package | Print run history. |
| [`modelforge list-tasks`](#modelforge-list-tasks) | base package | List the use cases that can be trained. |
| [`modelforge tensorboard`](#modelforge-tensorboard) | `[cli]` extra | Start TensorBoard for a log directory. |
| [`modelforge-nb`](#modelforge-nb) | base package | Prompt-driven notebook interface. |

Commands that need `[cli]` exit with an install hint when it is missing:

```bash
pip install "modelforge-finetuning[cli]"
```

## modelforge run

Starts the FastAPI server and the browser console.

```bash
modelforge run [--host HOST] [--port PORT] [--no-browser] [--token TOKEN]
```

| Option | Default | Effect |
|---|---|---|
| `--host HOST` | `0.0.0.0` | Interface to bind. |
| `--port PORT` | `8000` | Port to bind. |
| `--no-browser` | off | Print the link without opening a browser. Use on remote machines. |
| `--token TOKEN` | random | Use a fixed startup token instead of generating one, so bookmarks keep working across restarts. |

Output:

```text
$ modelforge run --host 127.0.0.1 --no-browser
ModelForge v4 browser UI
Serving on: http://127.0.0.1:8000
Open:       http://127.0.0.1:8000/app?token=…
The startup token is required for API and WebSocket access.
```

When `--host` is `0.0.0.0` or `::`, the **Open** link uses `127.0.0.1`. The token is 32 random bytes, URL-safe encoded, unless `--token` is given. Every `/api` request except `GET /api/health` needs it, as `Authorization: Bearer <token>`, an `X-ModelForge-Token` header, or a `?token=` query parameter. See the [REST API](../api-reference/rest-api.md).

On startup, any run left `queued` or `running` by a previous process is marked `failed`. See [Checkpoints and resuming](../user-guide/checkpoints.md#if-the-server-stops).

!!! warning
    The default `--host 0.0.0.0` listens on every interface. Use `--host 127.0.0.1` on shared networks and connect through an SSH tunnel.

Guide: [Browser console](../user-guide/browser-console.md).

## modelforge cli

Starts the interactive terminal wizard: Hugging Face login check, hardware, GPU selection, use case, objective, recommendation, model, dataset, output directory, confirmation, then training in the foreground.

```bash
modelforge cli [--no-tensorboard] [--tensorboard-logdir DIR]
```

| Option | Default | Effect |
|---|---|---|
| `--no-tensorboard` | off | Do not write TensorBoard logs. |
| `--tensorboard-logdir DIR` | `<output dir>/tensorboard` | Directory for TensorBoard event logs. Each launch writes to its own `run-<hex>` subfolder of it. Ignored with `--no-tensorboard`. |

Requires the `[cli]` extra. Completed runs are recorded in the legacy ledger shown by `modelforge history --legacy`.

Guide: [Terminal wizard](../user-guide/terminal-cli.md).

## modelforge recommend

Detects hardware, plans a fine-tune and prints the result. Nothing is downloaded or trained.

```bash
modelforge recommend [--use-case SLUG] [--objective {supervised,preference,reinforcement}] [--n N]
```

| Option | Default | Effect |
|---|---|---|
| `--use-case SLUG` | prompt | Use case slug: `instruction_following`, `text_generation`, `summarization` or `extractive_qa`. If omitted, a numbered list is shown and you enter a number or a slug. |
| `--objective` | `supervised` | `supervised`, `preference` or `reinforcement`. Must be one the use case supports. |
| `--n N` | `5` | Maximum number of candidate models. Must be positive. |

The plan uses every detected GPU; there is no GPU selection option.

```text
$ modelforge recommend --use-case instruction_following --objective preference
Task:      causal_language_modeling
Strategy:  dpo
PEFT:      qlora_4bit
Models:
  - org/model-a (1,543,714,304 params, 1,234,567 downloads)
  - ...
Warnings:
  - ...
```

The model names, tier and warnings depend on your hardware and the Hub at the time. The `Warnings:` block appears only when there are warnings. An unknown slug or an unsupported objective ends with a Python error.

See [How planning works](../configuration/configuration-guide.md).

## modelforge history

Prints run history, most recent first, up to 100 runs.

```bash
modelforge history [--legacy]
```

| Option | Default | Effect |
|---|---|---|
| `--legacy` | off | Show the terminal wizard's ledger instead of browser and notebook runs. |

Without `--legacy`, each line is a browser-console, REST API or notebook run:

```text
12: completed Qwen/Qwen2.5-0.5B-Instruct -> ./modelforge_output
```

That is `id: status model_id -> output_dir`. With no runs it prints `No runs recorded yet.`

With `--legacy`, each line is a completed terminal wizard run:

```text
3: 2026-11-14T18:02:11.482913 Qwen/Qwen2.5-0.5B-Instruct (qlora_4bit) -> ./modelforge_output/Qwen2.5-0.5B-Instruct_20261114T174512/
```

That is `id: timestamp model_id (peft) -> output_dir`, with the timestamp in UTC ISO format. With no runs it prints `No legacy finetune runs recorded yet.`

Both ledgers live in `modelforge.db` in the user data folder (`~/.local/share/ModelForge` on Linux).

## modelforge list-tasks

Lists the use cases that have at least one trainable objective, with their objectives and descriptions.

```bash
modelforge list-tasks
```

```text
$ modelforge list-tasks
instruction_following: Instruction Following [supervised, preference, reinforcement]
  Train a model to follow natural-language instructions and produce helpful, accurate responses. ...
...
```

Each entry is `slug: name [objectives]` followed by an indented description. The slugs are the values for `modelforge recommend --use-case` and for `use_case_slug` in the REST API. See [Use cases and tasks](../configuration/training-tasks.md).

## modelforge tensorboard

Starts TensorBoard in the foreground for a log directory. Stop it with Ctrl+C.

```bash
modelforge tensorboard [LOGDIR] [--host HOST] [--port PORT]
```

| Argument | Default | Effect |
|---|---|---|
| `LOGDIR` | `./modelforge_output` | Directory to read event logs from. TensorBoard searches it recursively, so a parent folder shows every run under it. |
| `--host HOST` | `127.0.0.1` | Interface to bind. |
| `--port PORT` | `6006` | Port to bind. |

It runs `tensorboard --logdir LOGDIR --host HOST --port PORT` and prints the URL first. If the `tensorboard` executable is not on `PATH`, it prints `TensorBoard is not installed or is not on PATH.` TensorBoard itself is a base dependency, but this helper needs the `[cli]` extra.

!!! note
    While a run with TensorBoard logging is training, the trainer already serves TensorBoard at `127.0.0.1:6006` from inside the training process. Use a different `--port` if you start this command at the same time.

## modelforge-nb

Starts the prompt-driven notebook interface. It takes no arguments.

```bash
modelforge-nb
```

The same entry point is available from Python, which is the way to use it inside Jupyter or Colab:

```python
from notebook_cli import main

main()
```

It shows a menu with **1. Fine-tune a model**, **2. List runs** and **3. Exit**. Training starts in a background thread; exiting the `modelforge-nb` process kills it.

Guide: [Notebooks](../user-guide/notebooks.md).

## Running from a source checkout

Without installing the console scripts, the same commands run as modules from the repository root:

```bash
python -m cli.main run --no-browser
python -m notebook_cli.main
```

## See also

- [Browser console](../user-guide/browser-console.md)
- [Terminal wizard](../user-guide/terminal-cli.md)
- [Notebooks](../user-guide/notebooks.md)
- [REST API](rest-api.md)
