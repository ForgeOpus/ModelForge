# Quick Start

This page takes you from an empty environment to a finished fine-tune in the browser console. It assumes an NVIDIA GPU on Linux. On Windows or macOS, read the installation page for your platform first, then come back for the first run.

## Requirements

- **Python 3.13** or newer. v4 does not run on Python 3.11 or 3.12.
- **An NVIDIA GPU with CUDA** for real training. Apple Silicon (MPS) and CPU-only machines work, but only with small models. See [Hardware and Memory](../configuration/hardware-profiles.md).
- **Disk space** for base models, checkpoints and outputs. The plan warns when free space looks too small for the model's weights.

## 1. Install

Create a virtual environment and install the package:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install modelforge-finetuning
```

Optional extras:

| Extra | Adds | When to use it |
|---|---|---|
| `[cli]` | click, rich | The terminal wizard (`modelforge cli`) and `modelforge tensorboard`. |
| `[gpu]` | liger-kernel | Fused kernels that cut memory on CUDA GPUs. No CPU or Windows wheels. |

```bash
pip install "modelforge-finetuning[cli,gpu]"
```

PyTorch is not a direct dependency. It is pulled in by `accelerate`, `peft` and `bitsandbytes`. On Linux the default PyPI wheel includes CUDA. On Windows it does not, so install a CUDA build first. See [Windows](../installation/windows.md).

Check that PyTorch can see your GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Start the server

```
$ modelforge run
ModelForge v4 browser UI
Serving on: http://0.0.0.0:8000
Open:       http://127.0.0.1:8000/app?token=…
The startup token is required for API and WebSocket access.
```

Your browser opens the **Open** link automatically. The console stores the token, removes it from the address bar, and sends it with every request. A new token is generated each time the server starts.

| Flag | Default | Effect |
|---|---|---|
| `--host` | `0.0.0.0` | Interface to bind. |
| `--port` | `8000` | Port to bind. |
| `--no-browser` | off | Print the link without opening a browser. Use this on remote machines. |
| `--token` | random | Use a fixed token so bookmarks keep working across restarts. |

!!! warning "The server listens on every interface by default"
    Anyone on your network who has the token can start runs and read your data. Pass `--host 127.0.0.1` on shared networks, and reach a remote machine through an SSH tunnel. See [Post-Installation Setup](../installation/post-installation.md).

Public models and datasets need no login. For gated models such as Llama, accept the licence on the model's Hub page and run `hf auth login` once on the machine running the server.

## 3. Prepare a small dataset

Save this as `data/train.jsonl`. It is an instruction-following dataset for the supervised objective: one JSON object per line with `prompt` and `completion`.

```json
{"prompt": "Summarise this ticket: The app logs me out every time I update it.", "completion": "Customer is logged out after each update."}
{"prompt": "Summarise this ticket: I was charged twice for my March invoice.", "completion": "Customer was double-charged for the March invoice."}
{"prompt": "Summarise this ticket: The export button does nothing in Firefox.", "completion": "Export button is unresponsive in Firefox."}
```

`instruction` and `response` are accepted as aliases. Three rows is enough to check the pipeline end to end, not to train a useful model; a real dataset needs hundreds to thousands of rows. Column requirements for every use case and objective are in [Dataset Formats](../configuration/dataset-formats.md).

Paths are resolved on the server's machine, relative to the directory where you ran `modelforge run`.

## 4. Plan and start a run

Open **New run** in the console. The left side is a four-step form; the right side is the **Plan**.

1. **Task.** Pick **Instruction Following**, then the **Supervised** objective. Objectives a use case can't train are greyed out.
2. **Dataset.** Enter `./data/train.jsonl`, keep **Local** selected, and press **Validate**. It reports the normalized columns or the first problem it finds.
3. **Base model.** Press **Generate recommendation**. The server detects your hardware, picks a tier, and searches the Hub for up to five models that fit. Pick one from the table, or type any Hub model ID into **Or enter a different model ID**.
4. **Output.** Set an output directory, for example `./modelforge_output/first-run`, leave **Write TensorBoard logs** on, and press **Start training**.

Before you start, read the Plan: the tier and model, the strategy and parallelism, the **VRAM estimate**, the **PEFT decision**, the full **Training config** and any warnings. The config is read-only. [How Planning Works](../configuration/configuration-guide.md) explains each part.

!!! tip "Give each run its own output directory"
    The form defaults to `./modelforge_output`. Two runs writing to the same folder overwrite each other's weights and checkpoints.

## 5. Watch it train

Starting a run takes you to its detail page:

- **Training log** streams status, loss and step updates live.
- **Checkpoints** lists each saved step (every 500 steps), with **Resume** on failed or cancelled runs.
- **Cancel run** stops after the current step.

Only one run trains at a time. Starting another while one is active is refused.

To see the loss curves, launch TensorBoard on the output folder (needs the `[cli]` extra):

```bash
modelforge tensorboard ./modelforge_output/first-run
```

It serves on `http://127.0.0.1:6006` by default.

## 6. Try the result

When the run completes, open **Playground**, pick the run, choose **Final model** or a checkpoint, and send a message. See [Playground](../user-guide/playground.md).

The output directory holds the trained weights (or only the adapter for LoRA and QLoRA runs), the tokenizer, checkpoints, TensorBoard logs and a `modelforge_config.json` describing how the model was trained.

## Without a browser

On a headless server, use the terminal wizard instead. It asks the same questions and trains in the terminal:

```bash
pip install "modelforge-finetuning[cli]"
modelforge cli
```

To see what the planner would choose without training:

```bash
modelforge recommend --use-case instruction_following --objective supervised
```

See [Terminal Wizard](../user-guide/terminal-cli.md) and [CLI Commands](../api-reference/cli.md).

## Stopping and restarting

Press Ctrl+C in the terminal running `modelforge run`. Any run that was queued or running when the server stopped is marked failed on the next start; its checkpoints are intact, so you can resume it. See [Checkpoints and Resuming](../user-guide/checkpoints.md).

## Next steps

- [Fine-Tuning Concepts](concepts.md) if you are new to fine-tuning.
- [Dataset Formats](../configuration/dataset-formats.md) to prepare real data.
- [Browser Console](../user-guide/browser-console.md) for every page of the console.
- [Common Issues](../troubleshooting/common-issues.md) if something goes wrong.
