# Terminal Wizard

`modelforge cli` is an interactive wizard that plans and runs a fine-tune entirely in the terminal, which suits headless servers reached over SSH. It uses the same hardware detection, planning, dataset validation and training code as the browser console. This page lists what it needs, every prompt in order, and where its output and history go.

## Requirements

The wizard needs the optional `[cli]` extra, which adds `click` and `rich`:

```bash
pip install "modelforge-finetuning[cli]"
```

Without it, `modelforge cli` exits with:

```text
The terminal UI requires optional dependencies. Install with: pip install "modelforge-finetuning[cli]"
```

## Starting the wizard

```bash
modelforge cli
```

| Flag | Effect |
|---|---|
| `--no-tensorboard` | Do not write TensorBoard logs. |
| `--tensorboard-logdir DIR` | Write TensorBoard logs under `DIR` instead of `<output dir>/tensorboard`. Ignored with `--no-tensorboard`. |

TensorBoard logging is on by default.

## The prompts, in order

### 1. Banner and Hugging Face login

The wizard prints the ModelForge banner, then checks for a Hugging Face token in the `HF_TOKEN` environment variable or the cached login.

- If one is found: `HuggingFace Hub: logged in ✓`.
- If not, it explains that gated models and datasets will be inaccessible and asks:

  ```text
  Enter your HF token (or press Enter to skip):
  ```

  A pasted token is saved with `huggingface_hub.login` (not added to git credentials), so later runs find it. Press Enter to continue with public models and datasets only.

### 2. Detected hardware

A **Detected Hardware** table shows CPU cores, RAM, free and total disk, and one row per GPU with its VRAM (Apple Silicon shows `Apple MPS — shared`).

### 3. GPU selection

| GPUs found | Behaviour |
|---|---|
| None | Warns `No GPU detected — running in CPU-only mode` and continues. |
| One | `Auto-selected: <name>`. |
| Several | Lists them numbered from 1 and asks `Select GPUs (e.g. 1,2 or 'all'):`. |

Only the selected GPUs are used for planning and training. Selecting two or more trains on multiple processes; see [Multi-GPU](../strategies/multi-gpu.md).

### 4. Use case

A table of use cases with number, name and description, then `Select use case [1-N]:`. See [Use cases and tasks](../configuration/training-tasks.md).

### 5. Training objective

Lists the objectives the use case supports and that have a registered strategy, then asks `Select training objective [1-N]:`. If only one applies (Summarization and Extractive QA), it prints `Auto-selected: supervised` and moves on.

### 6. Recommendation

The wizard prints `Finding best models for your hardware...`, plans for the selected hardware with up to five candidates, and shows:

- a **Recommendation** panel: strategy, PEFT tier, task and provider,
- a **Candidate Models** table: rank, model ID, parameters (in billions) and downloads,
- a **Decision Rationale**: hardware, use case, PEFT and model filter,
- a **Warnings** panel, if there are any.

See [How planning works](../configuration/configuration-guide.md).

### 7. Model

The candidates are listed again, with one extra row, **Enter custom model ID**, then `Select model [1-N]:`. Choosing the last row asks for `Custom model ID:`.

### 8. Dataset

```text
  [1] Local file/directory
  [2] HuggingFace Hub dataset
Select dataset source [1-2]:
```

- **1** asks `Path to local file or directory:` and repeats until the path exists.
- **2** asks `HuggingFace dataset name (e.g. tatsu-lab/alpaca):`.

The dataset is then loaded and checked. If it has several splits, the first is used. Known column aliases are renamed, then the strategy validates the columns. If validation fails, the wizard prints `Dataset format invalid.`, the expected format, and asks `Try a different dataset? [Y/n]:`. Answering `n` exits with status 1. Streaming datasets skip column validation with a warning.

See [Dataset formats](../configuration/dataset-formats.md).

### 9. Output directory

```text
Default output directory: ./modelforge_output/<model short name>_<YYYYMMDDTHHMMSS>/
Output directory [press Enter for default]:
```

The default includes the model name and a timestamp, so separate runs do not overwrite each other.

### 10. Summary and confirmation

A **Training Summary** panel shows the model, dataset, task, strategy, PEFT approach, output directory and, if enabled, the TensorBoard directory. Then:

```text
Proceed with training? [Y/n]:
```

Answering `n` aborts without training.

## Training

Training runs in the foreground with a progress bar showing percentage, elapsed time, `step/total`, and the latest loss and epoch. `Training complete!` is printed at the end.

With TensorBoard logging on, the trainer also starts a TensorBoard server inside the process at `http://127.0.0.1:6006` and prints `TensorBoard started at …`. It tries to open a browser, which does nothing on a headless machine; forward the port with `ssh -L 6006:127.0.0.1:6006 user@host` to view it from your own machine. The server stops when the wizard exits.

Stop a run with Ctrl+C. There is no cancel-and-resume flow in the wizard. Checkpoints are written to `<output dir>/checkpoint-<step>` every 500 steps. Use the [browser console](browser-console.md) for runs you want to resume. See [Checkpoints and resuming](checkpoints.md).

## After training

When training finishes, the wizard prints:

```text
Model saved to ./modelforge_output/<name>_<timestamp>/
TensorBoard logs: ./modelforge_output/<name>_<timestamp>/tensorboard
Open with: tensorboard --logdir ./modelforge_output/<name>_<timestamp>/tensorboard --host 127.0.0.1 --port 6006
```

Each launch writes its TensorBoard events to its own `run-<random hex>` subfolder of the TensorBoard directory, so pointing TensorBoard at the printed directory shows every launch that used it. `modelforge tensorboard <dir>` starts the same viewer.

## History

Completed wizard runs are recorded in a separate **legacy** ledger in the same `modelforge.db` database (in the user data folder, `~/.local/share/ModelForge` on Linux), with model, use case, task, strategy, PEFT tier, dataset, output directory and the training config. A run is recorded only after training succeeds.

```bash
modelforge history --legacy
```

`modelforge history` without `--legacy` shows the browser console and notebook runs instead.


## See also

- [CLI reference](../api-reference/cli.md)
- [Browser console](browser-console.md)
- [Notebooks](notebooks.md)
- [Performance and memory](../troubleshooting/performance.md)
