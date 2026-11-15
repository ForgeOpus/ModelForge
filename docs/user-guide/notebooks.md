# Notebooks

`modelforge-nb` is a plain-Python, prompt-driven interface for Jupyter and Colab. It asks its questions with `input()`, so it needs no widgets and no terminal menus, and it starts training in the background so the notebook stays responsive. This page covers how to launch it, every prompt, where runs are recorded, and its limits.

## What it is

The notebook interface is a single function, `notebook_cli.main()`, also installed as the `modelforge-nb` command. It uses the same planning, dataset validation and training code as the browser console, and records runs in the same database, so notebook runs show up in the console's **Runs** page and in `modelforge history`.

It needs no extra beyond the base package. (A `[notebook]` extra is declared but adds nothing.)

There is no other public notebook API. For scripted, non-interactive use, call the [REST API](../api-reference/rest-api.md) of a running `modelforge run` server.

## Launching it

=== "Python cell (recommended)"

    ```python
    from notebook_cli import main

    main()
    ```

    The prompts appear under the cell and read your answers through the notebook's input box. Training runs in a background thread of the kernel, so it keeps going after `main()` returns.

=== "Terminal"

    ```bash
    modelforge-nb
    ```

    Useful in a plain terminal or a Jupyter terminal tab. Training runs in a background thread of this process, so **do not choose Exit while a run is training**: exiting ends the process and kills the run.

!!! note "Avoid `!modelforge-nb` in a cell"
    Jupyter shell commands (`!`) generally do not pass keyboard input to the program, so the prompts cannot be answered. Call `main()` from a Python cell instead.

The package must be importable from the kernel. In Colab, install it first:

```python
%pip install modelforge-finetuning
```

ModelForge requires Python 3.13 or newer. Check the runtime's version with `import sys; sys.version` before installing.

## The menu

```text
ModelForge Notebook UI
Pure Python prompts; no widgets or terminal menus.

1. Fine-tune a model
2. List runs
3. Exit
Choose an option:
```

| Option | Does |
|---|---|
| **1** | Runs the fine-tune prompts below and starts training in the background. |
| **2** | Prints every recorded run as `id: status model_id -> output_dir`. Use it to check on a background run. |
| **3** | Returns from `main()`. In a notebook the kernel, and any training, keeps running. |

The menu repeats until you choose 3.

## Fine-tune prompts

1. **Use case.** A numbered list of use cases, then `Select use case:`. See [Use cases and tasks](../configuration/training-tasks.md).
2. **Objective.** `Select objective:` from the use case's objectives, or chosen automatically when there is only one.
3. **Model.** The recommendation is computed for every detected GPU, and up to five models are listed with their parameter counts, plus `Enter a custom model id`. Choosing that asks for `Model id:`.
4. **Dataset.** `Dataset path or HuggingFace dataset id:`, then `Is this a HuggingFace dataset id? [y/N]:`.
5. **Validation.** The dataset is checked against the use case and objective, as the console's **Validate** does. If it fails, the error and the expected format are printed and you return to the menu.
6. **Output directory.** `Output directory:`. Press Enter for `./modelforge_output`, relative to the kernel's working directory.

The run then starts and prints:

```text
Started run 12. Use the browser UI or history command to monitor details.
```

Numbers are 1-based; an out-of-range or non-numeric answer re-asks.

## What a notebook run does

Notebook runs go through the same job manager as the browser console:

- They train on every detected GPU. There is no GPU selection prompt.
- TensorBoard logging is on, under `<output dir>/tensorboard/run-<id>`. With it on, the trainer also starts a TensorBoard server inside the kernel at `http://127.0.0.1:6006`.
- Checkpoints are saved every 500 steps and recorded for the run.
- On completion, `modelforge_config.json` is written to the output folder and the model is registered as an artifact.
- Only one run can train at a time within the session. Starting a second one, or entering a custom model that cannot be configured, raises the error out of `main()`; call `main()` again to return to the menu.

A custom model ID is checked against the Hub and the plan is rebuilt for it before training, as in the console.

There is no cancel option in the notebook interface. Interrupting the cell does not stop a background run; restarting the kernel does, and the run stays marked `running` until the next ModelForge startup marks it `failed`.

## Monitoring and resuming

- Choose option **2** for a quick status list.
- Run `modelforge history` in a terminal for the same list.
- Start `modelforge run` to see notebook runs and their checkpoints in the [browser console](browser-console.md). While a notebook run is training, its events live only in the kernel's memory, so the console shows its status and checkpoints but an empty training log. The timeline is saved to the database when the run ends, and the console shows it from then on.
- Resume a failed or interrupted notebook run from its checkpoints in the console. See [Checkpoints and resuming](checkpoints.md).

## See also

- [Browser console](browser-console.md)
- [Terminal wizard](terminal-cli.md)
- [CLI reference](../api-reference/cli.md#modelforge-nb)
- [Dataset formats](../configuration/dataset-formats.md)
