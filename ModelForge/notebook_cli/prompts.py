"""
Interactive prompt steps for the ModelForge notebook CLI wizard.

Each function handles one wizard step and returns a typed value.
Uses questionary in terminals; falls back to plain input() in notebooks.
Display panels use rich.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from ..services.training_service import TrainingService

console = Console()


def _is_notebook() -> bool:
    """Return True when running inside a Jupyter/Colab/Databricks kernel."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


_IN_NOTEBOOK = _is_notebook()


# ── Notebook-safe input() fallbacks ──────────────────────────────────────────

def _nb_select(message: str, choices: list) -> str:
    """Numbered-menu fallback for questionary.select()."""
    print(f"\n{message}")
    for i, choice in enumerate(choices, 1):
        title = choice.title if hasattr(choice, "title") else str(choice)
        print(f"  {i}) {title}")
    while True:
        raw = input("Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            c = choices[int(raw) - 1]
            return c.value if hasattr(c, "value") else c
        print(f"  Please enter a number between 1 and {len(choices)}.")


def _nb_text(message: str, default: str = "", validate=None) -> str:
    """Plain input() fallback for questionary.text()."""
    while True:
        raw = input(f"{message} ").strip()
        if not raw and default:
            raw = default
        if validate is None:
            return raw
        result = validate(raw)
        if result is True:
            return raw
        print(f"  {result}")


def _nb_path(message: str, validate=None) -> str:
    """Plain input() fallback for questionary.path()."""
    return _nb_text(message, validate=validate)


def _nb_confirm(message: str, default: bool = True) -> bool:
    """y/n fallback for questionary.confirm()."""
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{message} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n.")

# ── Choices ────────────────────────────────────────────────────────────────

_TASK_CHOICES = [
    questionary.Choice("Text Generation  (causal LM fine-tuning)", value="text-generation"),
    questionary.Choice("Summarization    (seq-to-seq fine-tuning)", value="summarization"),
    questionary.Choice(
        "Extractive QA    (question-answering fine-tuning)",
        value="extractive-question-answering",
    ),
]

_PROVIDER_CHOICES = [
    questionary.Choice("HuggingFace  (default, broad compatibility)", value="huggingface"),
    questionary.Choice("Unsloth      (faster training, single-GPU optimised)", value="unsloth"),
]

_STRATEGY_CHOICES = [
    questionary.Choice("SFT    – Supervised Fine-Tuning (recommended for most tasks)", value="sft"),
    questionary.Choice("QLoRA  – Quantised LoRA (memory-efficient fine-tuning)", value="qlora"),
    questionary.Choice("DPO    – Direct Preference Optimisation", value="dpo"),
    questionary.Choice("RLHF   – Preference alignment (DPO-based, no reward model needed)", value="rlhf"),
]

_TASK_FORMAT_HELP = {
    "text-generation": 'JSON Lines with "input" and "output" fields.',
    "summarization": 'JSON Lines with "document" and "summary" fields.',
    "extractive-question-answering": 'JSON Lines with "context", "question", and "answers" fields.',
}

_STRATEGY_FORMAT_HELP = {
    "dpo":  'JSON Lines with "prompt", "chosen", and "rejected" fields.',
    "rlhf": 'JSON Lines with "prompt", "chosen", and "rejected" fields.',
}


# ── Step functions ──────────────────────────────────────────────────────────

def prompt_task() -> str:
    """Ask the user to select a fine-tuning task."""
    if _IN_NOTEBOOK:
        return _nb_select("Select fine-tuning task:", _TASK_CHOICES)
    return questionary.select(
        "Select fine-tuning task:",
        choices=_TASK_CHOICES,
        style=questionary.Style([("selected", "fg:cyan bold")]),
    ).ask()


def prompt_model(recommended: str, alternatives: list[str]) -> str:
    """
    Ask the user to select or enter a model name.

    Args:
        recommended: Hardware-recommended primary model.
        alternatives: List of alternative model suggestions.

    Returns:
        HuggingFace model ID string.
    """
    choices = [questionary.Choice(f"{recommended}  (recommended for your hardware)", value=recommended)]
    for alt in alternatives:
        choices.append(questionary.Choice(alt, value=alt))
    choices.append(questionary.Choice("Enter a custom model ID…", value="__custom__"))

    if _IN_NOTEBOOK:
        selection = _nb_select("Select a base model:", choices)
        if selection == "__custom__":
            selection = _nb_text(
                "Enter HuggingFace model ID (e.g. meta-llama/Llama-3.2-1B):",
                validate=lambda v: bool(v.strip()) or "Model ID cannot be empty",
            )
        return selection

    selection = questionary.select(
        "Select a base model:",
        choices=choices,
        style=questionary.Style([("selected", "fg:cyan bold")]),
    ).ask()

    if selection == "__custom__":
        selection = questionary.text(
            "Enter HuggingFace model ID (e.g. meta-llama/Llama-3.2-1B):",
            validate=lambda v: bool(v.strip()) or "Model ID cannot be empty",
        ).ask()

    return selection


def prompt_dataset(task: str, strategy: str, training_service: "TrainingService") -> str:
    """
    Ask for a dataset path and validate it via TrainingService.

    Args:
        task: Selected task type.
        strategy: Selected strategy.
        training_service: TrainingService instance for validation.

    Returns:
        Validated dataset path string.
    """
    fmt = _STRATEGY_FORMAT_HELP.get(strategy) or _TASK_FORMAT_HELP.get(task, "JSON Lines file.")
    console.print(f"\n[dim]Expected format:[/dim] {fmt}")

    _path_validate = lambda v: os.path.isfile(v) or "File not found — please enter a valid path"

    while True:
        if _IN_NOTEBOOK:
            path = _nb_path("Path to dataset file (.jsonl / .json):", validate=_path_validate)
        else:
            path = questionary.path(
                "Path to dataset file (.jsonl / .json):",
                validate=lambda v: os.path.isfile(v.strip()) or "File not found — please enter a valid path",,
            ).ask()

        if path is not None:
            path = path.strip()

        if path is None:
            raise KeyboardInterrupt

        console.print("  [dim]Validating dataset…[/dim]")
        try:
            info = training_service.validate_and_prepare_dataset(path, task, strategy)
            console.print(
                f"  [green]✓[/green] Dataset valid — "
                f"[bold]{info['num_examples']}[/bold] examples, "
                f"fields: {info['fields']}"
            )
            return path
        except Exception as exc:
            console.print(f"  [red]✗[/red] Validation failed: {exc}")
            if _IN_NOTEBOOK:
                retry = _nb_confirm("Try a different file?", default=True)
            else:
                retry = questionary.confirm("Try a different file?", default=True).ask()
            if not retry:
                raise


def prompt_provider() -> str:
    """Ask the user to choose a training provider."""
    if _IN_NOTEBOOK:
        return _nb_select("Select training provider:", _PROVIDER_CHOICES)
    return questionary.select(
        "Select training provider:",
        choices=_PROVIDER_CHOICES,
        style=questionary.Style([("selected", "fg:cyan bold")]),
    ).ask()


def prompt_strategy() -> str:
    """Ask the user to choose a training strategy."""
    if _IN_NOTEBOOK:
        return _nb_select("Select training strategy:", _STRATEGY_CHOICES)
    return questionary.select(
        "Select training strategy:",
        choices=_STRATEGY_CHOICES,
        style=questionary.Style([("selected", "fg:cyan bold")]),
    ).ask()


def prompt_hyperparams(defaults: dict) -> dict:
    """
    Interactively configure hyperparameters, showing defaults.

    Users can press Enter to keep each default value.

    Args:
        defaults: Dictionary of default hyperparameter values.

    Returns:
        Dictionary with final hyperparameter values.
    """
    console.print(
        Panel(
            "[bold]Hyperparameter Configuration[/bold]\n"
            "[dim]Press Enter to accept the default for each setting.[/dim]",
            expand=False,
        )
    )

    params = {}

    def _ask_int(key: str, question: str) -> int:
        default = defaults.get(key)
        validate = lambda v: v.isdigit() and int(v) >= 1 or "Must be a positive integer"
        if _IN_NOTEBOOK:
            raw = _nb_text(f"{question} [default: {default}]:", default=str(default), validate=validate)
        else:
            raw = questionary.text(
                f"{question} [default: {default}]:",
                default=str(default),
                validate=validate,
            ).ask()
        return int(raw)

    def _ask_float(key: str, question: str, min_val: float = 0.0) -> float:
        default = defaults.get(key)
        validate = lambda v: _valid_float(v, min_val)
        if _IN_NOTEBOOK:
            raw = _nb_text(f"{question} [default: {default}]:", default=str(default), validate=validate)
        else:
            raw = questionary.text(
                f"{question} [default: {default}]:",
                default=str(default),
                validate=validate,
            ).ask()
        return float(raw)

    def _valid_float(v: str, min_val: float) -> bool | str:
        try:
            f = float(v)
            return f > min_val or f"Must be > {min_val}"
        except ValueError:
            return "Must be a number"

    params["num_train_epochs"] = _ask_int("num_train_epochs", "Training epochs")
    params["per_device_train_batch_size"] = _ask_int(
        "per_device_train_batch_size", "Batch size per device"
    )
    params["gradient_accumulation_steps"] = _ask_int(
        "gradient_accumulation_steps", "Gradient accumulation steps"
    )
    params["learning_rate"] = _ask_float("learning_rate", "Learning rate", min_val=0.0)
    params["lora_r"] = _ask_int("lora_r", "LoRA rank (r)")
    params["lora_alpha"] = _ask_int("lora_alpha", "LoRA alpha")
    params["max_seq_length"] = _ask_int("max_seq_length", "Max sequence length")

    quant_choices = [
        questionary.Choice("4-bit (QLoRA, saves VRAM)", value="4bit"),
        questionary.Choice("8-bit (moderate savings)", value="8bit"),
        questionary.Choice("None (full precision / no bitsandbytes required)", value="none"),
    ]
    default_quant = "4bit" if defaults.get("use_4bit") else ("8bit" if defaults.get("use_8bit") else "none")
    if _IN_NOTEBOOK:
        quant = _nb_select(f"Quantization [default: {default_quant}]:", quant_choices)
    else:
        quant = questionary.select(
            f"Quantization [default: {default_quant}]:",
            choices=quant_choices,
            default=quant_choices[["4bit", "8bit", "none"].index(default_quant)],
        ).ask()
    params["use_4bit"] = quant == "4bit"
    params["use_8bit"] = quant == "8bit"

    if _IN_NOTEBOOK:
        params["use_chat_template"] = _nb_confirm(
            "Use model's native chat template for formatting? "
            "(recommended for instruct models)",
            default=False,
        )
    else:
        params["use_chat_template"] = questionary.confirm(
            "Use model's native chat template for formatting? "
            "(recommended for instruct models)",
            default=False,
        ).ask()

    return params


def prompt_confirm_config(config: dict) -> bool:
    """
    Show a summary table of the final config and ask for confirmation.

    Args:
        config: Full training configuration dictionary.

    Returns:
        True if confirmed, False if cancelled.
    """
    table = Table(title="Training Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="dim", width=30)
    table.add_column("Value")

    display_keys = [
        ("task", "Task"),
        ("model_name", "Base model"),
        ("provider", "Provider"),
        ("strategy", "Strategy"),
        ("dataset", "Dataset"),
        ("compute_specs", "Compute profile"),
        ("num_train_epochs", "Epochs"),
        ("per_device_train_batch_size", "Batch size"),
        ("gradient_accumulation_steps", "Gradient accum. steps"),
        ("learning_rate", "Learning rate"),
        ("lora_r", "LoRA rank"),
        ("lora_alpha", "LoRA alpha"),
        ("max_seq_length", "Max seq length"),
        ("use_4bit", "4-bit quantization"),
        ("use_8bit", "8-bit quantization"),
        ("use_chat_template", "Use chat template"),
    ]

    for key, label in display_keys:
        if key in config:
            table.add_row(label, str(config[key]))

    console.print(table)

    if _IN_NOTEBOOK:
        return _nb_confirm("Start training with this configuration?", default=True)
    return questionary.confirm(
        "Start training with this configuration?", default=True
    ).ask()
