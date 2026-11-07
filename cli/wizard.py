import os
import sys
from datetime import datetime

from huggingface_hub import login, get_token
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from utilities.hardware_detector import HardwareDetector
from utilities.training_objective import TrainingObjective
from recommendation_engine.language_use_cases.use_case import LanguageUseCase
from finetuning_strategies.strategies.strategy import FinetuningStrategy
from recommendation_engine.validation_models.recommendation import Recommendation
from dataset_manager.dataset_factory import DatasetFactory
from datasets import IterableDataset, DatasetDict
from workflow.registries import load_core_registries

console = Console()
load_core_registries()


def check_hf_login() -> None:
    token = os.environ.get("HF_TOKEN") or get_token()
    if token:
        rprint("[green]HuggingFace Hub: logged in ✓[/green]")
        return
    rprint("[yellow]HuggingFace Hub: not logged in.[/yellow]")
    rprint("Without a token, gated models and datasets will be inaccessible.")
    raw = console.input("Enter your HF token (or press Enter to skip): ").strip()
    if raw:
        login(token=raw, add_to_git_credential=False)
        rprint("[green]Logged in to HuggingFace Hub ✓[/green]")
    else:
        rprint("[dim]Skipping — only public models and datasets will be available.[/dim]")


_OBJECTIVE_TO_SLUG = {
    TrainingObjective.SUPERVISED: "sft",
    TrainingObjective.PREFERENCE: "dpo",
    TrainingObjective.REINFORCEMENT: "grpo",
}


def select_gpus(hardware: HardwareDetector.HardwareSpecs) -> tuple[HardwareDetector.HardwareSpecs, str]:
    gpus = hardware.gpus

    if len(gpus) == 0:
        rprint("[yellow]Warning: No GPU detected — running in CPU-only mode[/yellow]")
        return hardware, ""

    if len(gpus) == 1:
        rprint(f"Auto-selected: {gpus[0]['name']}")
        return hardware, "0"

    for i, gpu in enumerate(gpus):
        rprint(f"  [{i + 1}] {gpu['name']} — {gpu['total_memory_gb']} GB VRAM")

    while True:
        raw = console.input("Select GPUs (e.g. 1,2 or 'all'): ").strip()
        if raw.lower() == "all":
            indices = list(range(len(gpus)))
        else:
            try:
                one_indexed = [int(x.strip()) for x in raw.split(",")]
                if any(n < 1 or n > len(gpus) for n in one_indexed):
                    rprint(f"[red]Please enter numbers between 1 and {len(gpus)}.[/red]")
                    continue
                indices = [n - 1 for n in one_indexed]
            except ValueError:
                rprint("[red]Invalid input. Enter comma-separated numbers or 'all'.[/red]")
                continue

        chosen_gpus = [gpus[i] for i in indices]
        filtered = hardware.model_copy(update={"gpus": chosen_gpus})
        cuda_string = ",".join(str(i) for i in indices)
        return filtered, cuda_string


def select_use_case() -> type:
    slugs = list(LanguageUseCase.get_use_cases())

    table = Table(show_header=True)
    table.add_column("#", justify="right")
    table.add_column("Name")
    table.add_column("Description")

    for i, slug in enumerate(slugs, start=1):
        cls = LanguageUseCase.get_use_case_by_slug(slug)
        table.add_row(str(i), cls.name, cls.description)

    console.print(table)

    while True:
        raw = console.input(f"Select use case [1-{len(slugs)}]: ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(slugs):
                return LanguageUseCase.get_use_case_by_slug(slugs[choice - 1])
        except ValueError:
            pass
        rprint(f"[red]Enter a number between 1 and {len(slugs)}.[/red]")


def select_objective(use_case_cls: type) -> TrainingObjective:
    valid = [
        obj for obj in use_case_cls.compatible_objectives
        if (getattr(use_case_cls, "strategy_slug", None) or _OBJECTIVE_TO_SLUG.get(obj)) in FinetuningStrategy._registry
    ]

    if len(valid) == 1:
        rprint(f"Auto-selected: {valid[0].value}")
        return valid[0]

    for i, obj in enumerate(valid, start=1):
        rprint(f"  [{i}] {obj.value}")

    while True:
        raw = console.input(f"Select training objective [1-{len(valid)}]: ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(valid):
                return valid[choice - 1]
        except ValueError:
            pass
        rprint(f"[red]Enter a number between 1 and {len(valid)}.[/red]")


def select_model(recommendation: Recommendation) -> str:
    models = recommendation.recommended_models

    table = Table(show_header=True)
    table.add_column("#", justify="right")
    table.add_column("Model ID")
    table.add_column("Parameters", justify="right")
    table.add_column("Downloads", justify="right")

    for i, m in enumerate(models, start=1):
        table.add_row(str(i), m.model_id, f"{m.num_parameters:,}", f"{m.downloads:,}")

    custom_index = len(models) + 1
    table.add_row(str(custom_index), "[dim]Enter custom model ID[/dim]", "", "")
    console.print(table)

    while True:
        raw = console.input(f"Select model [1-{custom_index}]: ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(models):
                return models[choice - 1].model_id
            if choice == custom_index:
                return console.input("Custom model ID: ").strip()
        except ValueError:
            pass
        rprint(f"[red]Enter a number between 1 and {custom_index}.[/red]")


def select_dataset() -> tuple[str, bool]:
    rprint("  [1] Local file/directory")
    rprint("  [2] HuggingFace Hub dataset")

    while True:
        raw = console.input("Select dataset source [1-2]: ").strip()
        if raw == "1":
            while True:
                path = console.input("Path to local file or directory: ").strip()
                if os.path.exists(path):
                    return path, False
                rprint(f"[red]Path not found: {path}[/red]")
        elif raw == "2":
            name = console.input("HuggingFace dataset name (e.g. tatsu-lab/alpaca): ").strip()
            return name, True
        else:
            rprint("[red]Enter 1 or 2.[/red]")


def validate_and_load_dataset(path: str, is_remote: bool, recommendation: Recommendation):
    dataset = DatasetFactory.load(path, remote=is_remote)

    if isinstance(dataset, IterableDataset):
        rprint("[yellow]Warning: Streaming dataset detected — skipping column validation.[/yellow]")
        return dataset

    if isinstance(dataset, DatasetDict):
        split = next(iter(dataset.values()))
    else:
        split = dataset

    try:
        split = DatasetFactory.normalize_columns(
            split,
            recommendation.data_format.column_aliases,
            recommendation.data_format.optional_column_aliases,
        )
        strategy_cls = FinetuningStrategy._registry.get(recommendation.strategy_slug)
        if strategy_cls is None:
            raise ValueError(f"Training strategy '{recommendation.strategy_slug}' is unavailable.")
        strategy_cls.validate_data_format(split)
    except ValueError:
        rprint(f"\n[red]Dataset format invalid.[/red]")
        rprint(recommendation.data_format.user_facing_description)
        answer = console.input("Try a different dataset? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            path, is_remote = select_dataset()
            return validate_and_load_dataset(path, is_remote, recommendation)
        sys.exit(1)

    return split


def prompt_output_dir(model_id: str) -> str:
    short_name = model_id.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    default = f"./modelforge_output/{short_name}_{timestamp}/"
    rprint(f"Default output directory: [dim]{default}[/dim]")
    entered = console.input("Output directory [press Enter for default]: ").strip()
    return entered if entered else default
