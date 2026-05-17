from transformers import TrainerCallback
from rich.progress import Progress, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table, box
from rich.text import Text
from rich.rule import Rule
from rich import print as rprint
from utilities.hardware_detector import HardwareDetector
from recommendation_engine.validation_models.recommendation import Recommendation

console = Console()


def print_header() -> None:
    banner_lines = [
        (" __  __           _      _  _____", "bold red"),
        ("|  \\/  | ___   __| | ___| ||  ___|__  _ __ __ _  ___", "bold yellow"),
        ("| |\\/| |/ _ \\ / _` |/ _ \\ || |_ / _ \\| '__/ _` |/ _ \\", "bold green"),
        ("| |  | | (_) | (_| |  __/ ||  _| (_) | | | (_| |  __/", "bold cyan"),
        ("|_|  |_|\\___/ \\__,_|\\___|_||_|  \\___/|_|  \\__, |\\___|", "bold blue"),
        ("                                           |___/", "bold magenta"),
    ]
    banner_text = Text()
    for line, color in banner_lines:
        banner_text.append(line + "\n", style=color)
    console.print(banner_text)


def print_hardware(hardware: HardwareDetector.HardwareSpecs) -> None:
    table = Table(title="Detected Hardware", show_header=True)
    table.add_column("Component", style="bold")
    table.add_column("Details")

    table.add_row("CPU", f"{hardware.physical_core_count} cores")
    table.add_row("RAM", f"{hardware.ram_gb:.1f} GB")
    table.add_row(
        "Disk (free)",
        f"{hardware.available_disk_storage_gb:.1f} / {hardware.disk_storage_gb:.1f} GB",
    )

    for i, gpu in enumerate(hardware.gpus):
        if gpu["total_memory_gb"] == "N/A":
            detail = "Apple MPS — shared"
        else:
            detail = f"{gpu['name']} — {gpu['total_memory_gb']} GB VRAM"
        table.add_row(f"GPU {i}", detail)

    console.print(table)


_PEFT_COLORS = {
    "full_finetune": "bold green",
    "lora_bf16": "bold cyan",
    "qlora_8bit": "bold yellow",
    "qlora_4bit": "bold orange1",
}


def print_recommendation(recommendation: Recommendation) -> None:
    r = recommendation

    # ── Summary strip ──────────────────────────────────────────────────────────
    peft_color = _PEFT_COLORS.get(r.peft_approach, "bold white")
    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary.add_column(style="bold dim", no_wrap=True)
    summary.add_column()
    summary.add_row("Strategy",  r.strategy_slug)
    summary.add_row("PEFT tier", Text(r.peft_approach, style=peft_color))
    summary.add_row("Task",      r.task_slug)
    summary.add_row("Provider",  r.provider_slug)
    console.print(Panel(summary, title="[bold green]Recommendation[/bold green]", expand=False))

    # ── Candidate models ───────────────────────────────────────────────────────
    console.print(Rule("[bold]Candidate Models[/bold]", style="green"))
    models_table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    models_table.add_column("#",          justify="right",  style="dim",       no_wrap=True)
    models_table.add_column("Model ID",                     style="cyan",      no_wrap=True)
    models_table.add_column("Parameters", justify="right",                     no_wrap=True)
    models_table.add_column("Downloads",  justify="right",  style="dim green", no_wrap=True)
    for rank, m in enumerate(r.recommended_models, start=1):
        models_table.add_row(
            str(rank),
            m.model_id,
            f"{m.num_parameters / 1e9:.2f}B",
            f"{m.downloads:,}",
        )
    console.print(models_table)

    # ── Decision rationale ────────────────────────────────────────────────────
    console.print(Rule("[bold]Decision Rationale[/bold]", style="dim"))
    rationale = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    rationale.add_column(style="bold dim", no_wrap=True, width=14)
    rationale.add_column(overflow="fold")
    rationale.add_row("Hardware",     r.rationale.vram_summary)
    rationale.add_row("Use Case",     r.rationale.use_case_resolution)
    rationale.add_row("PEFT",         r.rationale.peft_decision)
    rationale.add_row("Model Filter", r.rationale.model_filter)
    console.print(rationale)

    # ── Warnings (only if present) ────────────────────────────────────────────
    if r.rationale.warnings:
        warn_text = "\n".join(f"[yellow]⚠[/yellow]  {w}" for w in r.rationale.warnings)
        console.print(Panel(warn_text, title="[bold yellow]Warnings[/bold yellow]", border_style="yellow", expand=False))


def print_training_summary(
    model_id: str,
    dataset_desc: str,
    output_dir: str,
    recommendation: Recommendation,
) -> None:
    body = (
        f"Model:          {model_id}\n"
        f"Dataset:        {dataset_desc}\n"
        f"Task:           {recommendation.task_slug}\n"
        f"Strategy:       {recommendation.strategy_slug}\n"
        f"PEFT approach:  {recommendation.peft_approach}\n"
        f"Output Dir:     {output_dir}"
    )
    rprint(Panel(body, title="[bold blue]Training Summary[/bold blue]"))


class RichTrainingCallback(TrainerCallback):

    def __init__(self):
        self.progress = None
        self.task_id = None
        self.current_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.current_step = 0
        self.progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            "•",
            MofNCompleteColumn(),
        )
        self.progress.start()
        self.task_id = self.progress.add_task("Training", total=state.max_steps)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = logs.get("step", self.current_step + 1)
        advance = step - self.current_step
        self.current_step = step
        loss = logs.get("loss", "N/A")
        epoch = logs.get("epoch", "N/A")
        self.progress.update(
            self.task_id,
            advance=advance,
            description=f"Training  loss={loss}  epoch={epoch}",
        )

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress is not None:
            self.progress.stop()
        rprint("[bold green]Training complete![/bold green]")
