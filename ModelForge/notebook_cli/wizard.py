"""
ModelForge Notebook CLI — interactive wizard entry point.

Run from the terminal with:
    modelforge-nb

Or from inside a notebook cell:
    from ModelForge.notebook_cli import run_cli
    run_cli()
"""
from __future__ import annotations

import threading
import time

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..dependencies import get_hardware_service, get_training_service
from ..schemas.training_schemas import TrainingConfig
from ..utilities.finetuning.quantization import QuantizationFactory
from .progress import ProgressDisplay
from .prompts import (
    console,
    prompt_confirm_config,
    prompt_dataset,
    prompt_hyperparams,
    prompt_model,
    prompt_provider,
    prompt_strategy,
    prompt_task,
)

_BANNER = """
 __  __           _      _  _____
|  \/  | ___   __| | ___| ||  ___|__  _ __ __ _  ___
| |\/| |/ _ \ / _` |/ _ \ || |_ / _ \| '__/ _` |/ _ \\
| |  | | (_) | (_| |  __/ ||  _| (_) | | | (_| |  __/
|_|  |_|\___/ \__,_|\___|_||_|  \___/|_|  \__, |\___|
                                           |___/
"""


class ModelForgeWizard:
    """
    Interactive wizard that walks the user through fine-tuning a model.

    Uses existing Python service classes directly — no REST API calls.
    """

    def __init__(self):
        self.console = Console()

    # ── Public ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full wizard flow."""
        try:
            self._print_banner()
            hardware_info = self._detect_hardware()
            task = self._select_task()
            model_name = self._select_model(task, hardware_info)
            provider = self._select_provider()
            strategy = self._select_strategy()
            dataset_path = self._select_dataset(task, strategy)
            hyperparams = self._configure_hyperparams(hardware_info, strategy)
            config = self._build_config(
                task=task,
                model_name=model_name,
                provider=provider,
                strategy=strategy,
                dataset=dataset_path,
                compute_specs=hardware_info.get("compute_profile", "low_end"),
                hyperparams=hyperparams,
            )
            confirmed = prompt_confirm_config(config)
            if not confirmed:
                self.console.print("\n[yellow]Training cancelled.[/yellow]")
                return
            self._run_training(config)
        except KeyboardInterrupt:
            self.console.print("\n\n[yellow]Interrupted — goodbye.[/yellow]")
        except Exception as exc:
            self.console.print(f"\n[red bold]Error:[/red bold] {exc}")
            raise

    # ── Steps ───────────────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        banner_text = Text(_BANNER, style="bold cyan")
        self.console.print(banner_text)
        self.console.print(
            Panel(
                "[bold]Welcome to ModelForge Notebook CLI[/bold]\n"
                "[dim]Fine-tune HuggingFace models interactively — "
                "no web server required.[/dim]",
                expand=False,
            )
        )

    def _detect_hardware(self) -> dict:
        self.console.print("\n[bold]Step 1/8 — Hardware Detection[/bold]")
        self.console.print("  [dim]Detecting GPU and system specs…[/dim]")
        try:
            hw_service = get_hardware_service()
            specs = hw_service.get_hardware_specs()

            from rich.table import Table

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Key", style="dim")
            table.add_column("Value")
            table.add_row("GPU", str(specs.get("gpu_name", "—")))
            table.add_row("VRAM", f"{specs.get('gpu_memory_gb', 0):.1f} GB")
            table.add_row("RAM", f"{specs.get('ram_gb', 0):.0f} GB")
            table.add_row("Compute profile", specs.get("compute_profile", "—"))
            if specs.get("cuda_version"):
                table.add_row("CUDA", specs["cuda_version"])
            self.console.print(table)
            return specs
        except Exception as exc:
            self.console.print(
                f"  [yellow]⚠ Hardware detection failed ({exc}). "
                "Defaulting to 'low_end' profile.[/yellow]"
            )
            return {"compute_profile": "low_end"}

    def _select_task(self) -> str:
        self.console.print("\n[bold]Step 2/8 — Task Selection[/bold]")
        return prompt_task()

    def _select_model(self, task: str, hardware_info: dict) -> str:
        self.console.print("\n[bold]Step 3/8 — Model Selection[/bold]")
        recommended = "gpt2"
        alternatives: list[str] = []
        try:
            hw_service = get_hardware_service()
            recs = hw_service.get_recommended_models(task)
            recommended = recs.get("recommended_model", recommended)
            alternatives = recs.get("possible_models", [])
        except Exception as exc:
            self.console.print(f"  [dim]Could not fetch recommendations: {exc}[/dim]")
        return prompt_model(recommended, alternatives)

    def _select_provider(self) -> str:
        self.console.print("\n[bold]Step 4/8 — Provider Selection[/bold]")
        return prompt_provider()

    def _select_strategy(self) -> str:
        self.console.print("\n[bold]Step 5/8 — Strategy Selection[/bold]")
        return prompt_strategy()

    def _select_dataset(self, task: str, strategy: str) -> str:
        self.console.print("\n[bold]Step 6/8 — Dataset[/bold]")
        training_service = get_training_service()
        return prompt_dataset(task, strategy, training_service)

    def _configure_hyperparams(self, hardware_info: dict, strategy: str) -> dict:
        self.console.print("\n[bold]Step 7/8 — Hyperparameters[/bold]")
        compute_profile = hardware_info.get("compute_profile", "low_end")
        quant_defaults = QuantizationFactory.get_recommended_config(compute_profile)

        defaults = {
            # from TrainingConfig defaults
            "num_train_epochs": TrainingConfig.model_fields["num_train_epochs"].default,
            "per_device_train_batch_size": TrainingConfig.model_fields["per_device_train_batch_size"].default,
            "gradient_accumulation_steps": TrainingConfig.model_fields["gradient_accumulation_steps"].default,
            "learning_rate": TrainingConfig.model_fields["learning_rate"].default,
            "lora_r": TrainingConfig.model_fields["lora_r"].default,
            "lora_alpha": TrainingConfig.model_fields["lora_alpha"].default,
            "max_seq_length": 2048,
            # quantization defaults from hardware profile
            **quant_defaults,
        }
        return prompt_hyperparams(defaults)

    # ── Config assembly ──────────────────────────────────────────────────────

    @staticmethod
    def _build_config(
        task: str,
        model_name: str,
        provider: str,
        strategy: str,
        dataset: str,
        compute_specs: str,
        hyperparams: dict,
    ) -> dict:
        """Merge all selections into a TrainingConfig-compatible dict."""
        config = {
            "task": task,
            "model_name": model_name,
            "provider": provider,
            "strategy": strategy,
            "dataset": dataset,
            "compute_specs": compute_specs,
            # hyperparameter overrides
            "num_train_epochs": hyperparams.get("num_train_epochs", 1),
            "per_device_train_batch_size": hyperparams.get("per_device_train_batch_size", 1),
            "per_device_eval_batch_size": hyperparams.get("per_device_train_batch_size", 1),
            "gradient_accumulation_steps": hyperparams.get("gradient_accumulation_steps", 4),
            "learning_rate": hyperparams.get("learning_rate", 2e-4),
            "lora_r": hyperparams.get("lora_r", 16),
            "lora_alpha": hyperparams.get("lora_alpha", 32),
            "max_seq_length": hyperparams.get("max_seq_length", 2048),
            "use_4bit": hyperparams.get("use_4bit", True),
            "use_8bit": hyperparams.get("use_8bit", False),
            # TrainingConfig defaults for the rest
            "lora_dropout": TrainingConfig.model_fields["lora_dropout"].default,
            "bnb_4bit_compute_dtype": TrainingConfig.model_fields["bnb_4bit_compute_dtype"].default,
            "bnb_4bit_quant_type": TrainingConfig.model_fields["bnb_4bit_quant_type"].default,
            "use_nested_quant": TrainingConfig.model_fields["use_nested_quant"].default,
            "fp16": TrainingConfig.model_fields["fp16"].default,
            "bf16": TrainingConfig.model_fields["bf16"].default,
            "gradient_checkpointing": TrainingConfig.model_fields["gradient_checkpointing"].default,
            "max_grad_norm": TrainingConfig.model_fields["max_grad_norm"].default,
            "weight_decay": TrainingConfig.model_fields["weight_decay"].default,
            "optim": TrainingConfig.model_fields["optim"].default,
            "lr_scheduler_type": TrainingConfig.model_fields["lr_scheduler_type"].default,
            "max_steps": TrainingConfig.model_fields["max_steps"].default,
            "warmup_ratio": TrainingConfig.model_fields["warmup_ratio"].default,
            "group_by_length": TrainingConfig.model_fields["group_by_length"].default,
            "packing": TrainingConfig.model_fields["packing"].default,
            "eval_split": TrainingConfig.model_fields["eval_split"].default,
            "eval_steps": TrainingConfig.model_fields["eval_steps"].default,
        }
        return config

    # ── Training execution ───────────────────────────────────────────────────

    def _run_training(self, config: dict) -> None:
        """Run training in a background thread, displaying live progress."""
        self.console.print("\n[bold]Step 8/8 — Training[/bold]")

        training_service = get_training_service()
        training_service.reset_training_status()

        progress = ProgressDisplay()
        progress.start("Initialising training…")

        result: dict = {}

        def _train():
            result.update(training_service.train_model(config))

        thread = threading.Thread(target=_train, daemon=True)
        thread.start()

        # Poll status and push updates to the progress display
        while thread.is_alive():
            status = training_service.get_training_status()
            value = status.get("progress", 0)
            message = status.get("message", "")
            progress.update(value, message)
            time.sleep(0.5)

        thread.join()

        # Final status sync
        final_status = training_service.get_training_status()
        progress.update(
            final_status.get("progress", 100),
            final_status.get("message", ""),
        )

        if result.get("success"):
            progress.close()
            self.console.print(
                Panel(
                    f"[bold green]Training complete![/bold green]\n\n"
                    f"[bold]Model ID:[/bold]   {result['model_id']}\n"
                    f"[bold]Model path:[/bold] {result['model_path']}",
                    title="Success",
                    border_style="green",
                    expand=False,
                )
            )
        else:
            progress.error()
            error_msg = result.get("error", result.get("message", "Unknown error"))
            self.console.print(
                Panel(
                    f"[bold red]Training failed[/bold red]\n\n{error_msg}",
                    title="Error",
                    border_style="red",
                    expand=False,
                )
            )


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the `modelforge-nb` console script."""
    ModelForgeWizard().run()


if __name__ == "__main__":
    main()
