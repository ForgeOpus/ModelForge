from __future__ import annotations

import subprocess

import click
from rich.console import Console
from rich.table import Table

from cli.display import (
    RichTrainingCallback,
    print_hardware,
    print_header,
    print_recommendation,
    print_training_summary,
)
from cli.runner import assemble_and_train
from cli.wizard import (
    check_hf_login,
    prompt_output_dir,
    select_dataset,
    select_gpus,
    select_model,
    select_objective,
    select_use_case,
    validate_and_load_dataset,
)
from utilities.database_manager import FinetuneLedger
from utilities.hardware_detector import HardwareDetector
from utilities.recommendation_engine import RecommendationEngine
from utilities.tensorboard import (
    DEFAULT_TENSORBOARD_HOST,
    DEFAULT_TENSORBOARD_PORT,
    default_log_dir,
    launch_command,
)

console = Console()


def run_terminal_wizard(tensorboard: bool = True, tensorboard_logdir: str | None = None) -> None:
    print_header()
    check_hf_login()
    hardware = HardwareDetector.detect_hardware()
    print_hardware(hardware)
    filtered_hardware, cuda_str = select_gpus(hardware)
    use_case_cls = select_use_case()
    objective = select_objective(use_case_cls)
    console.print("[bold]Finding best models for your hardware...[/bold]")
    recommendation = RecommendationEngine.recommend(use_case_cls.slug, objective, filtered_hardware, n=5)
    print_recommendation(recommendation)
    model_id = select_model(recommendation)
    dataset_path, is_remote = select_dataset()
    dataset = validate_and_load_dataset(dataset_path, is_remote, recommendation)
    output_dir = prompt_output_dir(model_id)
    resolved_tensorboard_logdir = tensorboard_logdir if tensorboard else None
    if tensorboard and resolved_tensorboard_logdir is None:
        resolved_tensorboard_logdir = default_log_dir(output_dir)
    print_training_summary(model_id, dataset_path, output_dir, recommendation, resolved_tensorboard_logdir)
    if not click.confirm("Proceed with training?", default=True):
        raise click.Abort()
    output_path = assemble_and_train(
        recommendation,
        model_id,
        dataset,
        output_dir,
        cuda_str,
        callbacks=[RichTrainingCallback()],
        tensorboard_log_dir=resolved_tensorboard_logdir,
    )
    FinetuneLedger().record(
        model_id=model_id,
        use_case=use_case_cls.slug,
        task_slug=recommendation.task_slug,
        strategy=recommendation.strategy_slug,
        peft=recommendation.peft_approach,
        dataset=dataset_path,
        output_dir=output_path,
        config_json=recommendation.strategy_config.model_dump_json(),
    )
    console.print(f"[green]Model saved to {output_path}[/green]")
    if resolved_tensorboard_logdir:
        console.print(f"TensorBoard logs: [bold]{resolved_tensorboard_logdir}[/bold]")
        console.print(f"Open with: [bold]{launch_command(resolved_tensorboard_logdir)}[/bold]")
    console.print("Run [bold]modelforge history[/bold] to view past runs or push to HuggingFace Hub.")


def recommend_terminal() -> None:
    print_header()
    check_hf_login()
    hardware = HardwareDetector.detect_hardware()
    print_hardware(hardware)
    filtered_hardware, _ = select_gpus(hardware)
    use_case_cls = select_use_case()
    objective = select_objective(use_case_cls)
    console.print("[bold]Finding best models for your hardware...[/bold]")
    recommendation = RecommendationEngine.recommend(use_case_cls.slug, objective, filtered_hardware, n=5)
    print_recommendation(recommendation)


def history_terminal() -> None:
    runs = FinetuneLedger().list_runs()
    if not runs:
        console.print("No finetune runs recorded yet.")
        return

    table = Table(title="Finetune History")
    table.add_column("ID", style="bold")
    table.add_column("Timestamp")
    table.add_column("Model")
    table.add_column("Use Case")
    table.add_column("PEFT")
    table.add_column("Output Dir")

    for run in runs:
        table.add_row(
            str(run["id"]),
            str(run["timestamp"]),
            run["model_id"],
            run["use_case"],
            run["peft"],
            run["output_dir"],
        )

    console.print(table)
    console.print("\nPush a run to HuggingFace Hub?")
    run_id_str = click.prompt("Enter run ID to push (or press Enter to skip)", default="", show_default=False)
    if not run_id_str.strip():
        return

    run = FinetuneLedger().get_run(int(run_id_str))
    if run is None:
        console.print(f"[red]No run found with ID {run_id_str}[/red]")
        return

    repo_name = click.prompt("HuggingFace repo ID (e.g. yourname/model-name)")
    from huggingface_hub import upload_folder
    upload_folder(folder_path=run["output_dir"], repo_id=repo_name, repo_type="model")
    console.print(f"[green]Pushed to https://huggingface.co/{repo_name}[/green]")


def tensorboard_cmd(logdir: str | None, host: str, port: int) -> None:
    resolved_logdir = logdir or "./modelforge_output"
    console.print(f"Starting TensorBoard for [bold]{resolved_logdir}[/bold]")
    console.print(f"URL: [bold]http://{host}:{port}[/bold]")
    try:
        subprocess.run(
            [
                "tensorboard",
                "--logdir",
                resolved_logdir,
                "--host",
                host,
                "--port",
                str(port),
            ],
            check=False,
        )
    except FileNotFoundError:
        console.print("[red]TensorBoard is not installed or is not on PATH.[/red]")
        console.print("Install project dependencies, then run this command again.")
