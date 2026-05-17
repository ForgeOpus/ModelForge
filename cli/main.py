import click
from rich.console import Console
from rich.table import Table
from utilities.hardware_detector import HardwareDetector
from utilities.recommendation_engine import RecommendationEngine
from cli.display import print_header, print_hardware, print_recommendation, print_training_summary, RichTrainingCallback
from cli.wizard import check_hf_login, select_gpus, select_use_case, select_objective, select_model, select_dataset, validate_and_load_dataset, prompt_output_dir
from cli.runner import assemble_and_train
from utilities.database_manager import FinetuneLedger

console = Console()


@click.group()
def cli():
    pass


@cli.command()
def run():
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
    print_training_summary(model_id, dataset_path, output_dir, recommendation)
    if not click.confirm("Proceed with training?", default=True):
        raise click.Abort()
    output_path = assemble_and_train(recommendation, model_id, dataset, output_dir, cuda_str, callbacks=[RichTrainingCallback()])
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
    console.print("Run [bold]modelforge history[/bold] to view past runs or push to HuggingFace Hub.")


@cli.command()
def recommend():
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


@cli.command(name="list-tasks")
def list_tasks():
    from recommendation_engine.language_use_cases.use_case import LanguageUseCase
    from recommendation_engine import (
        TextGenerationUseCase,
        InstructionFollowingUseCase,
        SummarizationUseCase,
        ExtractiveQAUseCase,
    )

    table = Table(title="Available Use Cases")
    table.add_column("Use Case", style="bold")
    table.add_column("Slug")
    table.add_column("Compatible Objectives")
    table.add_column("Description")

    for use_case_cls in LanguageUseCase.list_use_case_classes():
        objectives = ", ".join(obj.value for obj in use_case_cls.compatible_objectives)
        table.add_row(
            use_case_cls.__name__,
            use_case_cls.slug,
            objectives,
            use_case_cls.description,
        )

    console.print(table)


@cli.command()
def history():
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
