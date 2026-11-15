from __future__ import annotations

from utilities.training_objective import TrainingObjective
from workflow.jobs import TrainingJobManager
from workflow.models import DatasetValidationRequest, RecommendationRequest, TrainingRunRequest
from workflow.service import WorkflowService


def main() -> None:
    service = None
    jobs = None
    print("ModelForge Notebook UI")
    print("Pure Python prompts; no widgets or terminal menus.")
    while True:
        print("\n1. Fine-tune a model")
        print("2. List runs")
        print("3. Exit")
        choice = input("Choose an option: ").strip()
        if choice == "1":
            service, jobs = _ensure_runtime(service, jobs)
            _run_training_flow(service, jobs)
        elif choice == "2":
            service, jobs = _ensure_runtime(service, jobs)
            for run in jobs.ledger.list_runs():
                print(f"{run['id']}: {run['status']} {run['model_id']} -> {run['output_dir']}")
        elif choice == "3":
            if jobs is not None and jobs.has_active_jobs():
                print("A training run is still active. Wait for it to finish before exiting.")
                continue
            return
        else:
            print("Please enter 1, 2, or 3.")


def _ensure_runtime(service, jobs):
    if service is None:
        service = WorkflowService()
    if jobs is None:
        jobs = TrainingJobManager(workflow=service)
    return service, jobs


def _run_training_flow(service: WorkflowService, jobs: TrainingJobManager) -> None:
    use_cases = service.list_use_cases()
    print("\nUse cases")
    for index, use_case in enumerate(use_cases, start=1):
        print(f"{index}. {use_case['name']} ({use_case['slug']})")
    use_case_slug = use_cases[_prompt_index("Select use case: ", len(use_cases))]["slug"]
    objective = _prompt_objective(use_cases, use_case_slug)
    recommendation = service.recommend(
        RecommendationRequest(use_case_slug=use_case_slug, objective=objective, n=5)
    )
    print("\nRecommended models")
    for index, model in enumerate(recommendation.recommended_models, start=1):
        print(f"{index}. {model.model_id} ({model.num_parameters:,} params)")
    custom_index = len(recommendation.recommended_models) + 1
    print(f"{custom_index}. Enter a custom model id")
    model_choice = _prompt_index("Select model: ", custom_index)
    if model_choice == custom_index - 1:
        model_id = input("Model id: ").strip()
    else:
        model_id = recommendation.recommended_models[model_choice].model_id
    dataset_path = input("Dataset path or HuggingFace dataset id: ").strip()
    remote = input("Is this a HuggingFace dataset id? [y/N]: ").strip().lower() in {"y", "yes"}
    validation = service.validate_dataset(
        DatasetValidationRequest(
            path=dataset_path,
            remote=remote,
            use_case_slug=use_case_slug,
            objective=objective,
        )
    )
    if not validation.valid:
        print(f"Dataset invalid: {validation.error}")
        if validation.format_description:
            print(validation.format_description)
        return
    output_dir = input("Output directory: ").strip() or "./modelforge_output"
    run = jobs.start_run(
        TrainingRunRequest(
            use_case_slug=use_case_slug,
            objective=objective,
            model_id=model_id,
            dataset_path=dataset_path,
            dataset_remote=remote,
            output_dir=output_dir,
            recommendation=recommendation.model_dump(),
        )
    )
    print(f"Started run {run['id']}. Use the browser UI or history command to monitor details.")


def _prompt_objective(use_cases: list[dict], use_case_slug: str) -> TrainingObjective:
    use_case = next(item for item in use_cases if item["slug"] == use_case_slug)
    objectives = use_case["compatible_objectives"]
    if len(objectives) == 1:
        return TrainingObjective(objectives[0])
    for index, objective in enumerate(objectives, start=1):
        print(f"{index}. {objective}")
    return TrainingObjective(objectives[_prompt_index("Select objective: ", len(objectives))])


def _prompt_index(prompt: str, count: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
            if 1 <= value <= count:
                return value - 1
        except ValueError:
            pass
        print(f"Enter a number between 1 and {count}.")


if __name__ == "__main__":
    main()
