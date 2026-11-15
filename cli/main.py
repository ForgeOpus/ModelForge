from __future__ import annotations

import argparse
import sys
import webbrowser

from utilities.database_manager import FinetuneLedger, WebRunLedger
from utilities.hardware_detector import HardwareDetector
from utilities.recommendation_engine import RecommendationEngine
from utilities.tensorboard import DEFAULT_TENSORBOARD_HOST, DEFAULT_TENSORBOARD_PORT
from utilities.training_objective import TrainingObjective
from workflow.registries import load_core_registries
from workflow.service import WorkflowService


def cli(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelforge")
    subcommands = parser.add_subparsers(dest="command")

    run = subcommands.add_parser("run", help="Start the browser UI.")
    run.add_argument("--host", default="0.0.0.0", help="Host to bind. Defaults to 0.0.0.0.")
    run.add_argument("--port", default=8000, type=int, help="Port to bind.")
    run.add_argument("--no-browser", action="store_true", help="Do not open the UI in a browser.")
    run.add_argument("--token", default=None, help="Use a fixed startup token instead of generating one.")
    run.set_defaults(func=run_web)

    terminal = subcommands.add_parser("cli", help="Start the optional terminal wizard.")
    terminal.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    terminal.add_argument("--tensorboard-logdir", default=None, help="Directory for TensorBoard event logs.")
    terminal.set_defaults(func=run_terminal)

    recommend = subcommands.add_parser("recommend", help="Print hardware-aware model recommendations.")
    recommend.add_argument("--use-case", default=None, help="Use case slug.")
    recommend.add_argument("--objective", default="supervised", choices=[item.value for item in TrainingObjective])
    recommend.add_argument("--n", default=5, type=int)
    recommend.set_defaults(func=run_recommend)

    history = subcommands.add_parser("history", help="Print run history.")
    history.add_argument("--legacy", action="store_true", help="Show the legacy terminal finetune ledger.")
    history.set_defaults(func=run_history)

    list_tasks = subcommands.add_parser("list-tasks", help="List available use cases.")
    list_tasks.set_defaults(func=run_list_tasks)

    tensorboard = subcommands.add_parser("tensorboard", help="Launch TensorBoard for a log directory.")
    tensorboard.add_argument("logdir", nargs="?", default=None)
    tensorboard.add_argument("--host", default=DEFAULT_TENSORBOARD_HOST)
    tensorboard.add_argument("--port", default=DEFAULT_TENSORBOARD_PORT, type=int)
    tensorboard.set_defaults(func=run_tensorboard)

    return parser


def run_web(args: argparse.Namespace) -> None:
    import uvicorn
    from web.app import create_app
    from web.auth import TokenAuth

    auth = TokenAuth(args.token)
    url = auth.public_url(args.host, args.port)
    print("ModelForge v4 browser UI")
    print(f"Serving on: http://{args.host}:{args.port}")
    print(f"Open:       {url}")
    print("The startup token is required for API and WebSocket access.")
    if not args.no_browser:
        webbrowser.open(url)
    app = create_app(auth=auth)
    uvicorn.run(app, host=args.host, port=args.port)


def run_terminal(args: argparse.Namespace) -> None:
    try:
        from cli.terminal import run_terminal_wizard
    except ImportError as exc:
        raise SystemExit(
            "The terminal UI requires optional dependencies. "
            "Install with: pip install \"modelforge-finetuning[cli]\""
        ) from exc
    run_terminal_wizard(
        tensorboard=not args.no_tensorboard,
        tensorboard_logdir=args.tensorboard_logdir,
    )


def run_recommend(args: argparse.Namespace) -> None:
    load_core_registries()
    service = WorkflowService()
    use_case = args.use_case or _prompt_use_case(service)
    hardware = HardwareDetector.detect_hardware()
    recommendation = RecommendationEngine.recommend(
        use_case,
        TrainingObjective(args.objective),
        hardware,
        n=args.n,
    )
    print(f"Task:      {recommendation.task_slug}")
    print(f"Strategy:  {recommendation.strategy_slug}")
    print(f"PEFT:      {recommendation.peft_approach}")
    print("Models:")
    for model in recommendation.recommended_models:
        print(f"  - {model.model_id} ({model.num_parameters:,} params, {model.downloads:,} downloads)")
    if recommendation.rationale.warnings:
        print("Warnings:")
        for warning in recommendation.rationale.warnings:
            print(f"  - {warning}")


def run_history(args: argparse.Namespace) -> None:
    if args.legacy:
        runs = FinetuneLedger().list_runs()
        for run in runs:
            print(f"{run['id']}: {run['timestamp']} {run['model_id']} ({run['peft']}) -> {run['output_dir']}")
        if not runs:
            print("No legacy finetune runs recorded yet.")
        return
    runs = WebRunLedger().list_runs()
    for run in runs:
        print(f"{run['id']}: {run['status']} {run['model_id']} -> {run['output_dir']}")
    if not runs:
        print("No runs recorded yet.")


def run_list_tasks(args: argparse.Namespace) -> None:
    service = WorkflowService()
    for use_case in service.list_use_cases():
        objectives = ", ".join(use_case["compatible_objectives"])
        print(f"{use_case['slug']}: {use_case['name']} [{objectives}]")
        print(f"  {use_case['description']}")


def run_tensorboard(args: argparse.Namespace) -> None:
    try:
        from cli.terminal import tensorboard_cmd
    except ImportError as exc:
        raise SystemExit(
            "The TensorBoard terminal helper requires optional dependencies. "
            "Install with: pip install \"modelforge-finetuning[cli]\""
        ) from exc
    tensorboard_cmd(args.logdir, args.host, args.port)


def _prompt_use_case(service: WorkflowService) -> str:
    use_cases = service.list_use_cases()
    print("Available use cases:")
    for index, use_case in enumerate(use_cases, start=1):
        print(f"  {index}. {use_case['name']} ({use_case['slug']})")
    while True:
        raw = input("Select use case: ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(use_cases):
                return use_cases[choice - 1]["slug"]
        except ValueError:
            if any(use_case["slug"] == raw for use_case in use_cases):
                return raw
        print("Enter a number from the list or a use case slug.")


if __name__ == "__main__":
    cli(sys.argv[1:])
