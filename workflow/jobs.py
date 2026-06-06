from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from dataset_manager.dataset_factory import DatasetFactory
from utilities.database_manager import WebRunLedger
from utilities.parallelism_strategy import ParallelismStrategy
from utilities.parallelism_validator import validate_parallelism_compatibility
from utilities.tensorboard import default_log_dir
from workflow.models import RecommendationRequest, TrainingEvent, TrainingRunRequest
from workflow.service import WorkflowService
from workflow.training import assemble_and_train


class CancelledTrainingError(RuntimeError):
    pass


class JobEventCallback(TrainerCallback):
    def __init__(self, manager: "TrainingJobManager", run_id: int) -> None:
        self.manager = manager
        self.run_id = run_id

    def on_log(self, args, state, control, logs=None, **kwargs):
        payload = {
            "step": state.global_step,
            "max_steps": state.max_steps,
            "logs": logs or {},
        }
        self.manager.emit(self.run_id, "log", f"Training step {state.global_step}", payload)

    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self.manager.ledger.record_checkpoint(self.run_id, str(checkpoint_path), state.global_step)
        self.manager.emit(
            self.run_id,
            "checkpoint",
            f"Checkpoint saved at step {state.global_step}",
            {"path": str(checkpoint_path), "step": state.global_step},
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.manager.is_cancel_requested(self.run_id):
            control.should_training_stop = True
            self.manager.emit(self.run_id, "cancel", "Cancellation requested; stopping after current step.")


class TrainingJobManager:
    """In-process v1 job manager for one or more local training runs."""

    def __init__(
        self,
        workflow: WorkflowService | None = None,
        ledger: WebRunLedger | None = None,
        max_active_jobs: int = 1,
        recover_interrupted: bool = True,
    ) -> None:
        self.workflow = workflow or WorkflowService()
        self.ledger = ledger or WebRunLedger()
        self.max_active_jobs = max_active_jobs
        self._threads: dict[int, threading.Thread] = {}
        self._cancel_events: dict[int, threading.Event] = {}
        self._event_queues: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        if recover_interrupted:
            self.ledger.mark_interrupted_runs()

    def start_run(self, request: TrainingRunRequest) -> dict[str, Any]:
        self._assert_capacity()
        recommendation = (
            self.workflow.recommendation_from_dict(request.recommendation)
            if request.recommendation
            else self.workflow.recommend(
                RecommendationRequest(
                    use_case_slug=request.use_case_slug,
                    objective=request.objective,
                    gpu_indices=request.gpu_indices,
                    n=5,
                )
            )
        )
        if request.parallelism_strategy_override is not None:
            recommendation = recommendation.model_copy(
                update={"parallelism_strategy": request.parallelism_strategy_override}
            )
        validate_parallelism_compatibility(recommendation)
        tensorboard_log_dir = request.tensorboard_log_dir if request.tensorboard else None
        if request.tensorboard and tensorboard_log_dir is None:
            tensorboard_log_dir = default_log_dir(request.output_dir)
        run_id = self.ledger.create_run(
            model_id=request.model_id,
            use_case=request.use_case_slug,
            objective=request.objective.value,
            dataset=request.dataset_path,
            output_dir=request.output_dir,
            tensorboard_log_dir=tensorboard_log_dir,
            config_json=recommendation.model_dump_json(),
        )
        self.ledger.update_run(
            run_id,
            task_slug=recommendation.task_slug,
            strategy=recommendation.strategy_slug,
            peft=recommendation.peft_approach,
        )
        with self._lock:
            self._event_queues[run_id] = queue.Queue()
            self._cancel_events[run_id] = threading.Event()
        thread = threading.Thread(
            target=self._run_training,
            args=(run_id, request, recommendation, tensorboard_log_dir),
            daemon=True,
        )
        with self._lock:
            self._threads[run_id] = thread
        thread.start()
        return self.ledger.get_run(run_id) or {"id": run_id}

    def cancel_run(self, run_id: int) -> dict[str, Any]:
        event = self._cancel_events.get(run_id)
        if event is None:
            raise ValueError(f"No active job found for run {run_id}.")
        event.set()
        self.emit(run_id, "cancel", "Cancellation requested.")
        return self.ledger.get_run(run_id) or {"id": run_id}

    def resume_run(self, run_id: int, checkpoint_path: str) -> dict[str, Any]:
        run = self.ledger.get_run(run_id)
        if run is None:
            raise ValueError(f"No run found with id {run_id}.")
        if not self._checkpoint_belongs_to_run(run_id, checkpoint_path):
            raise ValueError(f"Checkpoint is not registered for run {run_id}: {checkpoint_path}")
        request = TrainingRunRequest(
            use_case_slug=run["use_case"],
            objective=run["objective"],
            model_id=run["model_id"],
            dataset_path=run["dataset"],
            output_dir=run["output_dir"],
            tensorboard=bool(run.get("tensorboard_log_dir")),
            tensorboard_log_dir=run.get("tensorboard_log_dir"),
            recommendation=self._load_config_json(run),
            resume_from_checkpoint=checkpoint_path,
        )
        return self.start_run(request)

    def emit(self, run_id: int, event_type: str, message: str = "", payload: dict[str, Any] | None = None) -> None:
        event = TrainingEvent(
            run_id=run_id,
            type=event_type,
            message=message,
            payload=payload or {},
        ).model_dump()
        q = self._event_queues.get(run_id)
        if q is not None:
            q.put(event)

    def get_event_queue(self, run_id: int) -> queue.Queue[dict[str, Any]]:
        with self._lock:
            return self._event_queues.setdefault(run_id, queue.Queue())

    def is_cancel_requested(self, run_id: int) -> bool:
        event = self._cancel_events.get(run_id)
        return event.is_set() if event else False

    def _assert_capacity(self) -> None:
        with self._lock:
            live_threads = [thread for thread in self._threads.values() if thread.is_alive()]
            if len(live_threads) >= self.max_active_jobs:
                raise RuntimeError("A training run is already active. Cancel or wait for it before starting another.")

    def _checkpoint_belongs_to_run(self, run_id: int, checkpoint_path: str) -> bool:
        requested = str(Path(checkpoint_path))
        return any(str(Path(item["path"])) == requested for item in self.ledger.list_checkpoints(run_id))

    def _run_training(self, run_id: int, request: TrainingRunRequest, recommendation, tensorboard_log_dir: str | None) -> None:
        try:
            self.ledger.update_run(run_id, status="running")
            self.emit(run_id, "status", "Training started.")
            hardware = self.workflow.detect_hardware()
            _, cuda_devices = self.workflow.select_gpus(hardware, request.gpu_indices)
            dataset = DatasetFactory.load(request.dataset_path, remote=request.dataset_remote)
            if hasattr(dataset, "values"):
                dataset = next(iter(dataset.values()))
            assemble_and_train(
                recommendation,
                request.model_id,
                dataset,
                request.output_dir,
                cuda_devices,
                callbacks=[JobEventCallback(self, run_id)],
                tensorboard_log_dir=tensorboard_log_dir,
                resume_from_checkpoint=request.resume_from_checkpoint,
            )
            status = "cancelled" if self.is_cancel_requested(run_id) else "completed"
            self.ledger.update_run(run_id, status=status)
            self.emit(run_id, status, f"Training {status}.", {"output_dir": request.output_dir})
        except Exception as exc:
            if self.is_cancel_requested(run_id):
                self.ledger.update_run(run_id, status="cancelled", error=str(exc))
                self.emit(run_id, "cancelled", "Training cancelled.", {"error": str(exc)})
            else:
                self.ledger.update_run(run_id, status="failed", error=str(exc))
                self.emit(run_id, "error", str(exc), {"error": str(exc)})
        finally:
            with self._lock:
                self._threads.pop(run_id, None)
                self._cancel_events.pop(run_id, None)

    @staticmethod
    def _load_config_json(run: dict[str, Any]) -> dict[str, Any] | None:
        import json

        raw = run.get("config_json")
        if not raw:
            return None
        return json.loads(raw)
