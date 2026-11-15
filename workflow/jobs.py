from __future__ import annotations

import json
import queue
import threading
from collections import deque
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from dataset_manager.dataset_factory import DatasetFactory
from utilities.database_manager import WebRunLedger
from utilities.parallelism_strategy import ParallelismStrategy
from utilities.parallelism_validator import validate_parallelism_compatibility
from utilities.recommendation_engine import RecommendationEngine
from utilities.tensorboard import default_log_dir, isolated_log_dir
from workflow.models import RecommendationRequest, TrainingEvent, TrainingRunRequest
from workflow.service import WorkflowService
from workflow.training import assemble_and_train


_TASK_SLUG_TO_AUTO_CLASS = {
    "question_answering": "AutoModelForQuestionAnswering",
    "causal_language_modeling": "AutoModelForCausalLM",
    "seq2seq_modeling": "AutoModelForSeq2SeqLM",
    "token_classification": "AutoModelForTokenClassification",
}

_TASK_SLUG_TO_PIPELINE = {
    "question_answering": "question-answering",
    "causal_language_modeling": "text-generation",
    "seq2seq_modeling": "summarization",
    "token_classification": "token-classification",
}


def _directory_size(path: Path) -> int | None:
    """Total bytes on disk, or None if the tree cannot be walked."""
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return None


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
        max_events_per_run: int = 5000,
        max_history_runs: int = 20,
    ) -> None:
        self.workflow = workflow or WorkflowService()
        self.ledger = ledger or WebRunLedger()
        self.max_active_jobs = max_active_jobs
        self.max_events_per_run = max_events_per_run
        self.max_history_runs = max_history_runs
        self._threads: dict[int, threading.Thread] = {}
        self._cancel_events: dict[int, threading.Event] = {}
        # One list of live subscriber queues per run, so several browser tabs can
        # watch the same run instead of stealing each other's events.
        self._subscribers: dict[int, list[queue.Queue[dict[str, Any]]]] = {}
        # Bounded replay buffer per run. Without this, reloading the page mid-run
        # leaves the live view permanently blank -- the queue is drained
        # destructively and nothing that already happened can be recovered.
        self._event_history: dict[int, deque[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        # Serializes admission of a new training job with playground inference.
        self.resource_gate = threading.Lock()
        if recover_interrupted:
            self.ledger.mark_interrupted_runs()

    def start_run(self, request: TrainingRunRequest) -> dict[str, Any]:
        if not self.resource_gate.acquire(blocking=False):
            raise RuntimeError("Playground inference is using model resources. Wait for it to finish before starting training.")
        try:
            return self._start_run(request)
        finally:
            self.resource_gate.release()

    def _start_run(self, request: TrainingRunRequest) -> dict[str, Any]:
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
                    model_id=request.model_id,
                )
            )
        )
        if request.parallelism_strategy_override is not None:
            recommendation = recommendation.model_copy(
                update={"parallelism_strategy": request.parallelism_strategy_override}
            )
        validate_parallelism_compatibility(recommendation)
        if request.recommendation:
            _, strategy_cls, task_slug, strategy_slug, _ = RecommendationEngine._resolve_use_case_and_strategy(
                request.use_case_slug, request.objective
            )
            if recommendation.task_slug != task_slug or recommendation.strategy_slug != strategy_slug:
                raise ValueError("Recommendation does not match the selected use case and objective.")
            if recommendation.provider_slug not in strategy_cls.supported_providers:
                raise ValueError("Recommendation provider is not supported by this strategy.")
            if recommendation.peft_approach not in {"full_finetune", "lora_bf16", "qlora_8bit", "qlora_4bit"}:
                raise ValueError("Unknown fine-tuning approach.")
            candidates = recommendation.recommended_models
            if not candidates or request.model_id != candidates[0].model_id:
                chosen = next((model for model in candidates if model.model_id == request.model_id), None)
                if chosen is None:
                    from recommendation_engine.validation_models.recommendation import RecommendedModel
                    result = self.workflow.validate_model_id(request.model_id, task_slug)
                    if not result["valid"] or not result.get("num_parameters"):
                        raise ValueError("Cannot safely configure this model without valid parameter metadata.")
                    chosen = RecommendedModel(model_id=request.model_id, num_parameters=result["num_parameters"], downloads=0)
                hardware, _ = self.workflow.select_gpus(self.workflow.detect_hardware(), request.gpu_indices)
                recommendation = RecommendationEngine.recommend(
                    request.use_case_slug, request.objective, hardware, selected_model=chosen,
                )
                if request.parallelism_strategy_override is not None:
                    recommendation = recommendation.model_copy(update={"parallelism_strategy": request.parallelism_strategy_override})
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
        if tensorboard_log_dir:
            tensorboard_log_dir = isolated_log_dir(tensorboard_log_dir, run_id)
        self.ledger.update_run(
            run_id,
            tensorboard_log_dir=tensorboard_log_dir,
            launch_json=json.dumps({"gpu_indices": request.gpu_indices, "dataset_remote": request.dataset_remote}),
            task_slug=recommendation.task_slug,
            strategy=recommendation.strategy_slug,
            peft=recommendation.peft_approach,
        )
        with self._lock:
            self._event_history[run_id] = deque(maxlen=self.max_events_per_run)
            self._subscribers.setdefault(run_id, [])
            self._cancel_events[run_id] = threading.Event()
            self._evict_stale_history()
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
        launch = json.loads(run.get("launch_json") or "{}")
        request = TrainingRunRequest(
            use_case_slug=run["use_case"],
            objective=run["objective"],
            model_id=run["model_id"],
            dataset_path=run["dataset"],
            dataset_remote=launch.get("dataset_remote", False),
            gpu_indices=launch.get("gpu_indices", []),
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
        with self._lock:
            history = self._event_history.setdefault(run_id, deque(maxlen=self.max_events_per_run))
            history.append(event)
            listeners = list(self._subscribers.get(run_id, ()))
        for listener in listeners:
            listener.put(event)

    def subscribe(self, run_id: int) -> queue.Queue[dict[str, Any]]:
        """Register a new listener for a run. Every subscriber sees every event."""
        listener: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(run_id, []).append(listener)
        return listener

    def unsubscribe(self, run_id: int, listener: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            listeners = self._subscribers.get(run_id)
            if not listeners:
                return
            try:
                listeners.remove(listener)
            except ValueError:
                pass

    def get_event_history(self, run_id: int) -> list[dict[str, Any]]:
        """Everything emitted for this run, oldest first.

        Falls back to the archive written when the run finished, so reopening
        an old run after a restart still shows its loss curve and timeline."""
        with self._lock:
            live = list(self._event_history.get(run_id, ()))
        if live:
            return live
        raw = self.ledger.load_run_events(run_id)
        if not raw:
            return []
        try:
            return json.loads(raw)
        except ValueError:
            return []

    def _archive_events(self, run_id: int) -> None:
        """Persist the timeline once, at the end of a run."""
        with self._lock:
            events = list(self._event_history.get(run_id, ()))
        if not events:
            return
        try:
            self.ledger.save_run_events(run_id, json.dumps(events))
        except Exception:  # noqa: BLE001 - archiving must never fail a run
            pass

    def get_event_queue(self, run_id: int) -> queue.Queue[dict[str, Any]]:
        """Kept for callers predating subscribe(); returns an independent listener."""
        return self.subscribe(run_id)

    def _evict_stale_history(self) -> None:
        """Bound total memory: keep replay buffers for the most recent runs only.
        Caller must hold the lock."""
        if len(self._event_history) <= self.max_history_runs:
            return
        for stale_id in sorted(self._event_history)[: -self.max_history_runs]:
            if self._subscribers.get(stale_id):
                continue
            self._event_history.pop(stale_id, None)
            self._subscribers.pop(stale_id, None)

    def is_cancel_requested(self, run_id: int) -> bool:
        event = self._cancel_events.get(run_id)
        return event.is_set() if event else False

    def _assert_capacity(self) -> None:
        with self._lock:
            live_threads = [thread for thread in self._threads.values() if thread.is_alive()]
            if len(live_threads) >= self.max_active_jobs:
                raise RuntimeError("A training run is already active. Cancel or wait for it before starting another.")

    def has_active_jobs(self) -> bool:
        with self._lock:
            return any(thread.is_alive() for thread in self._threads.values())

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
            if status == "completed":
                self._register_artifact(run_id, request, recommendation)
            self.emit(run_id, status, f"Training {status}.", {"output_dir": request.output_dir})
            self._archive_events(run_id)
        except Exception as exc:
            if self.is_cancel_requested(run_id):
                self.ledger.update_run(run_id, status="cancelled", error=str(exc))
                self.emit(run_id, "cancelled", "Training cancelled.", {"error": str(exc)})
                self._archive_events(run_id)
            else:
                self.ledger.update_run(run_id, status="failed", error=str(exc))
                self.emit(run_id, "error", str(exc), {"error": str(exc)})
                self._archive_events(run_id)
        finally:
            with self._lock:
                self._threads.pop(run_id, None)
                self._cancel_events.pop(run_id, None)

    def _register_artifact(self, run_id: int, request: TrainingRunRequest, recommendation) -> None:
        """Record the finished model and describe it on disk.

        modelforge_config.json is what lets the playground reload an adapter
        later: a PEFT checkpoint records its base model, but not which
        AutoModel class or pipeline task it was trained for."""
        output_dir = Path(request.output_dir)
        if not output_dir.is_dir():
            # Training just finished, so the directory should exist. If it does
            # not, something is wrong upstream -- don't paper over it by
            # creating one.
            self.emit(run_id, "log", f"Output directory not found, skipping registration: {output_dir}")
            return
        try:
            (output_dir / "modelforge_config.json").write_text(
                json.dumps(
                    {
                        "base_model": request.model_id,
                        "task_slug": recommendation.task_slug,
                        "use_case": request.use_case_slug,
                        "objective": str(getattr(request.objective, "value", request.objective)),
                        "strategy": recommendation.strategy_slug,
                        "peft_approach": recommendation.peft_approach,
                        "model_class": _TASK_SLUG_TO_AUTO_CLASS.get(recommendation.task_slug, "AutoModelForCausalLM"),
                        "pipeline_task": _TASK_SLUG_TO_PIPELINE.get(recommendation.task_slug, "text-generation"),
                        "modelforge_version": "4.0.0",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            # Losing the descriptor should not fail an otherwise good run.
            self.emit(run_id, "log", f"Could not write modelforge_config.json: {exc}")

        try:
            self.ledger.record_model(
                run_id=run_id,
                name=f"{request.model_id.split('/')[-1]}-{run_id}",
                base_model=request.model_id,
                model_path=str(output_dir),
                task_slug=recommendation.task_slug,
                use_case=request.use_case_slug,
                peft_approach=recommendation.peft_approach,
                size_bytes=_directory_size(output_dir),
            )
        except Exception as exc:
            self.emit(run_id, "log", f"Could not register the trained model: {exc}")

    @staticmethod
    def _load_config_json(run: dict[str, Any]) -> dict[str, Any] | None:
        import json

        raw = run.get("config_json")
        if not raw:
            return None
        return json.loads(raw)
