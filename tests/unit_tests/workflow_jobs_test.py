from __future__ import annotations

import time

import pytest

from utilities.database_manager import WebRunLedger
from utilities.parallelism_strategy import ParallelismStrategy
from utilities.training_objective import TrainingObjective
from workflow.jobs import TrainingJobManager
from workflow.models import TrainingRunRequest


class FakeRecommendation:
    task_slug = "causal_language_modeling"
    strategy_slug = "sft"
    peft_approach = "qlora_4bit"
    parallelism_strategy = ParallelismStrategy.DATA_PARALLEL

    def model_dump_json(self):
        return '{"fake": true}'

    def model_copy(self, *, update=None):
        """Minimal Pydantic-compatible model_copy that applies the update dict."""
        import copy
        clone = copy.copy(self)
        for k, v in (update or {}).items():
            object.__setattr__(clone, k, v)
        return clone


class FakeWorkflow:
    def recommend(self, request):
        return FakeRecommendation()

    def detect_hardware(self):
        class Hardware:
            gpus = []

            def model_copy(self, update):
                clone = Hardware()
                clone.gpus = update.get("gpus", self.gpus)
                return clone

        return Hardware()

    def select_gpus(self, hardware, gpu_indices):
        return hardware, ""


def wait_for(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition was not met before timeout")


def make_request():
    return TrainingRunRequest(
        use_case_slug="instruction_following",
        objective=TrainingObjective.SUPERVISED,
        model_id="org/model",
        dataset_path="dataset.jsonl",
        output_dir="./out",
        tensorboard=False,
    )


def test_job_manager_records_completed_run(tmp_path, monkeypatch):
    ledger = WebRunLedger(tmp_path / "runs.db")
    manager = TrainingJobManager(workflow=FakeWorkflow(), ledger=ledger, recover_interrupted=False)
    monkeypatch.setattr("workflow.jobs.DatasetFactory.load", lambda *args, **kwargs: object())
    monkeypatch.setattr("workflow.jobs.assemble_and_train", lambda *args, **kwargs: "./out")

    run = manager.start_run(make_request())

    wait_for(lambda: ledger.get_run(run["id"])["status"] == "completed")
    saved = ledger.get_run(run["id"])
    assert saved["model_id"] == "org/model"
    assert saved["strategy"] == "sft"


def test_job_manager_enforces_single_active_run(tmp_path, monkeypatch):
    ledger = WebRunLedger(tmp_path / "runs.db")
    manager = TrainingJobManager(workflow=FakeWorkflow(), ledger=ledger, recover_interrupted=False)
    monkeypatch.setattr("workflow.jobs.DatasetFactory.load", lambda *args, **kwargs: object())

    def slow_train(*args, **kwargs):
        time.sleep(0.3)
        return "./out"

    monkeypatch.setattr("workflow.jobs.assemble_and_train", slow_train)
    manager.start_run(make_request())

    with pytest.raises(RuntimeError, match="already active"):
        manager.start_run(make_request())


def test_resume_requires_registered_checkpoint(tmp_path):
    ledger = WebRunLedger(tmp_path / "runs.db")
    manager = TrainingJobManager(workflow=FakeWorkflow(), ledger=ledger, recover_interrupted=False)
    run_id = ledger.create_run(
        model_id="org/model",
        use_case="instruction_following",
        objective="supervised",
        dataset="dataset.jsonl",
        output_dir="./out",
    )

    with pytest.raises(ValueError, match="not registered"):
        manager.resume_run(run_id, "./out/checkpoint-1")


def test_ledger_marks_interrupted_runs(tmp_path):
    ledger = WebRunLedger(tmp_path / "runs.db")
    run_id = ledger.create_run(
        model_id="org/model",
        use_case="instruction_following",
        objective="supervised",
        dataset="dataset.jsonl",
        output_dir="./out",
    )

    changed = ledger.mark_interrupted_runs()

    assert changed == 1
    assert ledger.get_run(run_id)["status"] == "failed"


# ---------------------------------------------------------------------------
# Parallelism override tests
# ---------------------------------------------------------------------------

def test_start_run_applies_parallelism_override(tmp_path, monkeypatch):
    """When parallelism_strategy_override is set, the recommendation passed to
    assemble_and_train must reflect the overridden strategy."""
    ledger = WebRunLedger(tmp_path / "runs.db")
    manager = TrainingJobManager(workflow=FakeWorkflow(), ledger=ledger, recover_interrupted=False)
    monkeypatch.setattr("workflow.jobs.DatasetFactory.load", lambda *args, **kwargs: object())

    captured = {}

    def capture_train(recommendation, *args, **kwargs):
        captured["parallelism_strategy"] = recommendation.parallelism_strategy
        return "./out"

    monkeypatch.setattr("workflow.jobs.assemble_and_train", capture_train)

    request = make_request()
    request = request.model_copy(update={"parallelism_strategy_override": ParallelismStrategy.DATA_PARALLEL})
    run = manager.start_run(request)

    wait_for(lambda: ledger.get_run(run["id"])["status"] == "completed")
    assert captured["parallelism_strategy"] == ParallelismStrategy.DATA_PARALLEL


def test_start_run_fsdp_lora_override_raises(tmp_path, monkeypatch):
    """Overriding parallelism_strategy to FSDP when peft_approach is lora_bf16
    must raise ValueError before any training thread is spawned."""
    ledger = WebRunLedger(tmp_path / "runs.db")

    class LoraWorkflow(FakeWorkflow):
        def recommend(self, request):
            rec = FakeRecommendation()
            rec.peft_approach = "lora_bf16"
            return rec

    manager = TrainingJobManager(workflow=LoraWorkflow(), ledger=ledger, recover_interrupted=False)
    monkeypatch.setattr("workflow.jobs.DatasetFactory.load", lambda *args, **kwargs: object())

    spawned = {"count": 0}
    original_thread_start = __import__("threading").Thread.start

    def counting_start(self):
        spawned["count"] += 1
        original_thread_start(self)

    monkeypatch.setattr("threading.Thread.start", counting_start)

    request = make_request()
    request = request.model_copy(update={"parallelism_strategy_override": ParallelismStrategy.FSDP})

    with pytest.raises(ValueError):
        manager.start_run(request)

    assert spawned["count"] == 0, "No training thread should be spawned when validation fails"


def test_tensorboard_logs_are_isolated_for_each_launch_and_resume(tmp_path, monkeypatch):
    from pathlib import Path
    from utilities.tensorboard import viewer_log_dir
    ledger = WebRunLedger(tmp_path / 'runs.db')
    manager = TrainingJobManager(workflow=FakeWorkflow(), ledger=ledger, recover_interrupted=False)
    monkeypatch.setattr('workflow.jobs.threading.Thread.start', lambda self: None)
    request = make_request().model_copy(update={'tensorboard': True, 'output_dir': str(tmp_path / 'out')})
    first = manager.start_run(request)
    manager._threads.clear()
    second = manager.start_run(request.model_copy(update={'tensorboard_log_dir': first['tensorboard_log_dir']}))
    assert Path(first['tensorboard_log_dir']).name == f"run-{first['id']}"
    assert Path(second['tensorboard_log_dir']).name == f"run-{second['id']}"
    assert first['tensorboard_log_dir'] != second['tensorboard_log_dir']
    assert viewer_log_dir(first['tensorboard_log_dir']) == viewer_log_dir(second['tensorboard_log_dir']) == str(tmp_path / 'out' / 'tensorboard')
