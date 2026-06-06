from __future__ import annotations

import time

import pytest

from utilities.database_manager import WebRunLedger
from utilities.training_objective import TrainingObjective
from workflow.jobs import TrainingJobManager
from workflow.models import TrainingRunRequest


class FakeRecommendation:
    task_slug = "causal_language_modeling"
    strategy_slug = "sft"
    peft_approach = "qlora_4bit"

    def model_dump_json(self):
        return '{"fake": true}'


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
