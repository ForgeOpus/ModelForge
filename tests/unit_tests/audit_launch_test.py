import json

from utilities.database_manager import WebRunLedger
from utilities.precision import training_precision
from workflow.jobs import TrainingJobManager
import pytest


def test_resume_preserves_launch_selection(tmp_path, monkeypatch):
    ledger = WebRunLedger(tmp_path / "runs.db")
    run_id = ledger.create_run(model_id="org/model", use_case="instruction_following", objective="supervised", dataset="org/data", output_dir="out")
    ledger.update_run(run_id, launch_json=json.dumps({"gpu_indices": [1], "dataset_remote": True}))
    ledger.record_checkpoint(run_id, "out/checkpoint-1", 1)
    manager = TrainingJobManager(workflow=object(), ledger=ledger, recover_interrupted=False)
    monkeypatch.setattr(manager, "start_run", lambda request: request)
    request = manager.resume_run(run_id, "out/checkpoint-1")
    assert request.gpu_indices == [1]
    assert request.dataset_remote is True


def test_older_cuda_uses_fp16(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("utilities.precision.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("utilities.precision.torch.cuda.get_device_capability", lambda index: (7, 5))
    hardware = SimpleNamespace(gpus=[{"total_memory_gb": 16}])
    assert training_precision(hardware) == (False, True)


def test_mismatched_recommendation_rejected_before_run(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from workflow.models import TrainingRunRequest
    workflow = SimpleNamespace(recommendation_from_dict=lambda data: SimpleNamespace(
        task_slug="seq2seq_modeling", strategy_slug="sft", parallelism_strategy="data_parallel"))
    manager = TrainingJobManager(workflow=workflow, ledger=WebRunLedger(tmp_path / "runs.db"), recover_interrupted=False)
    monkeypatch.setattr("workflow.jobs.validate_parallelism_compatibility", lambda recommendation: None)
    monkeypatch.setattr("workflow.jobs.RecommendationEngine._resolve_use_case_and_strategy", lambda *args: (None, None, "causal_language_modeling", "sft", None))
    request = TrainingRunRequest(use_case_slug="instruction_following", model_id="org/model", dataset_path="data", output_dir="out", recommendation={"supplied": True})
    with pytest.raises(ValueError, match="does not match"):
        manager.start_run(request)
    assert manager.ledger.list_runs() == []
