import json

from utilities.database_manager import WebRunLedger
from utilities.precision import training_precision
from workflow.jobs import TrainingJobManager


def test_resume_preserves_launch_selection(tmp_path, monkeypatch):
    ledger = WebRunLedger(tmp_path / "runs.db")
    run_id = ledger.create_run(model_id="org/model", use_case="instruction_following", objective="supervised", dataset="org/data", output_dir="out")
    ledger.update_run(run_id, launch_json=json.dumps({"gpu_indices": [1], "dataset_remote": True}))
    ledger.record_checkpoint(run_id, "out/checkpoint-1", 1)
    manager = TrainingJobManager(ledger=ledger, recover_interrupted=False)
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
