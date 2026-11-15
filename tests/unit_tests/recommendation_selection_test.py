from types import SimpleNamespace
from unittest.mock import patch

import pytest

from utilities.hardware_detector import HardwareDetector
from utilities.recommendation_engine import RecommendationEngine
from utilities.training_objective import TrainingObjective
from recommendation_engine.validation_models.recommendation import RecommendedModel


def hardware(vram):
    return HardwareDetector.HardwareSpecs(
        physical_core_count=8, ram_gb=64, disk_storage_gb=500,
        available_disk_storage_gb=200,
        gpus=[{"name": "GPU", "total_memory_gb": size} for size in vram],
    )


@pytest.mark.parametrize("size,tier", [(8e9, "qlora_4bit"), (3e9, "qlora_8bit"), (1.5e9, "lora_bf16"), (135e6, "full_finetune")])
def test_dpo_selects_model_before_training_method(size, tier):
    models = [RecommendedModel(model_id="org/small-instruct", num_parameters=100_000_000, downloads=1000),
              RecommendedModel(model_id="org/larger-instruct", num_parameters=int(size), downloads=500)]
    with patch.object(RecommendationEngine, "_query_hub", return_value=(models, False, [])) as query:
        result = RecommendationEngine.recommend("instruction_following", TrainingObjective.PREFERENCE, hardware([16]))
    assert result.strategy_slug == "dpo"
    assert result.peft_approach == tier
    assert result.recommended_models[0].num_parameters == size
    assert query.call_args.kwargs["max_params"] > 8e9
    assert query.call_args.kwargs["instruction_only"] is True
    assert (result.peft_config is None) == (tier == "full_finetune")


def test_adapter_models_must_fit_smallest_gpu_and_all_listed_models_fit_selected_tier():
    models = [RecommendedModel(model_id=f"org/{size}", num_parameters=size, downloads=100)
              for size in (8_000_000_000, 1_500_000_000)]
    with patch.object(RecommendationEngine, "_query_hub", return_value=(models, False, [])):
        result = RecommendationEngine.recommend("instruction_following", TrainingObjective.PREFERENCE, hardware([16, 8]))
    assert result.recommended_models[0].num_parameters == 1_500_000_000
    assert all(m.num_parameters <= result.vram_estimate.max_params_by_tier[result.peft_approach]
               for m in result.recommended_models)


def test_instruction_candidates_include_chat_metadata_and_exclude_plain_base_models():
    candidates = [SimpleNamespace(id=name, safetensors=SimpleNamespace(total=1_000_000_000), downloads=100, tags=tags)
                  for name, tags in [("org/base", []), ("org/model-instruct", []), ("org/model3", ["conversational"])]]
    with patch("utilities.recommendation_engine.HfApi") as api:
        api.return_value.list_models.return_value = candidates
        results, _, _ = RecommendationEngine._query_hub("text-generation", 2e9, 5, 2e9, instruction_only=True)
    assert [m.model_id for m in results] == ["org/model-instruct", "org/model3"]


def test_over_budget_candidates_do_not_produce_an_unusable_configuration():
    model = RecommendedModel(model_id="org/huge-instruct", num_parameters=100_000_000_000, downloads=100)
    with patch.object(RecommendationEngine, "_query_hub", return_value=([model], False, [])):
        result = RecommendationEngine.recommend("instruction_following", TrainingObjective.PREFERENCE, hardware([16]))
    assert result.recommended_models == []
    assert result.strategy_config is None

@pytest.mark.parametrize('size,tier,lr,quantization', [
    (8_190_735_360, 'qlora_4bit', 2e-7, {'quantization_config': {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True}}),
    (4_022_468_096, 'qlora_8bit', 5e-7, {'quantization_config': {'load_in_8bit': True}}),
    (1_543_714_304, 'lora_bf16', 5e-7, {}),
    (751_632_384, 'lora_bf16', 5e-6, {}),
])
def test_selected_model_preview_uses_its_own_precision_and_learning_rate(monkeypatch, size, tier, lr, quantization):
    from workflow.service import WorkflowService
    from workflow.models import RecommendationRequest
    service = WorkflowService()
    monkeypatch.setattr(service, 'detect_hardware', lambda: hardware([16]))
    monkeypatch.setattr(service, 'validate_model_id', lambda *args: {'valid': True, 'num_parameters': size})
    with patch.object(RecommendationEngine, '_query_hub', side_effect=AssertionError('An explicit model must not run another shortlist search')):
        result = service.recommend(RecommendationRequest(use_case_slug='instruction_following', objective='preference', model_id='org/selected'))
    assert result.recommended_models[0].model_id == 'org/selected'
    assert result.peft_approach == tier
    assert result.strategy_config.model_init_kwargs == quantization
    assert result.strategy_config.learning_rate == lr


def test_run_rebuilds_stale_large_model_plan_for_smaller_selection(tmp_path, monkeypatch):
    import json
    from workflow.service import WorkflowService
    from workflow.jobs import TrainingJobManager
    from workflow.models import TrainingRunRequest
    from utilities.database_manager import WebRunLedger
    big = RecommendedModel(model_id='org/big', num_parameters=8_190_735_360, downloads=100)
    small = RecommendedModel(model_id='org/small', num_parameters=751_632_384, downloads=100)
    with patch.object(RecommendationEngine, '_query_hub', return_value=([big, small], False, [])):
        original = RecommendationEngine.recommend('instruction_following', TrainingObjective.PREFERENCE, hardware([16]))
    service = WorkflowService()
    monkeypatch.setattr(service, 'detect_hardware', lambda: hardware([16]))
    monkeypatch.setattr('workflow.jobs.threading.Thread.start', lambda self: None)
    ledger = WebRunLedger(tmp_path / 'runs.db')
    manager = TrainingJobManager(workflow=service, ledger=ledger, recover_interrupted=False)
    run = manager.start_run(TrainingRunRequest(use_case_slug='instruction_following', objective='preference', model_id=small.model_id,
        dataset_path='unused', output_dir=str(tmp_path / 'model'), tensorboard=False, recommendation=original.model_dump()))
    saved = ledger.get_run(run['id'])
    config = json.loads(saved['config_json'])
    assert saved['peft'] == config['peft_approach'] == 'lora_bf16'
    assert config['strategy_config']['model_init_kwargs'] == {}
    assert config['strategy_config']['learning_rate'] == 5e-6
    assert config['recommended_models'][0]['model_id'] == small.model_id
