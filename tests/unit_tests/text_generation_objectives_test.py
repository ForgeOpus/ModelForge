from datasets import Dataset
from unittest.mock import patch
import pytest

from recommendation_engine.language_use_cases.use_case import LanguageUseCase
from recommendation_engine.validation_models.recommendation import RecommendedModel
from utilities.hardware_detector import HardwareDetector
from utilities.training_objective import TrainingObjective
from utilities.recommendation_engine import RecommendationEngine
from workflow.models import DatasetValidationRequest
from workflow.service import WorkflowService


def test_text_generation_exposes_preference_and_reinforcement_contracts():
    use_case = LanguageUseCase.get_use_case_by_slug("text_generation")

    assert use_case.compatible_objectives == [
        TrainingObjective.SUPERVISED,
        TrainingObjective.PREFERENCE,
        TrainingObjective.REINFORCEMENT,
    ]
    assert use_case.data_formats[TrainingObjective.PREFERENCE].column_aliases == {
        "preferred_continuation": "chosen",
        "rejected_continuation": "rejected",
    }
    assert use_case.data_formats[TrainingObjective.PREFERENCE].optional_column_aliases == {
        "input_text": "prompt",
    }
    assert use_case.data_formats[TrainingObjective.REINFORCEMENT].column_aliases == {
        "input_text": "prompt",
        "ideal_continuation": "answer",
    }


def test_text_generation_objectives_resolve_to_executable_strategies():
    for objective, strategy_slug in (
        (TrainingObjective.PREFERENCE, "dpo"),
        (TrainingObjective.REINFORCEMENT, "grpo"),
    ):
        _, strategy, task_slug, resolved_slug, _ = RecommendationEngine._resolve_use_case_and_strategy(
            "text_generation", objective
        )
        assert strategy.slug == strategy_slug
        assert resolved_slug == strategy_slug
        assert task_slug == "causal_language_modeling"


def test_shared_dataset_validation_normalizes_text_generation_preference_aliases(monkeypatch):
    service = WorkflowService()
    dataset = Dataset.from_dict({
        "input_text": ["Once upon a time"],
        "preferred_continuation": [" there was a story."],
        "rejected_continuation": [" //"],
    })
    monkeypatch.setattr("workflow.service.DatasetFactory.load", lambda *args, **kwargs: dataset)

    result = service.validate_dataset(DatasetValidationRequest(
        path="unused.jsonl",
        remote=True,
        use_case_slug="text_generation",
        objective=TrainingObjective.PREFERENCE,
    ))

    assert result.valid is True
    assert result.columns == ["prompt", "chosen", "rejected"]


@pytest.mark.parametrize("use_case_slug", ["instruction_following", "text_generation"])
def test_shared_dataset_validation_accepts_implicit_prompt_preference_pairs(monkeypatch, use_case_slug):
    service = WorkflowService()
    dataset = Dataset.from_dict({
        "chosen": ["The preferred continuation."],
        "rejected": ["The rejected continuation."],
    })
    monkeypatch.setattr("workflow.service.DatasetFactory.load", lambda *args, **kwargs: dataset)

    result = service.validate_dataset(DatasetValidationRequest(
        path="unused.jsonl",
        remote=True,
        use_case_slug=use_case_slug,
        objective=TrainingObjective.PREFERENCE,
    ))

    assert result.valid is True
    assert result.columns == ["chosen", "rejected"]


def test_text_generation_preference_recommendation_has_nonzero_gpu_budget():
    hardware = HardwareDetector.HardwareSpecs(
        physical_core_count=8,
        disk_storage_gb=500,
        available_disk_storage_gb=200,
        ram_gb=64,
        gpus=[{"name": "GPU", "total_memory_gb": 24.0}],
    )
    candidate = RecommendedModel(model_id="tiny/model", num_parameters=125_000_000, downloads=1)
    with patch.object(RecommendationEngine, "_query_hub", return_value=([candidate], False, [])):
        recommendation = RecommendationEngine.recommend(
            "text_generation", TrainingObjective.PREFERENCE, hardware
        )

    assert recommendation.strategy_slug == "dpo"
    assert recommendation.vram_estimate.adjusted_vram_gb > 1.0
    assert recommendation.vram_estimate.max_params_by_tier[recommendation.peft_approach] > 0
