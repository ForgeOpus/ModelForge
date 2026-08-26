from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_manager.dataset_factory import DatasetFactory
from datasets import DatasetDict, IterableDataset
from finetuning_strategies.strategies.strategy import FinetuningStrategy
from providers.provider_factory import ProviderFactory
from recommendation_engine.language_use_cases.use_case import LanguageUseCase
from utilities.hardware_detector import HardwareDetector
from utilities.recommendation_engine import RecommendationEngine
from utilities.recommendation_engine import OBJECTIVE_STRATEGY_MAP
from utilities.training_objective import TrainingObjective
from workflow.models import DatasetValidationRequest, RecommendationRequest
from workflow.registries import load_core_registries


@dataclass
class DatasetValidationResult:
    valid: bool
    path: str
    remote: bool
    columns: list[str]
    format_description: str | None = None
    error: str | None = None
    # None for streaming datasets, which have no length to report.
    row_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "path": self.path,
            "remote": self.remote,
            "columns": self.columns,
            "format_description": self.format_description,
            "error": self.error,
            "row_count": self.row_count,
        }


class WorkflowService:
    """UI-independent operations shared by web, terminal, and notebook flows."""

    def __init__(self) -> None:
        load_core_registries()

    def detect_hardware(self) -> HardwareDetector.HardwareSpecs:
        return HardwareDetector.detect_hardware()

    def select_gpus(
        self,
        hardware: HardwareDetector.HardwareSpecs,
        gpu_indices: list[int] | None,
    ) -> tuple[HardwareDetector.HardwareSpecs, str]:
        if not gpu_indices:
            return hardware, ",".join(str(i) for i in range(len(hardware.gpus)))
        if any(index < 0 or index >= len(hardware.gpus) for index in gpu_indices):
            raise ValueError(f"GPU indices must be between 0 and {len(hardware.gpus) - 1}.")
        chosen = [hardware.gpus[index] for index in gpu_indices]
        return hardware.model_copy(update={"gpus": chosen}), ",".join(str(i) for i in gpu_indices)

    def list_use_cases(self) -> list[dict[str, Any]]:
        return [
            {
                "name": use_case_cls.name,
                "slug": use_case_cls.slug,
                "description": use_case_cls.description,
                "task_slug": use_case_cls.task_slug,
                "hub_task_filter": use_case_cls.hub_task_filter,
                "compatible_objectives": [objective.value for objective in use_case_cls.compatible_objectives],
                "data_formats": {
                    objective.value: spec.model_dump()
                    for objective, spec in use_case_cls.data_formats.items()
                },
            }
            for use_case_cls in (
                LanguageUseCase.get_use_case_by_slug(slug)
                for slug in LanguageUseCase.get_use_cases()
            )
        ]

    def list_strategies(self) -> list[dict[str, Any]]:
        return [
            {
                "name": cls.name,
                "slug": cls.slug,
                "description": cls.description,
                "supported_providers": cls.supported_providers,
                "supported_tasks": cls.supported_tasks,
                "extra_model_copies": cls.extra_model_copies,
                "batch_multiplier": cls.batch_multiplier,
            }
            for cls in FinetuningStrategy._registry.values()
        ]

    def system_info(self) -> dict[str, Any]:
        return {
            "name": "ModelForge",
            "package": "modelforge-finetuning",
            "version": "4.0.0",
            "providers": ProviderFactory.get_available_providers(),
            "strategies": self.list_strategies(),
            "use_cases": self.list_use_cases(),
        }

    def recommend(self, request: RecommendationRequest):
        hardware = self.detect_hardware()
        filtered_hardware, _ = self.select_gpus(hardware, request.gpu_indices)
        return RecommendationEngine.recommend(
            request.use_case_slug,
            request.objective,
            filtered_hardware,
            n=request.n,
        )

    def validate_dataset(self, request: DatasetValidationRequest) -> DatasetValidationResult:
        format_description = None
        strategy_cls = None
        if request.use_case_slug:
            use_case_cls = LanguageUseCase.get_use_case_by_slug(request.use_case_slug)
            format_description = use_case_cls.data_formats[request.objective].user_facing_description
            strategy_cls = FinetuningStrategy._registry.get(OBJECTIVE_STRATEGY_MAP[request.objective])

        try:
            if not request.remote:
                path = Path(request.path)
                if not path.exists():
                    raise FileNotFoundError(f"Dataset path not found: {request.path}")
            dataset = DatasetFactory.load(request.path, remote=request.remote)
            if isinstance(dataset, IterableDataset):
                return DatasetValidationResult(True, request.path, request.remote, [], format_description)
            split = next(iter(dataset.values())) if isinstance(dataset, DatasetDict) else dataset
            columns = list(getattr(split, "column_names", []) or getattr(split, "features", {}).keys())
            if strategy_cls is not None:
                strategy_cls.validate_data_format(split)
            try:
                row_count = len(split)
            except TypeError:
                row_count = None
            return DatasetValidationResult(
                True, request.path, request.remote, columns, format_description, row_count=row_count
            )
        except Exception as exc:
            return DatasetValidationResult(
                False,
                request.path,
                request.remote,
                [],
                format_description,
                str(exc),
            )

    def validate_model_id(self, model_id: str, task_slug: str | None = None) -> dict[str, Any]:
        """Check a hand-entered model ID before a run is committed to it.

        The old UI accepted any string and only failed once training had
        started; this answers the question up front."""
        model_id = (model_id or "").strip()
        if not model_id:
            return {"valid": False, "model_id": model_id, "num_parameters": None,
                    "error": "Enter a model ID, for example org/model."}
        try:
            from huggingface_hub import HfApi
            from huggingface_hub.utils import HfHubHTTPError

            try:
                info = HfApi().model_info(model_id, expand=["safetensors", "pipeline_tag"])
            except HfHubHTTPError as exc:
                # The Hub answers 401 rather than 404 for an unknown repo when
                # you are not signed in, so both mean the same thing here. The
                # raw error is a wall of Hub internals; say the useful part.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (401, 403, 404):
                    return {
                        "valid": False,
                        "model_id": model_id,
                        "num_parameters": None,
                        "error": "No model with that ID is visible on the Hub. Check the "
                                 "spelling, or run `huggingface-cli login` if it is private "
                                 "or gated.",
                    }
                return {"valid": False, "model_id": model_id, "num_parameters": None,
                        "error": f"The Hub returned an error ({status or 'unknown'})."}

            num_parameters = None
            safetensors = getattr(info, "safetensors", None)
            if safetensors is not None:
                num_parameters = getattr(safetensors, "total", None)

            return {
                "valid": True,
                "model_id": model_id,
                "num_parameters": num_parameters,
                "pipeline_tag": getattr(info, "pipeline_tag", None),
                "error": None,
            }
        except Exception:  # noqa: BLE001
            return {"valid": False, "model_id": model_id, "num_parameters": None,
                    "error": "Could not reach the Hugging Face Hub. Check your connection."}

    @staticmethod
    def recommendation_to_json(recommendation) -> str:
        return recommendation.model_dump_json()

    @staticmethod
    def recommendation_from_dict(data: dict[str, Any]):
        from recommendation_engine.validation_models.recommendation import Recommendation

        return Recommendation.model_validate(data)

    @staticmethod
    def pretty_json(data: Any) -> str:
        return json.dumps(data, indent=2, sort_keys=True, default=str)
