from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from utilities.training_objective import TrainingObjective
from utilities.parallelism_strategy import ParallelismStrategy


RunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class GpuSelection(BaseModel):
    indices: list[int] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    model_id: str | None = None
    use_case_slug: str
    objective: TrainingObjective = TrainingObjective.SUPERVISED
    gpu_indices: list[int] | None = None
    n: int = 5

    @field_validator("n")
    @classmethod
    def validate_n(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n must be positive")
        return value


class DatasetValidationRequest(BaseModel):
    path: str
    remote: bool = False
    use_case_slug: str | None = None
    objective: TrainingObjective = TrainingObjective.SUPERVISED


class TrainingRunRequest(BaseModel):
    use_case_slug: str
    objective: TrainingObjective = TrainingObjective.SUPERVISED
    model_id: str
    dataset_path: str
    dataset_remote: bool = False
    output_dir: str
    gpu_indices: list[int] = Field(default_factory=list)
    tensorboard: bool = True
    tensorboard_log_dir: str | None = None
    recommendation: dict[str, Any] | None = None
    resume_from_checkpoint: str | None = None
    parallelism_strategy_override: ParallelismStrategy | None = None

    @field_validator("model_id", "dataset_path", "output_dir")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value


class CheckpointRecord(BaseModel):
    id: int
    run_id: int
    path: str
    step: int | None = None
    created_at: str


class RunRecord(BaseModel):
    id: int
    status: RunStatus
    model_id: str
    use_case: str
    objective: str
    task_slug: str | None = None
    strategy: str | None = None
    peft: str | None = None
    dataset: str
    output_dir: str
    config_json: str = "{}"
    tensorboard_log_dir: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class TrainingEvent(BaseModel):
    run_id: int
    type: str
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(
        default_factory=time.time,
        description=(
            "Unix timestamp in seconds, not milliseconds, from time.time(). "
            "Clients must not treat this value as Date.now() milliseconds."
        ),
    )
