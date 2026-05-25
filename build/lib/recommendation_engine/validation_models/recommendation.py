from __future__ import annotations

from typing import Any, List
from pydantic import BaseModel, Field, ConfigDict

from recommendation_engine.language_use_cases.use_case import DataFormatSpec


class RecommendedModel(BaseModel):
    """A single candidate base model returned by the Hub query."""

    model_id: str = Field(..., description="HuggingFace repo ID (e.g. 'google/flan-t5-base')")
    num_parameters: int = Field(..., description="Parameter count as reported by the Hub")
    downloads: int = Field(..., description="Monthly download count — used as quality proxy for ranking")


class Rationale(BaseModel):
    """Structured explanation of every decision the engine made."""

    vram_summary: str = Field(..., description="Detected VRAM, GPU count, and multi-GPU note")
    use_case_resolution: str = Field(
        ...,
        description="How (use_case + objective) resolved to (task_slug + strategy_slug)"
    )
    peft_decision: str = Field(
        ...,
        description="PEFT tier selected and the math behind it (P_max per tier, bootstrap reasoning)"
    )
    model_filter: str = Field(
        ...,
        description="Parameter range, Hub task filter used, sort order; relaxed-retry note if applicable"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Free-form warnings (tight VRAM, MPS limits, CPU-only, empty Hub result, etc.)"
    )

    def __str__(self) -> str:
        lines = [
            "=== Recommendation Rationale ===",
            f"Hardware:        {self.vram_summary}",
            f"Use Case:        {self.use_case_resolution}",
            f"PEFT Decision:   {self.peft_decision}",
            f"Model Filter:    {self.model_filter}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        return "\n".join(lines)


class Recommendation(BaseModel):
    """Full output of RecommendationEngine.recommend()."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    recommended_models: List[RecommendedModel] = Field(
        ..., description="Ranked list of compatible base models (descending by downloads)"
    )
    task_slug: str = Field(..., description="Resolved FinetuningTask slug (from use_case_slug)")
    strategy_slug: str = Field(..., description="Resolved strategy slug (from training_objective)")
    peft_approach: str = Field(
        ..., description="One of: full_finetune, lora_bf16, qlora_8bit, qlora_4bit"
    )
    provider_slug: str = Field(..., description="Provider to use (e.g. 'huggingface')")
    strategy_config: Any = Field(
        ..., description="Typed Pydantic config for the resolved strategy (e.g. SFTRecommendedConfig)"
    )
    peft_config: Any = Field(
        None, description="Typed Pydantic LoRA/QLoRA params if applicable; None for full fine-tuning"
    )
    data_format: DataFormatSpec = Field(
        ..., description="User-facing data format spec for the resolved (use case, objective) pair"
    )
    rationale: Rationale = Field(..., description="Structured explanation of all engine decisions")
