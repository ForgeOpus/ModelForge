"""
RecommendationEngine: resolves a language use case + training objective + hardware specs into a
fully-resolved training recommendation covering base models, strategy, PEFT approach, provider,
and typed config parameters.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
from huggingface_hub import HfApi


from recommendation_engine.language_use_cases import LanguageUseCase, DataFormatSpec
from finetuning_strategies.strategies import FinetuningStrategy
from finetuning_strategies.tasks import FinetuningTask

from utilities.training_objective import TrainingObjective
from utilities.hardware_detector import HardwareDetector
from recommendation_engine.validation_models.recommendation import (
    RecommendedModel, Rationale, Recommendation, VramEstimate,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ACTIVATION_BUDGET_GB: float = 2.0   # conservative midpoint 1-4 GB with gradient_checkpointing
ADAPTER_COST_GB: float = 0.1        # ~10x headroom over rank-16 LoRA; also covers QLoRA dequant buffers
BOOTSTRAP_TIER: str = "qlora_4bit"  # pessimistic upper bound for Step 3 bootstrap

OBJECTIVE_STRATEGY_MAP: Dict[TrainingObjective, str] = {
    TrainingObjective.SUPERVISED:    "sft",
    TrainingObjective.PREFERENCE:    "dpo",
    TrainingObjective.REINFORCEMENT: "grpo",
}


class RecommendationEngine:
    """Resolves (use_case, objective, hardware) → Recommendation. No instantiation needed."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def recommend(
        use_case_slug: str,
        training_objective: TrainingObjective,
        hardware: HardwareDetector.HardwareSpecs,
        n: int = 5,
    ) -> Recommendation:
        """
        Run the full recommendation pipeline and return a Recommendation.

        Raises ValueError for: n <= 0, unknown use_case_slug, incompatible objective,
        misconfigured (task, strategy) pair, strategy overhead exceeding raw VRAM.

        Args:
            use_case_slug: identifies the language use case (e.g. "text_generation").
            training_objective: identifies the training objective (e.g. TrainingObjective.SUPERVISED).
            hardware: detailed hardware specs to inform the recommendation.
            n: number of recommended models to return (default 5).
        
        Returns:
            A Recommendation object containing the recommended models, strategy, PEFT approach, provider,
            config parameters, data format spec, and a detailed rationale for the recommendation.
        """
        # Guard
        if n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}.")

        # Step 1 — Raw VRAM
        raw_vram_gb, is_cpu_only, is_mps = RecommendationEngine._compute_raw_vram(hardware)

        # Step 2 — Resolve use case and strategy
        use_case_cls, strategy_cls, task_slug, strategy_slug, data_format = RecommendationEngine._resolve_use_case_and_strategy(use_case_slug, training_objective)

        # Step 3 — Adjusted VRAM
        adjusted_vram_gb = RecommendationEngine._compute_adjusted_vram(
            raw_vram_gb,
            strategy_cls.extra_model_copies,
            strategy_cls.batch_multiplier,
        )
        if adjusted_vram_gb < 0 and not is_cpu_only:
            raise ValueError(
                f"Strategy '{strategy_slug}' requires more VRAM than is available on this hardware "
                f"for any viable model size. Adjusted VRAM: {adjusted_vram_gb:.2f} GB."
            )

        # Step 4 — PEFT tier
        peft_tier, max_params, max_params_by_tier = RecommendationEngine._select_peft_tier(
            adjusted_vram_gb, is_cpu_only, is_mps, hardware.ram_gb,
        )

        # Step 5 — Hub query
        recommended_models, used_relaxed_retry, hub_warnings = RecommendationEngine._query_hub(
            hub_task_filter=use_case_cls.hub_task_filter,
            max_params=max_params,
            n=n,
            tier_p_max=max_params_by_tier[peft_tier],
        )

        # Step 6 — Provider
        provider_slug = "huggingface"

        # Step 7 — Strategy config
        if recommended_models:
            chosen_model = recommended_models[0]
            strategy_config, peft_config = strategy_cls.build_recommended_config(
                peft_tier, hardware, chosen_model
            )
        else:
            # Hub returned 0 results — skip config building
            strategy_config = None
            peft_config = None

        # Step 8 — Rationale
        all_warnings = list(hub_warnings)  # copy

        if is_cpu_only:
            all_warnings.append(
                "CPU-only training detected. Training will be extremely slow. "
                "bitsandbytes quantisation is not supported on CPU and has been disabled in the config."
            )
        if is_mps:
            all_warnings.append(
                "Apple MPS detected. bitsandbytes 4-bit and 8-bit quantisation are not supported on MPS. "
                "bf16 support is partial. Capped at LoRA bf16."
            )
        multi_gpu = sum(1 for g in hardware.gpus if isinstance(g["total_memory_gb"], (int, float))) > 1
        if multi_gpu:
            all_warnings.append(
                "Multi-GPU setup detected. VRAM has been summed across all GPUs. "
                "This engine does not generate DeepSpeed ZeRO or FSDP configuration — configure manually."
            )
        if not is_cpu_only and not is_mps:
            # Check for tight VRAM
            if max_params_by_tier.get("lora_bf16", 0) > 0 and peft_tier == "qlora_4bit":
                all_warnings.append("VRAM is tight. QLoRA 4-bit was selected as the only viable tier.")
        if peft_tier == "qlora_4bit" and max_params == 0:
            all_warnings.append(
                "No single-GPU tier is viable for this hardware. Even QLoRA 4-bit may OOM. "
                "Consider multi-GPU setup."
            )
        if not recommended_models:
            all_warnings.append(
                f"No models found on the Hub matching task '{use_case_cls.hub_task_filter}' "
                f"with at most {max_params / 1e9:.1f}B parameters. "
                "No strategy config was generated. Try increasing n or relaxing hardware constraints."
            )
        if used_relaxed_retry:
            all_warnings.append(
                "Initial Hub query returned 0 results. Retried with a 50% relaxed parameter upper bound."
            )

        # Disk space check
        if recommended_models:
            bytes_per_param = {"full_finetune": 4.0, "lora_bf16": 2.0, "qlora_8bit": 1.0, "qlora_4bit": 0.5}
            bpp = bytes_per_param.get(peft_tier, 0.5)
            largest_model_gb = max(m.num_parameters * bpp / 1e9 for m in recommended_models)
            if hardware.available_disk_storage_gb < largest_model_gb:
                all_warnings.append(
                    f"Available disk space ({hardware.available_disk_storage_gb:.1f} GB) may be insufficient "
                    f"for the largest recommended model (~{largest_model_gb:.1f} GB weight footprint at {peft_tier})."
                )

        # Build VRAM summary string
        gpu_count = len(hardware.gpus)
        gpu_vram_desc = (
            "CPU-only (no GPU)" if is_cpu_only
            else "Apple MPS (8 GB effective)" if is_mps
            else f"{raw_vram_gb:.1f} GB across {gpu_count} GPU(s)"
        )

        # Build tier math string
        tier_lines = []
        for tier, p_max in max_params_by_tier.items():
            tier_lines.append(f"  {tier}: P_max={p_max / 1e9:.2f}B params")
        tier_math = "\n".join(tier_lines)

        rationale = Rationale(
            vram_summary=(
                f"{gpu_vram_desc}. Adjusted VRAM (after strategy overhead): {adjusted_vram_gb:.2f} GB. "
                + ("Multi-GPU summed. " if multi_gpu else "")
            ),
            use_case_resolution=(
                f"Use case '{use_case_slug}' with objective '{training_objective.value}' → "
                f"task_slug='{task_slug}', strategy_slug='{strategy_slug}'. "
                f"Compatible because: {use_case_cls.name} supports {training_objective.value}, "
                f"and {strategy_slug} supports {task_slug}."
            ),
            peft_decision=(
                f"Selected tier: {peft_tier} (P_max={max_params / 1e9:.2f}B). "
                f"Bootstrap: assumed qlora_4bit (0.5 B/param) to estimate model weight overhead before tier is known. "
                f"Per-tier P_max:\n{tier_math}"
            ),
            model_filter=(
                f"Queried Hub: pipeline_tag='{use_case_cls.hub_task_filter}', "
                f"sort=downloads desc, max_params={max_params / 1e9:.2f}B. "
                f"Found {len(recommended_models)} model(s)."
                + (" Relaxed-retry applied (+50% param bound)." if used_relaxed_retry else "")
            ),
            warnings=all_warnings,
        )

        bytes_per_param_by_tier = {
            "full_finetune": 18.0, "lora_bf16": 2.0, "qlora_8bit": 1.0, "qlora_4bit": 0.5,
        }
        vram_estimate = VramEstimate(
            raw_vram_gb=raw_vram_gb,
            adjusted_vram_gb=adjusted_vram_gb,
            activation_budget_gb=ACTIVATION_BUDGET_GB,
            adapter_cost_gb=ADAPTER_COST_GB,
            bytes_per_param=bytes_per_param_by_tier.get(peft_tier, 0.5),
            max_params_by_tier=max_params_by_tier,
        )

        return Recommendation(
            recommended_models=recommended_models,
            task_slug=task_slug,
            strategy_slug=strategy_slug,
            peft_approach=peft_tier,
            provider_slug=provider_slug,
            strategy_config=strategy_config,
            peft_config=peft_config,
            data_format=data_format,
            rationale=rationale,
            vram_estimate=vram_estimate,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_raw_vram(hardware: HardwareDetector.HardwareSpecs) -> Tuple[float, bool, bool]:
        """
        Returns (raw_vram_gb, is_cpu_only, is_mps).

        - Sums all numeric GPU VRAM entries.
        - MPS GPUs (total_memory_gb == "N/A") are detected and treated as 8 GB effective (conservative).
        - CPU-only: no GPUs at all → raw_vram_gb = 0.
        """
        is_mps = any(g["total_memory_gb"] == "N/A" for g in hardware.gpus)
        is_cpu_only = len(hardware.gpus) == 0

        if is_mps:
            return (8.0, False, True)

        total = sum(
            g["total_memory_gb"]
            for g in hardware.gpus
            if isinstance(g["total_memory_gb"], (int, float))
        )
        return (total, total == 0, False)

    @staticmethod
    def _resolve_use_case_and_strategy(
        use_case_slug: str,
        training_objective: TrainingObjective,
    ) -> Tuple[type, type, str, str, DataFormatSpec]:
        """
        Returns (use_case_cls, strategy_cls, task_slug, strategy_slug, data_format).
        Raises ValueError for: unknown use_case_slug, incompatible objective, misconfigured (task, strategy) pair.
        """
        # Step 1: Look up use case
        if not LanguageUseCase.is_valid_use_case(use_case_slug):
            raise ValueError(
                f"Unknown use case slug '{use_case_slug}'. "
                f"Registered: {sorted(LanguageUseCase.get_use_cases())}"
            )
        use_case_cls = LanguageUseCase.get_use_case_by_slug(use_case_slug)

        # Step 2: Check objective compatibility
        if training_objective not in use_case_cls.compatible_objectives:
            raise ValueError(
                f"Use case '{use_case_slug}' does not support objective '{training_objective.value}'. "
                f"Compatible objectives: {[o.value for o in use_case_cls.compatible_objectives]}"
            )

        # Step 3: Resolve task and strategy slugs
        task_slug = use_case_cls.task_slug
        strategy_slug = OBJECTIVE_STRATEGY_MAP[training_objective]

        # Step 4: Look up strategy
        if strategy_slug not in FinetuningStrategy._registry:
            raise ValueError(
                f"Strategy '{strategy_slug}' is not registered. "
                f"Registered strategies: {sorted(FinetuningStrategy._registry)}"
            )
        strategy_cls = FinetuningStrategy._registry[strategy_slug]

        # Step 5: Verify task compatibility
        if task_slug not in strategy_cls.supported_tasks:
            raise ValueError(
                f"Strategy '{strategy_slug}' does not support task '{task_slug}'. "
                f"This is a misconfiguration in the use case or strategy definition. "
                f"Supported tasks: {strategy_cls.supported_tasks}"
            )

        # Step 6: Get data format
        data_format = use_case_cls.data_formats[training_objective]

        return (use_case_cls, strategy_cls, task_slug, strategy_slug, data_format)

    @staticmethod
    def _compute_adjusted_vram(
        raw_vram_gb: float,
        extra_model_copies: int,
        batch_multiplier: float,
    ) -> float:
        """
        Applies strategy overhead to produce an adjusted VRAM budget (in GB).

        Uses BOOTSTRAP_TIER ("qlora_4bit", 0.5 bytes/param) to break the circular dependency
        between model size and tier selection. This is intentionally pessimistic (lower bound on
        adjusted VRAM), which makes tier selection conservative.

        Bootstrap math (from spec Step 3):
            P_bootstrap = (raw_vram_bytes - ACT_BUDGET_bytes - ADAPTER_bytes) / 0.5
            estimated_model_weight_gb = (0.5 X P_bootstrap) / 1e9

        Adjustment:
            adjusted = raw_vram_gb
                     - (extra_model_copies X estimated_model_weight_gb)
                     - ((batch_multiplier - 1.0) X ACTIVATION_BUDGET_GB)
        """
        act_bytes = ACTIVATION_BUDGET_GB * 1e9
        adapter_bytes = ADAPTER_COST_GB * 1e9
        raw_bytes = raw_vram_gb * 1e9

        p_bootstrap = (raw_bytes - act_bytes - adapter_bytes) / 0.5
        if p_bootstrap < 0:
            p_bootstrap = 0.0
        estimated_weight_gb = (0.5 * p_bootstrap) / 1e9

        adjusted = (
            raw_vram_gb
            - (extra_model_copies * estimated_weight_gb)
            - ((batch_multiplier - 1.0) * ACTIVATION_BUDGET_GB)
        )
        return adjusted

    @staticmethod
    def _select_peft_tier(
        adjusted_vram_gb: float,
        is_cpu_only: bool,
        is_mps: bool,
        ram_gb: float,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Returns (peft_tier, max_params, max_params_by_tier) where max_params is the upper bound
        for the chosen tier and max_params_by_tier maps each tier name to its P_max.

        Tier inequalities (from memory_strategy_guide.md §5):
            full_finetune : adjusted_vram_gb >= P × 18 bytes + ACTIVATION_BUDGET_GB
            lora_bf16     : adjusted_vram_gb >= P × 2  bytes + ADAPTER_COST_GB + ACTIVATION_BUDGET_GB
            qlora_8bit    : adjusted_vram_gb >= P × 1  byte  + ADAPTER_COST_GB + ACTIVATION_BUDGET_GB
            qlora_4bit    : adjusted_vram_gb >= P × 0.5 byte + ADAPTER_COST_GB + ACTIVATION_BUDGET_GB

        CPU-only: P_max capped at min(500M, ram_gb*1e9/3), tier always "qlora_4bit".
        MPS: cap at "lora_bf16" (bitsandbytes not supported on MPS).

        Selection order: prefer full_finetune > lora_bf16 > qlora_8bit > qlora_4bit.
        """
        if is_cpu_only:
            p_max_cpu = min(500_000_000, ram_gb * 1e9 / 3)
            return "qlora_4bit", p_max_cpu, {
                "full_finetune": 0.0, "lora_bf16": 0.0, "qlora_8bit": 0.0, "qlora_4bit": p_max_cpu
            }

        act = ACTIVATION_BUDGET_GB
        adapter = ADAPTER_COST_GB

        p_max_full = max(0.0, (adjusted_vram_gb - act) / 18 * 1e9)
        p_max_lora = max(0.0, (adjusted_vram_gb - adapter - act) / 2 * 1e9)
        p_max_8bit = max(0.0, (adjusted_vram_gb - adapter - act) / 1 * 1e9)
        p_max_4bit = max(0.0, (adjusted_vram_gb - adapter - act) / 0.5 * 1e9)

        by_tier = {
            "full_finetune": p_max_full,
            "lora_bf16": p_max_lora,
            "qlora_8bit": p_max_8bit,
            "qlora_4bit": p_max_4bit,
        }

        if is_mps:
            # MPS: cap at lora_bf16, no quantisation
            by_tier["qlora_8bit"] = 0.0
            by_tier["qlora_4bit"] = 0.0
            p_max_lora_mps = by_tier["lora_bf16"]
            if p_max_full > 0:
                return "full_finetune", p_max_full, by_tier
            elif p_max_lora_mps > 0:
                return "lora_bf16", p_max_lora_mps, by_tier
            else:
                return "lora_bf16", p_max_lora_mps, by_tier  # best we can do on MPS

        # Normal selection: prefer quality
        if p_max_full > 0:
            return "full_finetune", p_max_full, by_tier
        elif p_max_lora > 0:
            return "lora_bf16", p_max_lora, by_tier
        elif p_max_8bit > 0:
            return "qlora_8bit", p_max_8bit, by_tier
        elif p_max_4bit > 0:
            return "qlora_4bit", p_max_4bit, by_tier
        else:
            # No single-GPU tier viable — return qlora_4bit with max_params=0 so caller can warn
            return "qlora_4bit", 0.0, by_tier

    @staticmethod
    def _query_hub(
        hub_task_filter: str,
        max_params: float,
        n: int,
        tier_p_max: float,
    ) -> Tuple[List[RecommendedModel], bool, List[str]]:
        """
        Query HuggingFace Hub for candidate models.
        Returns (models, used_relaxed_retry, warnings).

        - Filters by pipeline_tag=hub_task_filter
        - Skips models with missing safetensors metadata (num_parameters unavailable)
        - Filters to models with num_parameters <= max_params
        - Sorts by downloads descending (Hub does this; we just collect in order)
        - Limits to n results
        - Retry: if 0 results, retry once with max_params * 1.5 capped at tier_p_max
        """
        api = HfApi()

        def _fetch(limit_params: float):
            results = []
            fetch_limit = n * 20
            missing_metadata_count = 0
            for model in api.list_models(
                pipeline_tag=hub_task_filter,
                sort="downloads",
                expand=["safetensors", "downloads"],
                limit=fetch_limit,
            ):
                if model.safetensors is None or model.safetensors.total is None:
                    missing_metadata_count += 1
                    continue
                total_params = model.safetensors.total
                if total_params > limit_params:
                    continue
                results.append(RecommendedModel(
                    model_id=model.id,
                    num_parameters=total_params,
                    downloads=model.downloads or 0,
                ))
                if len(results) >= n:
                    break
            return results, missing_metadata_count

        results, missing_count = _fetch(max_params)
        warnings = []
        if missing_count > 0:
            warnings.append(
                f"{missing_count} models were skipped due to missing parameter metadata on the Hub."
            )

        used_relaxed = False
        if len(results) == 0:
            relaxed_limit = min(max_params * 1.5, tier_p_max)
            results, missing_count2 = _fetch(relaxed_limit)
            used_relaxed = True
            if missing_count2 > 0:
                warnings.append(
                    f"Relaxed retry: {missing_count2} additional models skipped for missing metadata."
                )

        return results, used_relaxed, warnings
