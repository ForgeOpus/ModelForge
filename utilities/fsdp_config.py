from __future__ import annotations

from typing import Any, Dict


# Below this parameter count an FSDP unit is not worth its own all-gather. A size-based policy
# is a generic default that works across model families without knowing the transformer block class.
_FSDP_MIN_NUM_PARAMS = 100_000_000


def build_fsdp_config() -> Dict[str, Any]:
    """Return HF ``TrainingArguments`` overrides that activate FSDP full-shard.

    FSDP must be driven through the Trainer's own ``fsdp`` / ``fsdp_config`` arguments rather than
    by mutating ``AcceleratorState``: the Trainer builds the FSDP plugin from these arguments
    (``TrainingArguments._process_fsdp_args``), sets ``ACCELERATE_USE_FSDP``, and constructs its own
    ``Accelerator`` — a plugin stashed on the shared state beforehand is simply overwritten and
    ``is_fsdp_enabled`` would stay False, leaving the run silently data-parallel.

    Mixed precision is intentionally omitted: the Trainer derives FSDP mixed precision from the
    ``bf16`` / ``fp16`` flags already present in the recommended config.

    Restricted to full fine-tuning (enforced by ``validate_parallelism_compatibility``); FSDP cannot
    shard LoRA/QLoRA adapter parameters.
    """
    return {
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "min_num_params": _FSDP_MIN_NUM_PARAMS,
            "backward_prefetch": "backward_pre",
            "state_dict_type": "FULL_STATE_DICT",
            "activation_checkpointing": True,
        },
        # FSDP activation checkpointing replaces gradient_checkpointing; enabling both raises in
        # the HF Trainer and adds a redundant all-gather in the backward pass (transformers #30404).
        "gradient_checkpointing": False,
        # Incompatible with FSDP parameter sharding.
        "use_liger_kernel": False,
        "activation_offloading": False,
    }
