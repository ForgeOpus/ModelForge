from __future__ import annotations

from utilities.fsdp_config import build_fsdp_config


class TestBuildFsdpConfig:
    """Tests for build_fsdp_config — returns HF TrainingArguments overrides that activate FSDP."""

    def test_enables_full_shard_auto_wrap(self):
        config = build_fsdp_config()
        assert config["fsdp"] == "full_shard auto_wrap"

    def test_disables_gradient_checkpointing(self):
        # FSDP uses activation_checkpointing instead; enabling both raises in the HF Trainer.
        config = build_fsdp_config()
        assert config["gradient_checkpointing"] is False
        assert config["fsdp_config"]["activation_checkpointing"] is True

    def test_disables_incompatible_options(self):
        config = build_fsdp_config()
        assert config["use_liger_kernel"] is False
        assert config["activation_offloading"] is False

    def test_fsdp_config_contents(self):
        fsdp_config = build_fsdp_config()["fsdp_config"]
        assert fsdp_config["min_num_params"] == 100_000_000
        assert fsdp_config["backward_prefetch"] == "backward_pre"
        assert fsdp_config["state_dict_type"] == "FULL_STATE_DICT"

    def test_no_mixed_precision_keys(self):
        # Mixed precision is derived from the config's bf16/fp16 flags by the Trainer, not here.
        config = build_fsdp_config()
        assert "bf16" not in config
        assert "fp16" not in config
        assert "mixed_precision" not in config

    def test_returns_fresh_dict_each_call(self):
        a = build_fsdp_config()
        a["fsdp"] = "mutated"
        b = build_fsdp_config()
        assert b["fsdp"] == "full_shard auto_wrap"

    def test_applies_to_sft_config(self):
        # The overrides must be accepted by a real SFTConfig (TrainingArguments subclass).
        from trl import SFTConfig

        config = build_fsdp_config()
        sft_config = SFTConfig(
            output_dir="x",
            bf16=False,
            fp16=False,
            use_cpu=True,
            fsdp=config["fsdp"],
            fsdp_config=config["fsdp_config"],
            gradient_checkpointing=config["gradient_checkpointing"],
            use_liger_kernel=config["use_liger_kernel"],
            activation_offloading=config["activation_offloading"],
        )
        assert sft_config.fsdp_config["min_num_params"] == 100_000_000
        assert sft_config.gradient_checkpointing is False
