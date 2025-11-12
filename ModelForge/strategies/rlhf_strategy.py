"""
Reinforcement Learning from Human Feedback (RLHF) strategy implementation.
Uses TRL's PPOTrainer for RLHF-based fine-tuning.
"""
from typing import Any, Dict
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from ..logging_config import logger
from ..exceptions import TrainingError


class RLHFStrategy:
    """RLHF strategy using PPO (Proximal Policy Optimization)."""

    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "rlhf"

    def prepare_model(self, model: Any, config: Dict) -> Any:
        """
        Prepare model for RLHF training.

        Args:
            model: Base model instance
            config: Configuration with LoRA and RLHF settings

        Returns:
            Model prepared for RLHF
        """
        logger.info("Preparing model for RLHF")

        try:
            from trl import AutoModelForCausalLMWithValueHead
        except ImportError as e:
            raise TrainingError(
                "TRL is not installed with RLHF support. "
                "Install with: pip install trl[rlhf]"
            ) from e

        # If quantized, prepare for kbit training
        if config.get("use_4bit") or config.get("use_8bit"):
            model = prepare_model_for_kbit_training(model)

        # Apply LoRA if configured
        if config.get("use_lora", True):
            peft_config = LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=config.get("target_modules", "all-linear"),
            )
            model = get_peft_model(model, peft_config)

        # Wrap with value head for RLHF
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        logger.info("Model prepared for RLHF with value head")
        return model

    def prepare_dataset(self, dataset: Any, tokenizer: Any, config: Dict) -> Any:
        """
        Prepare dataset for RLHF.

        RLHF requires datasets with:
        - prompt: Input prompt
        - chosen: Preferred response
        - rejected: Non-preferred response

        Args:
            dataset: Raw dataset with RLHF fields
            tokenizer: Tokenizer instance
            config: Configuration dictionary

        Returns:
            Prepared dataset
        """
        logger.info("Preparing dataset for RLHF")

        # Validate required fields
        required_fields = self.get_required_dataset_fields()
        missing_fields = [f for f in required_fields if f not in dataset.column_names]

        if missing_fields:
            raise TrainingError(
                f"RLHF dataset missing required fields: {missing_fields}. "
                f"Required fields: {required_fields}"
            )

        logger.info(f"RLHF dataset prepared: {len(dataset)} examples")
        return dataset

    def create_trainer(
        self,
        model: Any,
        train_dataset: Any,
        eval_dataset: Any,
        tokenizer: Any,
        config: Dict,
        callbacks: list = None,
    ) -> Any:
        """
        Create PPOTrainer for RLHF.

        Args:
            model: Model with value head
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (not used in PPO)
            tokenizer: Tokenizer instance
            config: Training configuration
            callbacks: Training callbacks

        Returns:
            PPOTrainer instance
        """
        logger.info("Creating PPOTrainer for RLHF")

        try:
            from trl import PPOTrainer, PPOConfig
        except ImportError as e:
            raise TrainingError(
                "TRL is not installed with RLHF support. "
                "Install with: pip install trl[rlhf]"
            ) from e

        # Create PPO config
        ppo_config = PPOConfig(
            model_name=config.get("model_name", "model"),
            learning_rate=config.get("learning_rate", 1.41e-5),
            batch_size=config.get("batch_size", 128),
            mini_batch_size=config.get("mini_batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            optimize_cuda_cache=True,
            early_stopping=config.get("early_stopping", False),
            target_kl=config.get("target_kl", 0.1),
            ppo_epochs=config.get("ppo_epochs", 4),
            seed=config.get("seed", 0),
            log_with="tensorboard",
        )

        # Create trainer
        trainer = PPOTrainer(
            model=model,
            config=ppo_config,
            dataset=train_dataset,
            tokenizer=tokenizer,
        )

        logger.info("PPOTrainer created successfully")
        return trainer

    def get_required_dataset_fields(self) -> list:
        """
        Get required dataset fields for RLHF.

        Returns:
            List of required fields
        """
        return ["prompt", "chosen", "rejected"]
