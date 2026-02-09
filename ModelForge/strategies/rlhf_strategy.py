"""
Reinforcement Learning from Human Feedback (RLHF) strategy implementation.

Uses DPO (Direct Preference Optimization) as the underlying algorithm.
DPO is the modern, stable replacement for PPO-based RLHF - it achieves
equivalent alignment results without requiring a separate reward model,
making it practical for a no-code fine-tuning tool.

Dataset format is the same as DPO: prompt, chosen, rejected.
"""
from typing import Any, Dict
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training

from ..logging_config import logger
from ..exceptions import TrainingError


class RLHFStrategy:
    """
    RLHF strategy using DPO (Direct Preference Optimization).

    DPO replaces the traditional PPO-based RLHF pipeline by directly
    optimizing the policy using preference data, without needing a
    separate reward model. This makes it simpler, more stable, and
    more practical for end users.
    """

    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "rlhf"

    def prepare_model(self, model: Any, config: Dict) -> Any:
        """
        Prepare model for RLHF/DPO training.

        Args:
            model: Base model instance
            config: Configuration with LoRA and training settings

        Returns:
            Model prepared for training
        """
        logger.info("Preparing model for RLHF (DPO-based)")

        # If quantized, prepare for kbit training
        if config.get("use_4bit") or config.get("use_8bit"):
            model = prepare_model_for_kbit_training(model)

        logger.info("Model prepared for RLHF")
        return model

    def get_peft_config(self, config: Dict) -> LoraConfig:
        """
        Create the LoRA PEFT config for RLHF training.

        Args:
            config: Configuration dictionary

        Returns:
            LoraConfig instance
        """
        peft_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=config.get("target_modules", "all-linear"),
        )

        logger.info(f"RLHF LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")
        return peft_config

    def prepare_dataset(self, dataset: Any, tokenizer: Any, config: Dict) -> Any:
        """
        Prepare dataset for RLHF.

        Requires datasets with:
        - prompt: Input prompt
        - chosen: Preferred response
        - rejected: Non-preferred response

        DPOTrainer handles tokenization automatically.

        Args:
            dataset: Raw dataset with preference fields
            tokenizer: Tokenizer instance
            config: Configuration dictionary

        Returns:
            Validated dataset (DPOTrainer handles tokenization)
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
        Create DPOTrainer for RLHF-style preference learning.

        Args:
            model: Prepared model
            train_dataset: Training dataset with prompt/chosen/rejected
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer instance
            config: Training configuration
            callbacks: Training callbacks

        Returns:
            DPOTrainer instance
        """
        logger.info("Creating DPOTrainer for RLHF")

        try:
            from trl import DPOTrainer, DPOConfig
        except ImportError as e:
            raise TrainingError(
                "TRL is not installed. Install with: pip install trl"
            ) from e

        # RLHF-tuned DPO config
        training_args = DPOConfig(
            output_dir=config.get("output_dir", "./checkpoints"),
            num_train_epochs=config.get("num_train_epochs", 1),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            learning_rate=config.get("learning_rate", 1.41e-5),
            warmup_ratio=config.get("warmup_ratio", 0.1),
            weight_decay=config.get("weight_decay", 0.05),
            fp16=config.get("fp16", False),
            bf16=config.get("bf16", False),
            max_grad_norm=config.get("max_grad_norm", 0.3),
            logging_steps=config.get("logging_steps", 10),
            save_steps=config.get("save_steps", 100),
            # DPO-specific settings for RLHF behavior
            beta=config.get("beta", 0.1),
            loss_type=config.get("loss_type", "sigmoid"),
            max_length=config.get("max_seq_length", 512),
            max_prompt_length=config.get("max_prompt_length", 128),
            # Evaluation settings
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=config.get("eval_steps", 100),
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            report_to="tensorboard",
            logging_dir=config.get("logging_dir", "./training_logs"),
            ddp_find_unused_parameters=False,
        )

        # Build the PEFT config
        peft_config = self.get_peft_config(config)

        # Create DPOTrainer
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=callbacks or [],
        )

        logger.info("DPOTrainer created successfully for RLHF")
        return trainer

    def get_required_dataset_fields(self) -> list:
        """
        Get required dataset fields for RLHF.

        Returns:
            List of required fields
        """
        return ["prompt", "chosen", "rejected"]
