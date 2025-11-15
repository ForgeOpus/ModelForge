"""
Direct Preference Optimization (DPO) strategy implementation.
DPO is a simpler alternative to RLHF that doesn't require a reward model.
"""
from typing import Any, Dict
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from ..logging_config import logger
from ..exceptions import TrainingError


class DPOStrategy:
    """Direct Preference Optimization strategy."""

    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "dpo"

    def prepare_model(self, model: Any, config: Dict) -> Any:
        """
        Prepare model for DPO training.

        Args:
            model: Base model instance
            config: Configuration with LoRA settings

        Returns:
            Model prepared for DPO
        """
        logger.info("Preparing model for DPO")

        # If quantized, prepare for kbit training
        if config.get("use_4bit") or config.get("use_8bit"):
            model = prepare_model_for_kbit_training(model)

        # Apply LoRA
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

        logger.info("Model prepared for DPO with LoRA")
        return model

    def prepare_dataset(self, dataset: Any, tokenizer: Any, config: Dict) -> Any:
        """
        Prepare dataset for DPO.

        DPO requires datasets with:
        - prompt: Input prompt
        - chosen: Preferred response
        - rejected: Non-preferred response

        Args:
            dataset: Raw dataset with DPO fields
            tokenizer: Tokenizer instance
            config: Configuration dictionary

        Returns:
            Prepared dataset
        """
        logger.info("Preparing dataset for DPO")

        # Validate required fields
        required_fields = self.get_required_dataset_fields()
        missing_fields = [f for f in required_fields if f not in dataset.column_names]

        if missing_fields:
            raise TrainingError(
                f"DPO dataset missing required fields: {missing_fields}. "
                f"Required fields: {required_fields}"
            )

        logger.info(f"DPO dataset prepared: {len(dataset)} examples")
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
        Create DPOTrainer.

        Args:
            model: Prepared model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer instance
            config: Training configuration
            callbacks: Training callbacks

        Returns:
            DPOTrainer instance
        """
        logger.info("Creating DPOTrainer")

        try:
            from trl import DPOTrainer, DPOConfig
        except ImportError as e:
            raise TrainingError(
                "TRL with DPO support is not installed. "
                "Install with: pip install trl"
            ) from e

        # Create DPO config
        training_args = DPOConfig(
            output_dir=config.get("output_dir", "./checkpoints"),
            num_train_epochs=config.get("num_train_epochs", 1),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            learning_rate=config.get("learning_rate", 5e-4),
            warmup_ratio=config.get("warmup_ratio", 0.1),
            weight_decay=config.get("weight_decay", 0.05),
            fp16=config.get("fp16", False),
            bf16=config.get("bf16", False),
            max_grad_norm=config.get("max_grad_norm", 0.3),
            logging_steps=config.get("logging_steps", 10),
            save_steps=config.get("save_steps", 100),
            # DPO-specific settings
            beta=config.get("beta", 0.1),  # DPO temperature parameter
            loss_type=config.get("loss_type", "sigmoid"),  # sigmoid or hinge
            max_length=config.get("max_seq_length", 512),
            max_prompt_length=config.get("max_prompt_length", 128),
            # Evaluation settings
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=config.get("eval_steps", 100),
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            report_to="tensorboard",
            logging_dir=config.get("logging_dir", "./training_logs"),
            # Disable distributed training for Unsloth (required when using device_map='auto')
            ddp_find_unused_parameters=False,
        )

        # Create trainer
        # Note: DPOTrainer needs a reference model for computing KL divergence
        # We'll use the same model as reference (it will be frozen internally)
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # DPOTrainer will create a copy
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks or [],
        )

        logger.info("DPOTrainer created successfully")
        return trainer

    def get_required_dataset_fields(self) -> list:
        """
        Get required dataset fields for DPO.

        Returns:
            List of required fields
        """
        return ["prompt", "chosen", "rejected"]
