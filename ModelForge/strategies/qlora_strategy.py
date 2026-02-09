"""
QLoRA (Quantized Low-Rank Adaptation) strategy implementation.
QLoRA combines 4-bit quantization with LoRA for memory-efficient fine-tuning.
Uses TRL's SFTTrainer with QLoRA-optimized defaults.
"""
from typing import Any, Dict
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training

# Import unsloth first to prevent EOS token corruption
# This must come before transformers imports to ensure proper tokenizer initialization
try:
    import unsloth
except ImportError:
    pass

from trl import SFTTrainer, SFTConfig

from ..logging_config import logger


class QLoRAStrategy:
    """
    QLoRA strategy for memory-efficient fine-tuning.

    QLoRA specifically uses:
    - 4-bit NormalFloat quantization
    - Double quantization
    - Gradient checkpointing
    - Paged optimizers
    - Higher LoRA rank to compensate for quantization
    """

    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "qlora"

    def prepare_model(self, model: Any, config: Dict) -> Any:
        """
        Prepare model with QLoRA optimizations.

        Args:
            model: Base model instance (should be 4-bit quantized)
            config: Configuration with LoRA settings

        Returns:
            Model prepared for QLoRA training
        """
        logger.info("Preparing model for QLoRA")

        # QLoRA requires the model to be prepared for kbit training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.get("gradient_checkpointing", True)
        )

        # Enable gradient checkpointing for memory efficiency
        if config.get("gradient_checkpointing", True):
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        logger.info("Model prepared for QLoRA")
        return model

    def get_peft_config(self, config: Dict) -> LoraConfig:
        """
        Create the LoRA PEFT config with QLoRA-specific defaults.

        QLoRA typically uses higher rank to compensate for quantization.

        Args:
            config: Configuration dictionary

        Returns:
            LoraConfig instance
        """
        task_type_map = {
            "text-generation": TaskType.CAUSAL_LM,
            "summarization": TaskType.SEQ_2_SEQ_LM,
            "extractive-question-answering": TaskType.QUESTION_ANS,
        }
        task_type = task_type_map.get(config.get("task"), TaskType.CAUSAL_LM)

        # QLoRA-specific LoRA config with higher rank
        peft_config = LoraConfig(
            r=config.get("lora_r", 64),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.1),
            bias="none",
            task_type=task_type,
            target_modules=config.get("target_modules", "all-linear"),
        )

        logger.info(
            f"QLoRA LoRA config: r={peft_config.r}, "
            f"alpha={peft_config.lora_alpha}"
        )
        return peft_config

    def prepare_dataset(self, dataset: Any, tokenizer: Any, config: Dict) -> Any:
        """
        Prepare dataset for QLoRA. SFTTrainer handles tokenization automatically,
        so we just need to format the data into the expected structure.

        Args:
            dataset: Pre-formatted dataset with task-specific fields
            tokenizer: Tokenizer instance
            config: Configuration dictionary

        Returns:
            Dataset formatted for SFTTrainer (with 'text' field)
        """
        logger.info(f"Preparing dataset for QLoRA: {len(dataset)} examples")

        task = config.get("task", "text-generation")
        eos_token = tokenizer.eos_token or tokenizer.sep_token or ""

        def format_to_text(example):
            """Format each example into a single text field for SFTTrainer."""
            if task == "text-generation":
                prompt = example.get('prompt', '')
                completion = example.get('completion', '')
                text = f"USER: {prompt}\nASSISTANT: {completion}{eos_token}"

            elif task == "summarization":
                input_text = example.get('input', '')
                output_text = example.get('output', '')
                text = f"Summarize the following document:\n{input_text}\n\nSummary:\n{output_text}{eos_token}"

            elif task == "extractive-question-answering":
                context = example.get('context', '')
                question = example.get('question', '')
                answers = example.get('answers', {})

                if isinstance(answers, dict):
                    answer_text = answers.get("text", [""])[0] if "text" in answers else ""
                else:
                    answer_text = str(answers)

                text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer_text}{eos_token}"

            else:
                logger.warning(f"Unknown task type: {task}, using raw field concatenation")
                text = " ".join(str(v) for v in example.values() if isinstance(v, str))
                text += eos_token

            return {"text": text}

        # Format dataset - SFTTrainer will handle tokenization
        dataset = dataset.map(
            format_to_text,
            remove_columns=dataset.column_names,
            num_proc=1
        )

        logger.info(f"Dataset formatted for QLoRA: {len(dataset)} examples")
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
        Create SFTTrainer with QLoRA-specific optimizations.

        Args:
            model: Prepared model with QLoRA
            train_dataset: Formatted training dataset (with 'text' field)
            eval_dataset: Formatted evaluation dataset
            tokenizer: Tokenizer instance
            config: Training configuration
            callbacks: Training callbacks

        Returns:
            SFTTrainer instance configured for QLoRA
        """
        logger.info("Creating SFTTrainer with QLoRA optimizations")

        max_seq_length = config.get("max_seq_length", 2048)
        if max_seq_length == -1:
            max_seq_length = 2048

        # QLoRA-optimized training arguments
        training_args = SFTConfig(
            output_dir=config.get("output_dir", "./checkpoints"),
            num_train_epochs=config.get("num_train_epochs", 1),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            # QLoRA uses paged optimizers for memory efficiency
            optim=config.get("optim", "paged_adamw_32bit"),
            save_steps=config.get("save_steps", 0),
            logging_steps=config.get("logging_steps", 25),
            learning_rate=config.get("learning_rate", 2e-4),
            warmup_ratio=config.get("warmup_ratio", 0.03),
            weight_decay=config.get("weight_decay", 0.001),
            fp16=config.get("fp16", False),
            bf16=config.get("bf16", True),  # BF16 recommended for QLoRA
            max_grad_norm=config.get("max_grad_norm", 0.3),
            max_steps=config.get("max_steps", -1),
            group_by_length=config.get("group_by_length", True),
            lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
            report_to="tensorboard",
            logging_dir=config.get("logging_dir", "./training_logs"),
            # Gradient checkpointing for memory efficiency
            gradient_checkpointing=config.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # SFT-specific settings
            max_seq_length=max_seq_length,
            packing=config.get("packing", False),
            dataset_text_field="text",
            # Evaluation settings
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=config.get("eval_steps", 100),
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            ddp_find_unused_parameters=False,
        )

        # Build the PEFT config for SFTTrainer to apply LoRA
        peft_config = self.get_peft_config(config)

        # Create SFTTrainer - it handles tokenization and data collation
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=callbacks or [],
        )

        logger.info("QLoRA SFTTrainer created successfully")
        return trainer

    def get_required_dataset_fields(self) -> list:
        """
        Get required dataset fields for QLoRA.

        Returns:
            List of required fields (same as SFT)
        """
        return ["input", "output"]
