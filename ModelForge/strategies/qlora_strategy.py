"""
QLoRA (Quantized Low-Rank Adaptation) strategy implementation.
QLoRA combines 4-bit quantization with LoRA for memory-efficient fine-tuning.
"""
from typing import Any, Dict
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Import unsloth first to prevent EOS token corruption
# This must come before TRL imports to ensure proper tokenizer initialization
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
            Model with QLoRA adapters
        """
        logger.info("Preparing model for QLoRA")

        # QLoRA requires the model to be prepared for kbit training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.get("gradient_checkpointing", True)
        )

        # Get task type
        task_type_map = {
            "text-generation": TaskType.CAUSAL_LM,
            "summarization": TaskType.SEQ_2_SEQ_LM,
            "extractive-question-answering": TaskType.QUESTION_ANS,
        }
        task_type = task_type_map.get(config.get("task"), TaskType.CAUSAL_LM)

        # QLoRA-specific LoRA config
        peft_config = LoraConfig(
            r=config.get("lora_r", 64),  # QLoRA often uses higher rank
            lora_alpha=config.get("lora_alpha", 16),  # Lower alpha for QLoRA
            lora_dropout=config.get("lora_dropout", 0.1),
            bias="none",
            task_type=task_type,
            target_modules=config.get("target_modules", "all-linear"),
            # QLoRA-specific settings
            use_rslora=config.get("use_rslora", False),
            use_dora=config.get("use_dora", False),
        )

        # Apply PEFT
        model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing for memory efficiency
        if config.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()

        logger.info(
            f"Model prepared with QLoRA: r={peft_config.r}, "
            f"alpha={peft_config.lora_alpha}"
        )
        return model

    def prepare_dataset(self, dataset: Any, tokenizer: Any, config: Dict) -> Any:
        """
        Prepare dataset for QLoRA by consolidating all fields into a single 'text' field.

        Args:
            dataset: Pre-formatted dataset with task-specific fields
            tokenizer: Tokenizer instance (for EOS token)
            config: Configuration dictionary (contains task type)

        Returns:
            Dataset with consolidated 'text' field
        """
        logger.info(f"Preparing dataset for QLoRA: {len(dataset)} examples")

        # Get EOS token with SEP fallback
        eos_token = tokenizer.eos_token or tokenizer.sep_token or ""
        task = config.get("task", "text-generation")

        def create_text_field(example):
            """Consolidate all fields into a single 'text' field with EOS token."""
            if task == "text-generation":
                # Fields: prompt, completion (renamed in training_service._format_dataset)
                prompt = example.get('prompt', '')
                completion = example.get('completion', '')
                text = f"USER: {prompt}\nASSISTANT: {completion}{eos_token}"

            elif task == "summarization":
                # Fields: input, output (renamed in training_service._format_dataset)
                input_text = example.get('input', '')
                output_text = example.get('output', '')
                text = f"Summarize the following document:\n{input_text}\n\nSummary:\n{output_text}{eos_token}"

            elif task == "extractive-question-answering":
                # Fields: context, question, answers
                context = example.get('context', '')
                question = example.get('question', '')
                answers = example.get('answers', {})

                # Extract answer text from answers dict
                if isinstance(answers, dict):
                    answer_text = answers.get("text", [""])[0] if "text" in answers else ""
                else:
                    answer_text = str(answers)

                text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer_text}{eos_token}"

            else:
                # Fallback: concatenate all string fields
                logger.warning(f"Unknown task type: {task}, using raw field concatenation")
                text = " ".join(str(v) for v in example.values() if isinstance(v, str))
                text += eos_token

            return {"text": text}

        # Apply transformation and remove original columns
        dataset = dataset.map(create_text_field, remove_columns=dataset.column_names, num_proc=1)

        logger.info(f"Dataset prepared with consolidated 'text' field: {len(dataset)} examples")
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
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer instance
            config: Training configuration
            callbacks: Training callbacks

        Returns:
            SFTTrainer instance
        """
        logger.info("Creating SFTTrainer with QLoRA optimizations")

        # QLoRA-optimized training arguments
        training_args = SFTConfig(
            output_dir=config.get("output_dir", "./checkpoints"),
            num_train_epochs=config.get("num_train_epochs", 1),
            # QLoRA can use larger batch sizes due to memory efficiency
            per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            # QLoRA uses paged optimizers for memory efficiency
            optim=config.get("optim", "paged_adamw_32bit"),
            save_steps=config.get("save_steps", 0),
            logging_steps=config.get("logging_steps", 25),
            # QLoRA can often use higher learning rates
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
            max_seq_length=config.get("max_seq_length", None),
            packing=config.get("packing", False),
            # Gradient checkpointing for memory efficiency
            gradient_checkpointing=config.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # Evaluation settings
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=config.get("eval_steps", 100),
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            # Use tokenizer's EOS token instead of corrupted placeholder
            eos_token=config.get("eos_token"),
            # Disable completion_only_loss to avoid conflicts
            completion_only_loss=False,
            # Disable distributed training for Unsloth (required when using device_map='auto')
            ddp_find_unused_parameters=False,
        )

        # Create trainer (dataset has been formatted to 'text' field in prepare_dataset)
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            processing_class=tokenizer,
            callbacks=callbacks or [],
        )

        logger.info("QLoRA trainer created successfully")
        return trainer

    def get_required_dataset_fields(self) -> list:
        """
        Get required dataset fields for QLoRA.

        Returns:
            List of required fields (same as SFT)
        """
        return ["input", "output"]
