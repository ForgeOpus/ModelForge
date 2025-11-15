"""
Supervised Fine-Tuning (SFT) strategy implementation.
Uses TRL's SFTTrainer for standard supervised fine-tuning with LoRA.
"""
from typing import Any, Dict, Optional
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Import unsloth first to prevent EOS token corruption
# This must come before TRL imports to ensure proper tokenizer initialization
try:
    import unsloth
except ImportError:
    pass

from trl import SFTTrainer, SFTConfig

from ..logging_config import logger


class SFTStrategy:
    """Supervised Fine-Tuning strategy using TRL."""

    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "sft"

    def prepare_model(self, model: Any, config: Dict) -> Any:
        """
        Prepare model with LoRA adapters for SFT.

        Args:
            model: Base model instance
            config: Configuration with LoRA settings

        Returns:
            Model with PEFT adapters
        """
        logger.info("Preparing model for SFT with LoRA")

        # If quantized, prepare for kbit training
        if config.get("use_4bit") or config.get("use_8bit"):
            model = prepare_model_for_kbit_training(model)

        # Get task type
        task_type_map = {
            "text-generation": TaskType.CAUSAL_LM,
            "summarization": TaskType.SEQ_2_SEQ_LM,
            "extractive-question-answering": TaskType.QUESTION_ANS,
        }
        task_type = task_type_map.get(config.get("task"), TaskType.CAUSAL_LM)

        # Create LoRA config
        peft_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.1),
            bias="none",
            task_type=task_type,
            target_modules=config.get("target_modules", "all-linear"),
        )

        # Apply PEFT
        model = get_peft_model(model, peft_config)

        logger.info(f"Model prepared with LoRA: r={peft_config.r}, alpha={peft_config.lora_alpha}")
        return model

    def prepare_dataset(self, dataset: Any, tokenizer: Any, config: Dict) -> Any:
        """
        Prepare dataset for SFT by consolidating all fields into a single 'text' field.

        Args:
            dataset: Pre-formatted dataset with task-specific fields
            tokenizer: Tokenizer instance (for EOS token)
            config: Configuration dictionary (contains task type)

        Returns:
            Dataset with consolidated 'text' field
        """
        logger.info(f"Preparing dataset for SFT: {len(dataset)} examples")

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
        Create SFTTrainer instance.

        Args:
            model: Prepared model with PEFT
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer instance
            config: Training configuration
            callbacks: Training callbacks

        Returns:
            SFTTrainer instance
        """
        logger.info("Creating SFTTrainer")

        # Handle max_seq_length = -1 (use model's maximum)
        max_seq_length = config.get("max_seq_length")
        if max_seq_length == -1:
            # Attempt to get model's maximum sequence length from config
            try:
                if hasattr(model, 'config'):
                    # Try different attribute names that models use
                    if hasattr(model.config, 'max_position_embeddings'):
                        max_seq_length = model.config.max_position_embeddings
                    elif hasattr(model.config, 'n_positions'):
                        max_seq_length = model.config.n_positions
                    elif hasattr(model.config, 'max_sequence_length'):
                        max_seq_length = model.config.max_sequence_length
                    else:
                        max_seq_length = 2048  # Fallback
                else:
                    max_seq_length = 2048  # Fallback

                logger.info(f"max_seq_length was -1, resolved to model's max: {max_seq_length}")
            except Exception as e:
                logger.warning(f"Could not determine model's max sequence length: {e}. Using 2048.")
                max_seq_length = 2048
        elif max_seq_length is not None and max_seq_length <= 0:
            # Handle other invalid values
            logger.warning(f"Invalid max_seq_length: {max_seq_length}. Using 2048.")
            max_seq_length = 2048

        # Create training arguments
        training_args = SFTConfig(
            output_dir=config.get("output_dir", "./checkpoints"),
            num_train_epochs=config.get("num_train_epochs", 1),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            optim=config.get("optim", "paged_adamw_32bit"),
            save_steps=config.get("save_steps", 0),
            logging_steps=config.get("logging_steps", 25),
            learning_rate=config.get("learning_rate", 2e-4),
            warmup_ratio=config.get("warmup_ratio", 0.03),
            weight_decay=config.get("weight_decay", 0.001),
            fp16=config.get("fp16", False),
            bf16=config.get("bf16", False),
            max_grad_norm=config.get("max_grad_norm", 0.3),
            max_steps=config.get("max_steps", -1),
            group_by_length=config.get("group_by_length", True),
            lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
            report_to="tensorboard",
            logging_dir=config.get("logging_dir", "./training_logs"),
            max_length=max_seq_length,
            packing=config.get("packing", False),
            # Evaluation settings
            eval_strategy="steps" if eval_dataset else "no",
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

        logger.info("SFTTrainer created successfully")
        return trainer

    def get_required_dataset_fields(self) -> list:
        """
        Get required dataset fields for SFT.

        Returns:
            List of required fields (varies by task)
        """
        # Note: Actual fields depend on the task type
        # This will be validated by task-specific formatters
        return ["input", "output"]
