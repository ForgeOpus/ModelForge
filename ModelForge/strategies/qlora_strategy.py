"""
QLoRA (Quantized Low-Rank Adaptation) strategy implementation.
QLoRA combines 4-bit quantization with LoRA for memory-efficient fine-tuning.
"""
from typing import Any, Dict
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Import unsloth first to prevent EOS token corruption
# This must come before transformers imports to ensure proper tokenizer initialization
try:
    import unsloth
except ImportError:
    pass

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

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
        Prepare dataset for QLoRA by tokenizing text and creating labels.

        Args:
            dataset: Pre-formatted dataset with task-specific fields
            tokenizer: Tokenizer instance
            config: Configuration dictionary (contains task type, max_seq_length)

        Returns:
            Dataset with tokenized fields: input_ids, attention_mask, labels
        """
        logger.info(f"Preparing dataset for QLoRA: {len(dataset)} examples")

        # Get EOS token with SEP fallback
        eos_token = tokenizer.eos_token or tokenizer.sep_token or ""
        task = config.get("task", "text-generation")
        max_seq_length = config.get("max_seq_length", 2048)

        # Handle max_seq_length = -1 (use model's maximum)
        if max_seq_length == -1:
            max_seq_length = 2048  # Fallback default

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

        # Step 1: Create text field
        dataset = dataset.map(
            create_text_field,
            remove_columns=dataset.column_names,
            num_proc=1
        )

        # Step 2: Tokenize text
        def tokenize_function(examples):
            """Tokenize text and create labels for causal LM."""
            # Tokenize with truncation and padding
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",  # Pad to max_length for consistency
                return_tensors=None,  # Return lists, not tensors (datasets handles this)
            )

            # For causal LM: labels = input_ids
            # The model will shift internally for next-token prediction
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Apply tokenization
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],  # Remove text field, keep only tokenized
            num_proc=1,
        )

        logger.info(f"Dataset tokenized: {len(dataset)} examples with max_length={max_seq_length}")
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
        Create Trainer with QLoRA-specific optimizations.

        Args:
            model: Prepared model with QLoRA
            train_dataset: Tokenized training dataset
            eval_dataset: Tokenized evaluation dataset
            tokenizer: Tokenizer instance
            config: Training configuration
            callbacks: Training callbacks

        Returns:
            Trainer instance
        """
        logger.info("Creating Trainer with QLoRA optimizations")

        # QLoRA-optimized training arguments
        training_args = TrainingArguments(
            output_dir=config.get("output_dir", "./checkpoints"),
            num_train_epochs=config.get("num_train_epochs", 1),
            # QLoRA can use larger batch sizes due to memory efficiency
            per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            optim=config.get("optim", "adamw_torch"),
            save_steps=config.get("save_steps", 0),
            logging_steps=config.get("logging_steps", 25),
            # QLoRA can often use higher learning rates
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
            # Gradient checkpointing for memory efficiency
            gradient_checkpointing=config.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # Evaluation settings
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=config.get("eval_steps", 100),
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            # Disable distributed training for Unsloth (required when using device_map='auto')
            ddp_find_unused_parameters=False,
            use_cache=False,
        )

        # Create data collator for causal language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM
        )

        # Create standard Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
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
