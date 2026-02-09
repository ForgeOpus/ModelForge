"""
Supervised Fine-Tuning (SFT) strategy implementation.
Uses TRL's SFTTrainer for standard supervised fine-tuning with LoRA.
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


class SFTStrategy:
    """Supervised Fine-Tuning strategy using TRL's SFTTrainer."""

    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "sft"

    def prepare_model(self, model: Any, config: Dict) -> Any:
        """
        Prepare model for SFT. When using SFTTrainer with peft_config,
        the trainer handles PEFT wrapping automatically. For the HuggingFace
        provider path, we just prepare for kbit training if quantized.

        Args:
            model: Base model instance
            config: Configuration with LoRA settings

        Returns:
            Model prepared for training
        """
        logger.info("Preparing model for SFT")

        # If quantized, prepare for kbit training
        if config.get("use_4bit") or config.get("use_8bit"):
            model = prepare_model_for_kbit_training(model)

        logger.info("Model prepared for SFT")
        return model

    def get_peft_config(self, config: Dict) -> LoraConfig:
        """
        Create the LoRA PEFT config for SFTTrainer.

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

        peft_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.1),
            bias="none",
            task_type=task_type,
            target_modules=config.get("target_modules", "all-linear"),
        )

        logger.info(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")
        return peft_config

    def prepare_dataset(self, dataset: Any, tokenizer: Any, config: Dict) -> Any:
        """
        Prepare dataset for SFT. SFTTrainer handles tokenization automatically,
        so we just need to format the data into the expected structure.

        Args:
            dataset: Pre-formatted dataset with task-specific fields
            tokenizer: Tokenizer instance
            config: Configuration dictionary

        Returns:
            Dataset formatted for SFTTrainer (with 'text' field)
        """
        logger.info(f"Preparing dataset for SFT: {len(dataset)} examples")

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

        logger.info(f"Dataset formatted for SFT: {len(dataset)} examples")
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

        SFTTrainer handles tokenization, data collation, and label creation
        automatically from the text field.

        Args:
            model: Prepared model
            train_dataset: Formatted training dataset (with 'text' field)
            eval_dataset: Formatted evaluation dataset
            tokenizer: Tokenizer instance
            config: Training configuration
            callbacks: Training callbacks

        Returns:
            SFTTrainer instance
        """
        logger.info("Creating SFTTrainer")

        max_seq_length = config.get("max_seq_length", 2048)
        if max_seq_length == -1:
            max_seq_length = 2048

        # Create SFTConfig with training arguments
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

        logger.info("SFTTrainer created successfully")
        return trainer

    def get_required_dataset_fields(self) -> list:
        """
        Get required dataset fields for SFT.

        Returns:
            List of required fields (varies by task)
        """
        return ["input", "output"]
