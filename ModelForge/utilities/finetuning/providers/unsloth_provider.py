"""
Unsloth AI provider implementation for optimized fine-tuning.

This provider implements Unsloth's optimized fine-tuning workflow
with efficient memory usage and faster training.
"""
try:
    import unsloth
except ImportError:
    pass  # Unsloth is an optional dependency
import os
from typing import Dict, Any, Optional, List
from datasets import Dataset

from .base_provider import BaseProvider


class UnslothProvider(BaseProvider):
    """
    Unsloth provider for optimized fine-tuning.
    
    Uses Unsloth's FastLanguageModel for efficient training with 4-bit quantization
    and optimized LoRA implementation.
    """
    
    # Optimizer mapping for Unsloth compatibility
    OPTIMIZER_MAPPING = {
        "paged_adamw_32bit": "adamw_torch",
        "paged_adamw_8bit": "adamw_8bit",
        "adamw_hf": "adamw_torch",
        "adamw_torch": "adamw_torch",
        "adamw_8bit": "adamw_8bit",
    }
    
    @staticmethod
    def get_provider_name() -> str:
        """Get provider identifier."""
        return "unsloth"
    
    @staticmethod
    def is_available() -> bool:
        """Check if Unsloth dependencies are available."""
        try:
            from unsloth import FastLanguageModel
            return True
        except ImportError:
            return False
    
    @staticmethod
    def get_supported_tasks() -> List[str]:
        """Unsloth primarily supports text generation tasks."""
        return ["text-generation"]
    
    def __init__(self, model_name: str, task: str, compute_specs: str):
        """Initialize Unsloth provider."""
        super().__init__(model_name, task, compute_specs)
        
        # Unsloth currently best supports text-generation
        if task not in self.get_supported_tasks():
            raise ValueError(
                f"Unsloth provider currently supports {self.get_supported_tasks()}, "
                f"but got '{task}'"
            )
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize hyperparameters for Unsloth.
        
        Unsloth has specific requirements and optimizations.
        """
        validated = hyperparameters.copy()
        
        # Normalize optimizer
        optimizer = validated.get("optim", "adamw_8bit")
        validated["optim"] = self.OPTIMIZER_MAPPING.get(optimizer, "adamw_8bit")
        
        # Normalize max_seq_length (Unsloth requires explicit value)
        max_seq_length = validated.get("max_seq_length", -1)
        if max_seq_length is None or max_seq_length <= 0:
            validated["max_seq_length"] = 2048
            print(f"Unsloth: Setting max_seq_length to default 2048")
        
        # Warn about unsupported features
        if validated.get("use_8bit", False):
            print("Warning: Unsloth provider currently focuses on 4-bit quantization. 8-bit flag will be ignored.")
            validated["use_8bit"] = False
            validated["use_4bit"] = True
        
        return validated
    
    def format_dataset(self, dataset: Dataset) -> Dataset:
        """
        Format dataset for Unsloth training.
        
        Unsloth uses a unified "text" field with role markup.
        """
        if self.task == "text-generation":
            return self._format_text_generation(dataset)
        else:
            raise ValueError(f"Unsupported task for Unsloth: {self.task}")
    
    def _format_text_generation(self, dataset: Dataset) -> Dataset:
        """
        Format dataset for text generation with Unsloth.
        
        Converts to single text field with role headers.
        """
        # Rename columns if needed
        if "input" in dataset.column_names:
            dataset = dataset.rename_column("input", "prompt")
        if "output" in dataset.column_names:
            dataset = dataset.rename_column("output", "completion")
        
        def format_example(example):
            prompt = example.get("prompt", "")
            completion = example.get("completion", "")
            
            # Unsloth format with role markup
            formatted_text = (
                f"### User:\n{prompt}\n\n"
                f"### Assistant:\n{completion}<|endoftext|>"
            )
            
            return {"text": formatted_text}
        
        formatted = dataset.map(format_example)
        
        # Remove original columns, keep only 'text'
        columns_to_remove = [col for col in formatted.column_names if col != "text"]
        if columns_to_remove:
            formatted = formatted.remove_columns(columns_to_remove)
        
        return formatted
    
    def load_model(self, hyperparameters: Dict[str, Any]) -> None:
        """Load model using Unsloth's FastLanguageModel."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Please install it with: "
                "pip install unsloth"
            )
        
        max_seq_length = hyperparameters.get("max_seq_length", 2048)
        
        # Unsloth's optimized loading with 4-bit quantization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect optimal dtype
            load_in_4bit=hyperparameters.get("use_4bit", True),
        )
        
        self.model = model
        self.tokenizer = tokenizer
    
    def prepare_for_training(self, hyperparameters: Dict[str, Any]) -> Any:
        """Prepare model with Unsloth's optimized PEFT configuration."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("Unsloth is not installed")
        
        # Apply Unsloth's optimized LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=hyperparameters.get("lora_r", 16),
            lora_alpha=hyperparameters.get("lora_alpha", 16),
            lora_dropout=hyperparameters.get("lora_dropout", 0),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            use_gradient_checkpointing=hyperparameters.get("gradient_checkpointing", True),
            random_state=3407,
        )
        
        return self.model
    
    def train(
        self,
        output_dir: str,
        hyperparameters: Dict[str, Any],
        progress_callback: Optional[Any] = None
    ) -> str:
        """Execute training with Unsloth's SFTTrainer."""
        try:
            from trl import SFTTrainer
            from transformers import TrainingArguments
        except ImportError:
            raise ImportError("TRL is required for Unsloth training")
        
        callbacks = [progress_callback] if progress_callback else []
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=hyperparameters.get("num_train_epochs", 3),
            per_device_train_batch_size=hyperparameters.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=hyperparameters.get("gradient_accumulation_steps", 4),
            optim=hyperparameters.get("optim", "adamw_8bit"),
            warmup_ratio=hyperparameters.get("warmup_ratio", 0.03),
            learning_rate=hyperparameters.get("learning_rate", 2e-4),
            fp16=not hyperparameters.get("bf16", False),
            bf16=hyperparameters.get("bf16", False),
            logging_steps=hyperparameters.get("logging_steps", 1),
            save_steps=hyperparameters.get("save_steps", 0),
            weight_decay=hyperparameters.get("weight_decay", 0.01),
            lr_scheduler_type=hyperparameters.get("lr_scheduler_type", "linear"),
            seed=3407,
            report_to="tensorboard",
            logging_dir=hyperparameters.get("logging_dir", "./training_logs"),
        )
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=hyperparameters.get("max_seq_length", 2048),
            dataset_num_proc=2,
            packing=hyperparameters.get("packing", False),
            args=training_args,
            callbacks=callbacks,
            fp16=False,
            bf16=True
        )
        
        # Train the model
        trainer.train()
        
        return output_dir
    
    def save_model(self, save_path: str) -> None:
        """Save the fine-tuned model."""
        self.model.save_pretrained(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
    
    def get_model_class_name(self) -> str:
        """Get the model class name for config."""
        return "AutoPeftModelForCausalLM"
