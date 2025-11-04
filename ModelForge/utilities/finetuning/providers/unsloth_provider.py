"""
Unsloth AI provider implementation for fine-tuning.

This module provides Unsloth-based fine-tuning with optimized
memory usage and faster training speeds compared to standard HuggingFace.
"""

import os
from typing import Dict, List, Any, Tuple, Optional
from datasets import Dataset, load_dataset

from .base_provider import FinetuningProvider


class UnslothProvider(FinetuningProvider):
    """
    Unsloth AI provider for optimized fine-tuning.
    
    Implements the FinetuningProvider interface using Unsloth's
    optimized training infrastructure for faster and more memory-efficient
    fine-tuning of large language models.
    """
    
    # Task type mappings for Unsloth
    TASK_TYPE_MAP = {
        "text-generation": "causal",
        "summarization": "seq2seq",
        "extractive-question-answering": "causal",
    }
    
    def __init__(
        self,
        model_name: str,
        task: str,
        compute_specs: str = "low_end"
    ) -> None:
        """
        Initialize Unsloth provider.
        
        Args:
            model_name: Model identifier (HuggingFace format)
            task: Fine-tuning task type
            compute_specs: Hardware profile
        """
        super().__init__(model_name, task, compute_specs)
        self.unsloth_task = self.TASK_TYPE_MAP.get(task)
        self.output_dir: Optional[str] = None
        self.fine_tuned_name: Optional[str] = None
        
    def load_model(self, **kwargs) -> Tuple[Any, Any]:
        """
        Load model using Unsloth's FastLanguageModel.
        
        Args:
            **kwargs: Settings including max_seq_length, quantization, etc.
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Install it with: "
                "pip install unsloth"
            )
        
        # Determine quantization settings
        load_in_4bit = kwargs.get("use_4bit", False) or kwargs.get("load_in_4bit", False)
        load_in_8bit = kwargs.get("use_8bit", False) or kwargs.get("load_in_8bit", False)
        
        # Unsloth-specific parameters
        max_seq_length = kwargs.get("max_seq_length", 2048)
        if max_seq_length == -1 or max_seq_length is None:
            max_seq_length = 2048  # Unsloth default
        
        dtype = None  # Auto-detect
        if kwargs.get("bf16", False):
            import torch
            dtype = torch.bfloat16
        elif kwargs.get("fp16", False):
            import torch
            dtype = torch.float16
        
        # Load model with Unsloth optimizations
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        self.model = model
        self.tokenizer = tokenizer
        
        return model, tokenizer
    
    def prepare_dataset(self, dataset_path: str, **kwargs) -> Dataset:
        """
        Load and format dataset for Unsloth training.
        
        Args:
            dataset_path: Path to dataset file
            **kwargs: Additional dataset preparation parameters
            
        Returns:
            Formatted dataset
        """
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        # Format based on task type
        if self.task == "text-generation":
            dataset = dataset.rename_column("input", "prompt")
            dataset = dataset.rename_column("output", "completion")
            dataset = dataset.map(self._format_text_generation_example)
        elif self.task == "summarization":
            keys = dataset.column_names
            dataset = dataset.map(lambda x: self._format_summarization_example(x, keys))
            dataset = dataset.remove_columns(keys)
        elif self.task == "extractive-question-answering":
            keys = dataset.column_names
            dataset = dataset.map(lambda x: self._format_qa_example(x, keys))
            dataset = dataset.remove_columns(keys)
        
        self.dataset = dataset
        return dataset
    
    def _format_text_generation_example(self, example: dict) -> Dict[str, str]:
        """Format example for text generation with Unsloth."""
        # Unsloth uses a more standard chat format
        return {
            "text": f"### User:\n{example.get('prompt', '')}\n\n### Assistant:\n{example.get('completion', '')}<|endoftext|>"
        }
    
    def _format_summarization_example(self, example: dict, keys: List[str]) -> Dict[str, str]:
        """Format example for summarization with Unsloth."""
        if len(keys) < 2:
            keys = ["article", "summary"]
        return {
            "text": f"### Article:\n{example[keys[0]]}\n\n### Summary:\n{example[keys[1]]}<|endoftext|>"
        }
    
    def _format_qa_example(self, example: dict, keys: List[str]) -> Dict[str, str]:
        """Format example for question answering with Unsloth."""
        if len(keys) < 3:
            keys = ["context", "question", "answer"]
        return {
            "text": f"### Context:\n{example[keys[0]]}\n\n### Question:\n{example[keys[1]]}\n\n### Answer:\n{example[keys[2]]}<|endoftext|>"
        }
    
    def train(self, **kwargs) -> str:
        """
        Execute Unsloth fine-tuning with optimized LoRA/QLoRA.
        
        Args:
            **kwargs: Training configuration
            
        Returns:
            Path to saved model
        """
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments, TrainerCallback
        except ImportError as e:
            raise ImportError(f"Required package not available: {e}")
        
        # Ensure model and dataset are loaded
        if self.model is None or self.tokenizer is None:
            self.load_model(**kwargs)
        
        if self.dataset is None:
            raise ValueError("Dataset must be prepared before training")
        
        # Apply Unsloth PEFT (optimized LoRA)
        model = FastLanguageModel.get_peft_model(
            self.model,
            r=kwargs.get("lora_r", 16),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=kwargs.get("lora_alpha", 32),
            lora_dropout=kwargs.get("lora_dropout", 0.1),
            bias="none",
            use_gradient_checkpointing=kwargs.get("gradient_checkpointing", True),
            random_state=3407,
            use_rslora=False,  # Rank stabilized LoRA
            loftq_config=None,  # LoftQ quantization
        )
        
        # Progress callback
        class UnslothProgressCallback(TrainerCallback):
            """Callback to update global finetuning status during Unsloth training."""
            
            def __init__(self):
                super().__init__()
                from ....globals.globals_instance import global_manager
                self.global_manager = global_manager
                
            def on_log(self, args, state, control, logs=None, **kwargs):
                """Called when logging happens during training."""
                if state.max_steps <= 0:
                    return
                
                progress = min(95, int((state.global_step / state.max_steps) * 100))
                self.global_manager.finetuning_status["progress"] = progress
                self.global_manager.finetuning_status["message"] = (
                    f"Training step {state.global_step}/{state.max_steps} (Unsloth)"
                )
                
            def on_train_end(self, args, state, control, **kwargs):
                """Called at the end of training."""
                self.global_manager.finetuning_status["progress"] = 100
                self.global_manager.finetuning_status["message"] = "Training completed with Unsloth!"
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir or "./model_checkpoints",
            num_train_epochs=kwargs.get("num_train_epochs", 1),
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 4),
            warmup_steps=int(kwargs.get("warmup_ratio", 0.03) * 100),  # Approximate
            learning_rate=kwargs.get("learning_rate", 2e-4),
            fp16=kwargs.get("fp16", False),
            bf16=kwargs.get("bf16", False),
            logging_steps=kwargs.get("logging_steps", 1),
            optim=kwargs.get("optim", "adamw_8bit"),
            weight_decay=kwargs.get("weight_decay", 0.01),
            lr_scheduler_type=kwargs.get("lr_scheduler_type", "linear"),
            seed=3407,
            save_steps=kwargs.get("save_steps", 0),
            max_steps=kwargs.get("max_steps", -1),
            report_to="tensorboard",
            logging_dir="./training_logs",
        )
        
        # Create SFT trainer with Unsloth optimizations
        trainer = SFTTrainer(
            model=model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=kwargs.get("max_seq_length", 2048) if kwargs.get("max_seq_length", 2048) != -1 else 2048,
            dataset_num_proc=2,
            packing=kwargs.get("packing", False),
            args=training_args,
            callbacks=[UnslothProgressCallback()],
        )
        
        # Train with Unsloth optimizations
        trainer.train()
        
        # Save model
        save_path = self.fine_tuned_name or self.output_dir
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        return save_path
    
    def export_model(self, output_path: str, **kwargs) -> bool:
        """
        Export the fine-tuned Unsloth model.
        
        Supports multiple export formats including HuggingFace, GGUF, etc.
        
        Args:
            output_path: Path to export the model
            **kwargs: Export parameters (format, quantization_method, etc.)
            
        Returns:
            True if successful
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            return False
        
        export_format = kwargs.get("export_format", "huggingface")
        
        if export_format == "huggingface":
            # Already saved in HuggingFace format during training
            return True
        elif export_format == "gguf":
            # Export to GGUF format for llama.cpp
            if self.model is None:
                return False
            
            quantization_method = kwargs.get("quantization_method", "q4_k_m")
            self.model.save_pretrained_gguf(
                output_path,
                self.tokenizer,
                quantization_method=quantization_method
            )
            return True
        else:
            # Unsupported export format
            return False
    
    def get_supported_hyperparameters(self) -> List[str]:
        """
        Return list of Unsloth-supported hyperparameters.
        
        Returns:
            List of hyperparameter names
        """
        return [
            # Standard hyperparameters
            "num_train_epochs",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "use_4bit",
            "load_in_4bit",
            "fp16",
            "bf16",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "learning_rate",
            "weight_decay",
            "optim",
            "lr_scheduler_type",
            "max_steps",
            "warmup_ratio",
            "packing",
            "max_seq_length",
            # Unsloth-specific
            "use_rslora",
            "use_gradient_checkpointing",
        ]
    
    def validate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Unsloth-specific settings.
        
        Args:
            settings: Settings dictionary
            
        Returns:
            Validated settings
        """
        validated = {}
        
        for key, value in settings.items():
            if key in self.get_supported_hyperparameters():
                validated[key] = value
        
        # Extract output paths if present
        if "output_dir" in settings:
            self.output_dir = settings["output_dir"]
        if "fine_tuned_name" in settings:
            self.fine_tuned_name = settings["fine_tuned_name"]
        
        # Unsloth-specific validations
        if "max_seq_length" in validated:
            if validated["max_seq_length"] == -1 or validated["max_seq_length"] is None:
                validated["max_seq_length"] = 2048
        
        # Unsloth works best with specific optimizers
        if "optim" in validated:
            # Map to Unsloth-compatible optimizers
            optimizer_map = {
                "paged_adamw_32bit": "adamw_8bit",
                "paged_adamw_8bit": "adamw_8bit",
                "adamw_torch": "adamw_torch",
                "adamw_hf": "adamw_torch",
            }
            validated["optim"] = optimizer_map.get(validated["optim"], "adamw_8bit")
        
        return validated
    
    @classmethod
    def get_provider_name(cls) -> str:
        """Return provider name."""
        return "unsloth"
    
    @classmethod
    def get_provider_description(cls) -> str:
        """Return provider description."""
        return "Unsloth AI - 2x faster fine-tuning with reduced memory usage"
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if Unsloth dependencies are available.
        
        Returns:
            True if Unsloth is installed
        """
        try:
            import unsloth
            return True
        except ImportError:
            return False
