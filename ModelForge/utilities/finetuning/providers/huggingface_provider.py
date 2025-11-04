"""
HuggingFace provider implementation for fine-tuning.

This module provides HuggingFace Transformers-based fine-tuning,
maintaining backward compatibility with existing ModelForge workflows.
"""

import os
from typing import Dict, List, Any, Tuple, Optional
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig

from .base_provider import FinetuningProvider


class HuggingFaceProgressCallback(TrainerCallback):
    """
    Callback to update global finetuning status during HuggingFace training.
    """
    
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
            f"Training step {state.global_step}/{state.max_steps}"
        )
        
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        self.global_manager.finetuning_status["progress"] = 100
        self.global_manager.finetuning_status["message"] = "Training completed!"


class HuggingFaceProvider(FinetuningProvider):
    """
    HuggingFace Transformers provider for fine-tuning.
    
    Implements the FinetuningProvider interface using HuggingFace's
    transformers, peft, and trl libraries.
    """
    
    # Task type mappings for PEFT
    TASK_TYPE_MAP = {
        "text-generation": TaskType.CAUSAL_LM,
        "summarization": TaskType.SEQ_2_SEQ_LM,
        "extractive-question-answering": TaskType.QUESTION_ANS,
    }
    
    # Model class mappings
    MODEL_CLASS_MAP = {
        "text-generation": (AutoModelForCausalLM, "AutoPeftModelForCausalLM"),
        "summarization": (AutoModelForSeq2SeqLM, "AutoPeftModelForSeq2SeqLM"),
        "extractive-question-answering": (AutoModelForCausalLM, "AutoPeftModelForCausalLM"),
    }
    
    def __init__(
        self,
        model_name: str,
        task: str,
        compute_specs: str = "low_end"
    ) -> None:
        """
        Initialize HuggingFace provider.
        
        Args:
            model_name: HuggingFace model identifier
            task: Fine-tuning task type
            compute_specs: Hardware profile
        """
        super().__init__(model_name, task, compute_specs)
        self.peft_task_type = self.TASK_TYPE_MAP.get(task)
        self.model_class, self.peft_class_name = self.MODEL_CLASS_MAP.get(task, (None, None))
        self.output_dir: Optional[str] = None
        self.fine_tuned_name: Optional[str] = None
        
    def load_model(self, **kwargs) -> Tuple[Any, Any]:
        """
        Load HuggingFace model and tokenizer with quantization.
        
        Args:
            **kwargs: Settings including quantization config
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if self.model_class is None:
            raise ValueError(f"Unsupported task type: {self.task}")
        
        # Prepare quantization config
        bits_n_bytes_config = None
        use_4bit = kwargs.get("use_4bit", False)
        use_8bit = kwargs.get("use_8bit", False)
        
        if use_4bit:
            compute_dtype = getattr(torch, kwargs.get("bnb_4bit_compute_dtype", "float16"))
            bits_n_bytes_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=kwargs.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=kwargs.get("use_nested_quant", False),
            )
        elif use_8bit:
            bits_n_bytes_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load model
        device_map = kwargs.get("device_map", {"": 0})
        
        if bits_n_bytes_config:
            model = self.model_class.from_pretrained(
                self.model_name,
                quantization_config=bits_n_bytes_config,
                device_map=device_map,
                use_cache=False,
            )
        else:
            model = self.model_class.from_pretrained(
                self.model_name,
                device_map=device_map,
                use_cache=False,
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        self.model = model
        self.tokenizer = tokenizer
        
        return model, tokenizer
    
    def prepare_dataset(self, dataset_path: str, **kwargs) -> Dataset:
        """
        Load and format dataset for HuggingFace training.
        
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
        """Format example for text generation."""
        return {
            "prompt": "USER:" + example.get("prompt", ""),
            "completion": "ASSISTANT: " + example.get("completion", "") + "<|endoftext|>"
        }
    
    def _format_summarization_example(self, example: dict, keys: List[str]) -> Dict[str, str]:
        """Format example for summarization."""
        if len(keys) < 2:
            keys = ["article", "summary"]
        return {
            "text": f'''
                ["role": "system", "content": "You are a text summarization assistant."],
                ["role": "user", "content": {example[keys[0]]}],
                ["role": "assistant", "content": {example[keys[1]]}]
            '''
        }
    
    def _format_qa_example(self, example: dict, keys: List[str]) -> Dict[str, str]:
        """Format example for question answering."""
        if len(keys) < 3:
            keys = ["context", "question", "answer"]
        return {
            "text": f'''
                ["role": "system", "content": "You are a question answering assistant."],
                ["role": "user", "content": "Context: {example[keys[0]]}\nQuestion: {example[keys[1]]}"],
                ["role": "assistant", "content": {example[keys[2]]}]
            '''
        }
    
    def train(self, **kwargs) -> str:
        """
        Execute HuggingFace fine-tuning with PEFT/LoRA.
        
        Args:
            **kwargs: Training configuration
            
        Returns:
            Path to saved model
        """
        # Ensure model and dataset are loaded
        if self.model is None or self.tokenizer is None:
            self.load_model(**kwargs)
        
        if self.dataset is None:
            raise ValueError("Dataset must be prepared before training")
        
        # Configure LoRA
        lora_config = LoraConfig(
            lora_alpha=kwargs.get("lora_alpha", 32),
            lora_dropout=kwargs.get("lora_dropout", 0.1),
            r=kwargs.get("lora_r", 16),
            bias="none",
            task_type=self.peft_task_type,
            target_modules='all-linear',
        )
        
        # Apply PEFT
        model = get_peft_model(self.model, lora_config)
        
        # Configure training
        training_args = SFTConfig(
            output_dir=self.output_dir or "./model_checkpoints",
            num_train_epochs=kwargs.get("num_train_epochs", 1),
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 4),
            optim=kwargs.get("optim", "paged_adamw_32bit"),
            save_steps=kwargs.get("save_steps", 0),
            logging_steps=kwargs.get("logging_steps", 25),
            learning_rate=kwargs.get("learning_rate", 2e-4),
            warmup_ratio=kwargs.get("warmup_ratio", 0.03),
            weight_decay=kwargs.get("weight_decay", 0.001),
            fp16=kwargs.get("fp16", False),
            bf16=kwargs.get("bf16", False),
            max_grad_norm=kwargs.get("max_grad_norm", 0.3),
            max_steps=kwargs.get("max_steps", -1),
            group_by_length=kwargs.get("group_by_length", True),
            lr_scheduler_type=kwargs.get("lr_scheduler_type", "cosine"),
            report_to="tensorboard",
            logging_dir="./training_logs",
            max_length=None,
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=self.dataset,
            args=training_args,
            callbacks=[HuggingFaceProgressCallback()],
        )
        
        # Train
        trainer.train()
        
        # Save model
        save_path = self.fine_tuned_name or self.output_dir
        trainer.model.save_pretrained(save_path)
        
        return save_path
    
    def export_model(self, output_path: str, **kwargs) -> bool:
        """
        Export the fine-tuned model (already saved during training).
        
        Args:
            output_path: Path to export the model
            **kwargs: Additional export parameters
            
        Returns:
            True if successful
        """
        # HuggingFace models are already saved during training
        # This method exists for interface compatibility
        return True
    
    def get_supported_hyperparameters(self) -> List[str]:
        """
        Return list of supported hyperparameters.
        
        Returns:
            List of hyperparameter names
        """
        return [
            "num_train_epochs",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "use_4bit",
            "use_8bit",
            "bnb_4bit_compute_dtype",
            "bnb_4bit_quant_type",
            "use_nested_quant",
            "fp16",
            "bf16",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "max_grad_norm",
            "learning_rate",
            "weight_decay",
            "optim",
            "lr_scheduler_type",
            "max_steps",
            "warmup_ratio",
            "group_by_length",
            "packing",
            "device_map",
            "max_seq_length",
        ]
    
    def validate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate HuggingFace-specific settings.
        
        Args:
            settings: Settings dictionary
            
        Returns:
            Validated settings
        """
        # Basic validation - detailed validation happens in the router
        validated = {}
        
        for key, value in settings.items():
            if key in self.get_supported_hyperparameters():
                validated[key] = value
        
        # Extract output paths if present
        if "output_dir" in settings:
            self.output_dir = settings["output_dir"]
        if "fine_tuned_name" in settings:
            self.fine_tuned_name = settings["fine_tuned_name"]
        
        return validated
    
    @classmethod
    def get_provider_name(cls) -> str:
        """Return provider name."""
        return "huggingface"
    
    @classmethod
    def get_provider_description(cls) -> str:
        """Return provider description."""
        return "HuggingFace Transformers with PEFT/LoRA fine-tuning"
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if HuggingFace dependencies are available.
        
        Returns:
            True if available
        """
        try:
            import transformers
            import peft
            import trl
            return True
        except ImportError:
            return False
