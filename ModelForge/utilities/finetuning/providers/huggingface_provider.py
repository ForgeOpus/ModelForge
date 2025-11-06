"""
HuggingFace provider implementation for fine-tuning.

This provider implements the standard HuggingFace fine-tuning workflow
using transformers, peft, and trl libraries.
"""

import os
import torch
from typing import Dict, Any, Optional, List
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

from .base_provider import BaseProvider


class HuggingFaceProvider(BaseProvider):
    """
    HuggingFace provider for fine-tuning using transformers and PEFT.
    
    Supports text-generation, summarization, and extractive-question-answering tasks
    with LoRA fine-tuning and quantization options.
    """
    
    # Task to model class mapping
    TASK_TO_MODEL_CLASS = {
        "text-generation": (AutoModelForCausalLM, TaskType.CAUSAL_LM, "AutoPeftModelForCausalLM"),
        "summarization": (AutoModelForSeq2SeqLM, TaskType.SEQ_2_SEQ_LM, "AutoPeftModelForSeq2SeqLM"),
        "extractive-question-answering": (AutoModelForQuestionAnswering, TaskType.QUESTION_ANS, "AutoPeftModelForQuestionAnswering")
    }
    
    @staticmethod
    def get_provider_name() -> str:
        """Get provider identifier."""
        return "huggingface"
    
    @staticmethod
    def is_available() -> bool:
        """Check if HuggingFace dependencies are available."""
        try:
            import transformers
            import peft
            import trl
            return True
        except ImportError:
            return False
    
    def __init__(self, model_name: str, task: str, compute_specs: str):
        """Initialize HuggingFace provider."""
        super().__init__(model_name, task, compute_specs)
        
        if task not in self.TASK_TO_MODEL_CLASS:
            raise ValueError(f"Unsupported task: {task}")
        
        self.model_class, self.task_type, self.peft_model_class = self.TASK_TO_MODEL_CLASS[task]
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate hyperparameters for HuggingFace provider.
        
        HuggingFace accepts a broad set of hyperparameters with minimal restrictions.
        """
        # HuggingFace supports all standard hyperparameters
        return hyperparameters
    
    def format_dataset(self, dataset: Dataset) -> Dataset:
        """
        Format dataset according to task requirements.
        
        Args:
            dataset: Raw dataset from file
            
        Returns:
            Formatted dataset
        """
        if self.task == "text-generation":
            return self._format_text_generation(dataset)
        elif self.task == "summarization":
            return self._format_summarization(dataset)
        elif self.task == "extractive-question-answering":
            return self._format_question_answering(dataset)
        else:
            raise ValueError(f"Unsupported task: {self.task}")
    
    def _format_text_generation(self, dataset: Dataset) -> Dataset:
        """Format dataset for text generation task."""
        # Rename columns to expected format
        dataset = dataset.rename_column("input", "prompt")
        dataset = dataset.rename_column("output", "completion")
        
        def format_example(example):
            return {
                "prompt": "USER:" + example.get("prompt", ""),
                "completion": "ASSISTANT: " + example.get("completion", "") + "<|endoftext|>"
            }
        
        return dataset.map(format_example)
    
    def _format_summarization(self, dataset: Dataset) -> Dataset:
        """Format dataset for summarization task."""
        keys = dataset.column_names
        
        # Default keys if not provided
        if len(keys) < 2:
            keys = ["article", "summary"]
        
        def format_example(example):
            return {
                "text": f'''
                    ["role": "system", "content": "You are a text summarization assistant."],
                    ["role": "user", "content": {example[keys[0]]}],
                    ["role": "assistant", "content": {example[keys[1]]}]
                '''
            }
        
        formatted = dataset.map(format_example)
        return formatted.remove_columns(keys)
    
    def _format_question_answering(self, dataset: Dataset) -> Dataset:
        """Format dataset for question answering task."""
        if self.tokenizer is None:
            # Load tokenizer if not already loaded
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        
        def format_example(example):
            question = example["question"].strip()
            context = example["context"]
            answer = example["answers"]
            
            inputs = self.tokenizer(
                question,
                context,
                max_length=512,
                truncation="only_second",
                return_offsets_mapping=True,
                padding="max_length",
            )
            
            offset_mapping = inputs.pop("offset_mapping")
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(0)
            
            context_start = 0
            while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
                context_start += 1
            context_end = len(sequence_ids) - 1
            while context_end >= 0 and sequence_ids[context_end] != 1:
                context_end -= 1
            
            start_position = 0
            end_position = 0
            
            if not (offset_mapping[context_start][0] > end_char or
                    offset_mapping[context_end][1] < start_char):
                idx = context_start
                while idx <= context_end and offset_mapping[idx][0] <= start_char:
                    idx += 1
                start_position = idx - 1
                
                idx = context_end
                while idx >= context_start and offset_mapping[idx][1] >= end_char:
                    idx -= 1
                end_position = idx + 1
            
            inputs["start_positions"] = start_position
            inputs["end_positions"] = end_position
            return inputs
        
        return dataset.map(format_example, remove_columns=dataset.column_names)
    
    def load_model(self, hyperparameters: Dict[str, Any]) -> None:
        """Load model and tokenizer with quantization if specified."""
        # Prepare quantization config
        bits_n_bytes_config = None
        if hyperparameters.get("use_4bit", False):
            compute_dtype = getattr(torch, hyperparameters.get("bnb_4bit_compute_dtype", "float16"))
            bits_n_bytes_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=hyperparameters.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=hyperparameters.get("use_nested_quant", False),
            )
        elif hyperparameters.get("use_8bit", False):
            bits_n_bytes_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load model
        model_kwargs = {
            "device_map": hyperparameters.get("device_map", "auto"),
            "use_cache": False,
        }
        
        if bits_n_bytes_config is not None:
            model_kwargs["quantization_config"] = bits_n_bytes_config
        
        self.model = self.model_class.from_pretrained(self.model_name, **model_kwargs)
        
        # Load tokenizer if not already loaded (for QA task)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
    
    def prepare_for_training(self, hyperparameters: Dict[str, Any]) -> Any:
        """Apply LoRA adapters to the model."""
        peft_config = LoraConfig(
            lora_alpha=hyperparameters.get("lora_alpha", 16),
            lora_dropout=hyperparameters.get("lora_dropout", 0.1),
            r=hyperparameters.get("lora_r", 8),
            bias="none",
            task_type=self.task_type,
            target_modules='all-linear',
        )
        
        self.model = get_peft_model(self.model, peft_config)
        return self.model
    
    def train(
        self,
        output_dir: str,
        hyperparameters: Dict[str, Any],
        progress_callback: Optional[Any] = None
    ) -> str:
        """Execute training with HuggingFace Trainer."""
        callbacks = [progress_callback] if progress_callback else []
        
        if self.task == "extractive-question-answering":
            # Use standard Trainer for QA
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=hyperparameters.get("num_train_epochs", 3),
                per_device_train_batch_size=hyperparameters.get("per_device_train_batch_size", 1),
                gradient_accumulation_steps=hyperparameters.get("gradient_accumulation_steps", 1),
                optim=hyperparameters.get("optim", "paged_adamw_32bit"),
                save_steps=hyperparameters.get("save_steps", 0),
                logging_steps=hyperparameters.get("logging_steps", 25),
                learning_rate=hyperparameters.get("learning_rate", 2e-4),
                warmup_ratio=hyperparameters.get("warmup_ratio", 0.03),
                weight_decay=hyperparameters.get("weight_decay", 0.001),
                fp16=hyperparameters.get("fp16", False),
                bf16=hyperparameters.get("bf16", False),
                max_grad_norm=hyperparameters.get("max_grad_norm", 0.3),
                group_by_length=hyperparameters.get("group_by_length", True),
                lr_scheduler_type=hyperparameters.get("lr_scheduler_type", "constant"),
                report_to="tensorboard",
                logging_dir=hyperparameters.get("logging_dir", "./training_logs"),
            )
            
            trainer = Trainer(
                model=self.model,
                train_dataset=self.dataset,
                args=training_args,
                callbacks=callbacks,
            )
        else:
            # Use SFTTrainer for text generation and summarization
            training_args = SFTConfig(
                output_dir=output_dir,
                num_train_epochs=hyperparameters.get("num_train_epochs", 3),
                per_device_train_batch_size=hyperparameters.get("per_device_train_batch_size", 1),
                gradient_accumulation_steps=hyperparameters.get("gradient_accumulation_steps", 1),
                optim=hyperparameters.get("optim", "paged_adamw_32bit"),
                save_steps=hyperparameters.get("save_steps", 0),
                logging_steps=hyperparameters.get("logging_steps", 25),
                learning_rate=hyperparameters.get("learning_rate", 2e-4),
                warmup_ratio=hyperparameters.get("warmup_ratio", 0.03),
                weight_decay=hyperparameters.get("weight_decay", 0.001),
                fp16=hyperparameters.get("fp16", False),
                bf16=hyperparameters.get("bf16", False),
                max_grad_norm=hyperparameters.get("max_grad_norm", 0.3),
                max_steps=hyperparameters.get("max_steps", -1),
                group_by_length=hyperparameters.get("group_by_length", True),
                lr_scheduler_type=hyperparameters.get("lr_scheduler_type", "constant"),
                report_to="tensorboard",
                logging_dir=hyperparameters.get("logging_dir", "./training_logs"),
                max_length=None,
            )
            
            trainer = SFTTrainer(
                model=self.model,
                train_dataset=self.dataset,
                args=training_args,
                callbacks=callbacks,
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
        """Get the PEFT model class name."""
        return self.peft_model_class
