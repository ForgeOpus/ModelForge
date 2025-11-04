"""
Adapter for integrating provider-based finetuning with existing infrastructure.

This module provides a bridge between the new provider system and the
existing Finetuner-based workflow to maintain backward compatibility.
"""

import json
import os
from typing import Optional
from datasets import Dataset

from .providers import get_provider, ProviderRegistry
from ...globals.globals_instance import global_manager


class ProviderFinetuner:
    """
    Adapter class that uses the provider system while maintaining
    compatibility with the existing finetuning workflow.
    """
    
    def __init__(
        self,
        model_name: str,
        task: str,
        provider: str = "huggingface",
        compute_specs: str = "low_end"
    ) -> None:
        """
        Initialize provider-based finetuner.
        
        Args:
            model_name: Model identifier
            task: Task type (text-generation, summarization, extractive-question-answering)
            provider: Provider name (huggingface, unsloth)
            compute_specs: Hardware profile
        """
        self.model_name = model_name
        self.task = task
        self.provider_name = provider
        self.compute_specs = compute_specs
        
        # Get provider instance
        self.provider = get_provider(
            provider_name=provider,
            model_name=model_name,
            task=task,
            compute_specs=compute_specs
        )
        
        # Settings that will be set by set_settings()
        self.output_dir: Optional[str] = None
        self.fine_tuned_name: Optional[str] = None
        self.dataset_path: Optional[str] = None
        
        # Map task to pipeline task string
        self.pipeline_task = self._map_task_to_pipeline(task)
        
    def _map_task_to_pipeline(self, task: str) -> str:
        """Map task to pipeline task string."""
        task_map = {
            "text-generation": "text-generation",
            "summarization": "summarization",
            "extractive-question-answering": "question-answering",
        }
        return task_map.get(task, task)
    
    def set_settings(self, **kwargs) -> None:
        """
        Set training settings for the provider.
        
        Args:
            **kwargs: Settings dictionary
        """
        # Generate output paths using existing logic
        from .Finetuner import Finetuner
        
        uid = Finetuner.gen_uuid()
        safe_model_name = self.model_name.replace('/', '-').replace('\\', '-')
        
        # Use FileManager default directories
        default_dirs = global_manager.file_manager.return_default_dirs()
        self.fine_tuned_name = f"{default_dirs['models']}/{safe_model_name}_{uid}"
        self.output_dir = f"{default_dirs['model_checkpoints']}/{safe_model_name}_{uid}"
        
        # Add output paths to settings
        kwargs["output_dir"] = self.output_dir
        kwargs["fine_tuned_name"] = self.fine_tuned_name
        
        # Set provider settings
        self.provider.set_settings(**kwargs)
        
    def load_dataset(self, dataset_path: str) -> None:
        """
        Load and prepare the dataset.
        
        Args:
            dataset_path: Path to dataset file
        """
        self.dataset_path = dataset_path
        self.provider.prepare_dataset(dataset_path)
        
    def finetune(self) -> bool | str:
        """
        Execute the fine-tuning process.
        
        Returns:
            Path to saved model if successful, False otherwise
        """
        try:
            # Load model
            self.provider.load_model(**self.provider.settings)
            
            # Ensure dataset is loaded
            if self.provider.dataset is None:
                if self.dataset_path:
                    self.load_dataset(self.dataset_path)
                else:
                    raise ValueError("Dataset must be loaded before training")
            
            # Train
            model_path = self.provider.train(**self.provider.settings)
            
            # Build config file for playground compatibility
            config_file_result = self._build_config_file(
                model_path,
                self.pipeline_task
            )
            
            if not config_file_result:
                print("Warning: Failed to create config file. Model may not work in playground.")
            
            # Report finish
            self._report_finish()
            
            return model_path
            
        except Exception as e:
            print(f"Fine-tuning failed: {e}")
            self._report_finish(error=True, message=str(e))
            return False
    
    def _build_config_file(self, config_dir: str, pipeline_task: str) -> bool:
        """
        Build configuration file for the fine-tuned model.
        
        Args:
            config_dir: Directory to save config
            pipeline_task: Pipeline task string
            
        Returns:
            True if successful
        """
        # Determine model class based on provider and task
        model_class_map = {
            "huggingface": {
                "text-generation": "AutoPeftModelForCausalLM",
                "summarization": "AutoPeftModelForSeq2SeqLM",
                "question-answering": "AutoPeftModelForCausalLM",
            },
            "unsloth": {
                "text-generation": "AutoPeftModelForCausalLM",
                "summarization": "AutoPeftModelForSeq2SeqLM",
                "question-answering": "AutoPeftModelForCausalLM",
            }
        }
        
        model_class = model_class_map.get(self.provider_name, {}).get(
            pipeline_task,
            "AutoPeftModelForCausalLM"
        )
        
        try:
            config_path = os.path.join(config_dir, "modelforge_config.json")
            with open(config_path, "w") as f:
                config = {
                    "model_class": model_class,
                    "pipeline_task": pipeline_task,
                    "provider": self.provider_name,
                }
                json.dump(config, f, indent=4)
            print(f"Configuration file saved to {config_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration file: {e}")
            return False
    
    def _report_finish(self, error: bool = False, message: Optional[str] = None) -> None:
        """
        Report completion of fine-tuning.
        
        Args:
            error: True if an error occurred
            message: Error message if applicable
        """
        print("*" * 100)
        if not error:
            print(f"Model fine-tuned successfully using {self.provider_name}!")
            print(f"Model saved to {self.fine_tuned_name}")
            print("Try out your new model in our chat playground!")
        else:
            print("Model fine-tuning failed!")
            print(f"Error: {message}")
        print("*" * 100)
