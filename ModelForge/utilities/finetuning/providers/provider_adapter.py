"""
Provider adapter for bridging router and provider implementations.

This adapter mediates between the existing router workflow and the new
provider system, ensuring consistent artifact generation and status reporting.
"""

import os
import json
from typing import Dict, Any, Optional
from datasets import load_dataset

from .provider_registry import ProviderRegistry
from .base_provider import BaseProvider
from ..Finetuner import Finetuner


class UnslothProgressCallback:
    """Progress callback for Unsloth provider with (Unsloth) annotation."""
    
    def __init__(self):
        from ....globals.globals_instance import global_manager
        self.global_manager = global_manager
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Update progress with Unsloth annotation."""
        if state.max_steps <= 0:
            return
        
        progress = min(95, int((state.global_step / state.max_steps) * 100))
        self.global_manager.finetuning_status["progress"] = progress
        self.global_manager.finetuning_status["message"] = (
            f"Training step {state.global_step}/{state.max_steps} (Unsloth)"
        )
    
    def on_train_end(self, args, state, control, **kwargs):
        """Mark training as complete."""
        self.global_manager.finetuning_status["progress"] = 100
        self.global_manager.finetuning_status["message"] = "Training completed! (Unsloth)"


class ProviderFinetuner:
    """
    Adapter class bridging router and provider system.
    
    Generates consistent output paths and delegates training to selected provider.
    """
    
    def __init__(
        self,
        provider_name: str,
        model_name: str,
        task: str,
        compute_specs: str
    ):
        """
        Initialize provider adapter.
        
        Args:
            provider_name: Name of the provider to use
            model_name: Model to fine-tune
            task: Fine-tuning task
            compute_specs: Compute profile
        """
        self.provider_name = provider_name
        self.model_name = model_name
        self.task = task
        self.compute_specs = compute_specs
        
        # Create provider instance
        self.provider = ProviderRegistry.create_provider(
            provider_name=provider_name,
            model_name=model_name,
            task=task,
            compute_specs=compute_specs
        )
        
        # Initialize paths (will be set by set_settings)
        self.output_dir: Optional[str] = None
        self.fine_tuned_name: Optional[str] = None
        self.hyperparameters: Dict[str, Any] = {}
        self.dataset_path: Optional[str] = None
    
    def set_settings(self, **kwargs) -> None:
        """
        Set training settings and generate output paths.
        
        Uses the same path generation logic as legacy Finetuner for consistency.
        """
        from ....globals.globals_instance import global_manager
        
        # Generate unique ID and paths
        uid = Finetuner.gen_uuid()
        safe_model_name = self.model_name.replace('/', '-').replace('\\', '-')
        
        # Use FileManager default directories
        default_dirs = global_manager.file_manager.return_default_dirs()
        self.fine_tuned_name = f"{default_dirs['models']}/{safe_model_name}_{uid}"
        self.output_dir = f"{default_dirs['model_checkpoints']}/{safe_model_name}_{uid}"
        
        # Validate and normalize hyperparameters through provider
        self.hyperparameters = self.provider.validate_hyperparameters(kwargs)
        
        # Store dataset path if provided
        self.dataset_path = kwargs.get('dataset')
    
    def load_dataset(self, dataset_path: str) -> None:
        """
        Load and format dataset using provider.
        
        Args:
            dataset_path: Path to dataset file
        """
        self.dataset_path = dataset_path
        
        # Load raw dataset
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        # Format using provider
        self.provider.dataset = self.provider.format_dataset(dataset)
        
        print(f"Dataset loaded and formatted with {self.provider_name} provider")
        if len(self.provider.dataset) > 0:
            print("Sample:", self.provider.dataset[0])
    
    def finetune(self) -> bool | str:
        """
        Execute fine-tuning workflow.
        
        Returns:
            Path to fine-tuned model on success, False on failure
        """
        try:
            print(f"Starting fine-tuning with {self.provider_name} provider...")
            
            # Ensure dataset is loaded
            if self.provider.dataset is None and self.dataset_path:
                self.load_dataset(self.dataset_path)
            
            # Load model
            print("Loading model...")
            self.provider.load_model(self.hyperparameters)
            
            # Prepare for training (apply LoRA, etc.)
            print("Preparing model for training...")
            self.provider.prepare_for_training(self.hyperparameters)
            
            # Create progress callback
            if self.provider_name == "unsloth":
                progress_callback = UnslothProgressCallback()
            else:
                # Use standard progress callback for other providers
                from ..Finetuner import ProgressCallback
                progress_callback = ProgressCallback()
            
            # Train
            print("Starting training...")
            self.provider.train(
                output_dir=self.output_dir,
                hyperparameters=self.hyperparameters,
                progress_callback=progress_callback
            )
            
            # Save model
            print(f"Saving model to {self.fine_tuned_name}...")
            self.provider.save_model(self.fine_tuned_name)
            
            # Build configuration file
            config_file_path = os.path.abspath(self.fine_tuned_name)
            pipeline_task = self._normalize_pipeline_task(self.task)
            model_class = self.provider.get_model_class_name()
            
            config_result = self._build_config_file(
                config_file_path,
                pipeline_task,
                model_class,
                self.provider_name
            )
            
            if not config_result:
                print("Warning: Could not create configuration file")
            
            self._report_finish()
            return self.fine_tuned_name
            
        except Exception as e:
            print(f"Fine-tuning failed with {self.provider_name} provider: {e}")
            import traceback
            traceback.print_exc()
            self._report_finish(error=True, message=str(e))
            return False
    
    def _normalize_pipeline_task(self, task: str) -> str:
        """
        Normalize task identifier for pipeline compatibility.
        
        Args:
            task: Task from settings
            
        Returns:
            Pipeline-compatible task name
        """
        task_mapping = {
            "text-generation": "text-generation",
            "summarization": "summarization",
            "extractive-question-answering": "question-answering"
        }
        return task_mapping.get(task, task)
    
    def _build_config_file(
        self,
        config_dir: str,
        pipeline_task: str,
        model_class: str,
        provider: str
    ) -> bool:
        """
        Build configuration file with provider metadata.
        
        Args:
            config_dir: Directory where config will be saved
            pipeline_task: Pipeline task identifier
            model_class: Model class name
            provider: Provider name
            
        Returns:
            True on success, False on failure
        """
        try:
            config_path = os.path.join(config_dir, "modelforge_config.json")
            config = {
                "model_class": model_class,
                "pipeline_task": pipeline_task,
                "provider": provider
            }
            
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            
            print(f"Configuration file saved to {config_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration file: {e}")
            return False
    
    def _report_finish(self, error: bool = False, message: Optional[str] = None) -> None:
        """
        Report completion or failure.
        
        Args:
            error: Whether an error occurred
            message: Error message if applicable
        """
        print("*" * 100)
        if not error:
            print(f"Model fine-tuned successfully with {self.provider_name}!")
            print(f"Model saved to {self.fine_tuned_name}")
            print("Try out your new model in our chat playground!")
        else:
            print(f"Model fine-tuning failed with {self.provider_name}!")
            if message:
                print(f"Error: {message}")
        print("*" * 100)
