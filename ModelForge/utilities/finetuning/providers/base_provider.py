"""
Base provider interface for fine-tuning backends.

This module defines the abstract base class that all fine-tuning providers
must implement to ensure consistent behavior across different backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datasets import Dataset


class BaseProvider(ABC):
    """
    Abstract base class for fine-tuning providers.
    
    All provider implementations must inherit from this class and implement
    all abstract methods to ensure compatibility with the provider system.
    """
    
    def __init__(self, model_name: str, task: str, compute_specs: str):
        """
        Initialize the provider.
        
        Args:
            model_name: Name of the model to fine-tune
            task: Task type (text-generation, summarization, extractive-question-answering)
            compute_specs: Compute profile (low_end, mid_range, high_end)
        """
        self.model_name = model_name
        self.task = task
        self.compute_specs = compute_specs
        self.dataset: Optional[Dataset] = None
        self.model = None
        self.tokenizer = None
    
    @staticmethod
    @abstractmethod
    def get_provider_name() -> str:
        """
        Get the unique identifier for this provider.
        
        Returns:
            Provider name (e.g., 'huggingface', 'unsloth')
        """
        pass
    
    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """
        Check if this provider is available in the current environment.
        
        Returns:
            True if the provider's dependencies are installed, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize hyperparameters for this provider.
        
        Args:
            hyperparameters: Raw hyperparameters from settings
            
        Returns:
            Validated and normalized hyperparameters
            
        Raises:
            ValueError: If hyperparameters are invalid for this provider
        """
        pass
    
    @abstractmethod
    def format_dataset(self, dataset: Dataset) -> Dataset:
        """
        Format dataset according to provider and task requirements.
        
        Args:
            dataset: Raw dataset loaded from file
            
        Returns:
            Formatted dataset ready for training
        """
        pass
    
    @abstractmethod
    def load_model(self, hyperparameters: Dict[str, Any]) -> None:
        """
        Load the model and tokenizer with provider-specific configuration.
        
        Args:
            hyperparameters: Validated hyperparameters including quantization settings
        """
        pass
    
    @abstractmethod
    def prepare_for_training(self, hyperparameters: Dict[str, Any]) -> Any:
        """
        Prepare model for training (e.g., apply LoRA adapters).
        
        Args:
            hyperparameters: Validated hyperparameters including LoRA settings
            
        Returns:
            Configured model ready for training
        """
        pass
    
    @abstractmethod
    def train(
        self,
        output_dir: str,
        hyperparameters: Dict[str, Any],
        progress_callback: Optional[Any] = None
    ) -> str:
        """
        Execute the training process.
        
        Args:
            output_dir: Directory for training checkpoints
            hyperparameters: Validated hyperparameters for training
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the fine-tuned model
        """
        pass
    
    @abstractmethod
    def save_model(self, save_path: str) -> None:
        """
        Save the fine-tuned model and tokenizer.
        
        Args:
            save_path: Path where model should be saved
        """
        pass
    
    @abstractmethod
    def get_model_class_name(self) -> str:
        """
        Get the model class name for configuration file.
        
        Returns:
            Model class name (e.g., 'AutoPeftModelForCausalLM')
        """
        pass
    
    @staticmethod
    def get_supported_tasks() -> List[str]:
        """
        Get list of tasks supported by this provider.
        
        Returns:
            List of supported task names
        """
        return ["text-generation", "summarization", "extractive-question-answering"]
