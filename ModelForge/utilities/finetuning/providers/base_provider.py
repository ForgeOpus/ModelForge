"""
Base provider interface for fine-tuning implementations.

This module defines the abstract interface that all fine-tuning providers
must implement to ensure consistent behavior across different backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datasets import Dataset


class FinetuningProvider(ABC):
    """
    Abstract base class defining the interface for fine-tuning providers.
    
    All fine-tuning providers (HuggingFace, Unsloth, etc.) must implement
    this interface to ensure compatibility with the ModelForge pipeline.
    """
    
    def __init__(
        self,
        model_name: str,
        task: str,
        compute_specs: str = "low_end"
    ) -> None:
        """
        Initialize the fine-tuning provider.
        
        Args:
            model_name: Name or path of the model to fine-tune
            task: Task type (text-generation, summarization, extractive-question-answering)
            compute_specs: Hardware profile (low_end, mid_range, high_end)
        """
        self.model_name = model_name
        self.task = task
        self.compute_specs = compute_specs
        self.model = None
        self.tokenizer = None
        self.dataset: Optional[Dataset] = None
        self.settings: Dict[str, Any] = {}
        
    @abstractmethod
    def load_model(self, **kwargs) -> Tuple[Any, Any]:
        """
        Load the model and tokenizer with provider-specific configurations.
        
        Args:
            **kwargs: Provider-specific model loading parameters
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            Exception: If model loading fails
        """
        pass
    
    @abstractmethod
    def prepare_dataset(self, dataset_path: str, **kwargs) -> Dataset:
        """
        Load and prepare the dataset for training.
        
        Args:
            dataset_path: Path to the dataset file
            **kwargs: Provider-specific dataset preparation parameters
            
        Returns:
            Prepared dataset ready for training
            
        Raises:
            Exception: If dataset loading or preparation fails
        """
        pass
    
    @abstractmethod
    def train(self, **kwargs) -> str:
        """
        Execute the fine-tuning process.
        
        Args:
            **kwargs: Provider-specific training parameters
            
        Returns:
            Path to the saved fine-tuned model
            
        Raises:
            Exception: If training fails
        """
        pass
    
    @abstractmethod
    def export_model(self, output_path: str, **kwargs) -> bool:
        """
        Export the fine-tuned model to the specified path.
        
        Args:
            output_path: Directory to save the exported model
            **kwargs: Provider-specific export parameters
            
        Returns:
            True if export succeeds, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_hyperparameters(self) -> List[str]:
        """
        Return a list of hyperparameters supported by this provider.
        
        Returns:
            List of hyperparameter names
        """
        pass
    
    @abstractmethod
    def validate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize provider-specific settings.
        
        Args:
            settings: Dictionary of hyperparameters and configurations
            
        Returns:
            Validated settings dictionary
            
        Raises:
            ValueError: If settings are invalid
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_provider_name(cls) -> str:
        """
        Return the canonical name of this provider.
        
        Returns:
            Provider name (e.g., "huggingface", "unsloth")
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_provider_description(cls) -> str:
        """
        Return a human-readable description of this provider.
        
        Returns:
            Provider description
        """
        pass
    
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Check if this provider's dependencies are installed and available.
        
        Returns:
            True if provider can be used, False otherwise
        """
        pass
    
    def set_settings(self, **kwargs) -> None:
        """
        Set training settings from keyword arguments.
        
        Args:
            **kwargs: Settings to apply
        """
        validated_settings = self.validate_settings(kwargs)
        self.settings.update(validated_settings)
