"""
Training strategy abstraction layer for ModelForge.
Enables support for multiple training strategies (SFT, RLHF, DPO, QLoRA, etc.)
"""
from typing import Protocol, Any, Dict
from abc import abstractmethod


class TrainingStrategy(Protocol):
    """
    Interface for training strategies.
    Each strategy defines how to prepare models, datasets, and execute training.
    """

    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of the training strategy.

        Returns:
            Strategy name (e.g., "sft", "rlhf", "dpo")
        """
        ...

    @abstractmethod
    def prepare_model(self, model: Any, config: Dict) -> Any:
        """
        Prepare the model for this training strategy.

        Args:
            model: Base model instance
            config: Configuration dictionary with strategy-specific settings

        Returns:
            Prepared model (e.g., with PEFT adapters applied)
        """
        ...

    @abstractmethod
    def prepare_dataset(self, dataset: Any, tokenizer: Any, config: Dict) -> Any:
        """
        Prepare the dataset for this training strategy.

        Args:
            dataset: Raw dataset
            tokenizer: Tokenizer instance
            config: Configuration dictionary

        Returns:
            Prepared dataset
        """
        ...

    @abstractmethod
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
        Create a trainer instance for this strategy.

        Args:
            model: Prepared model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer instance
            config: Configuration dictionary
            callbacks: Optional list of callbacks

        Returns:
            Trainer instance
        """
        ...

    @abstractmethod
    def get_required_dataset_fields(self) -> list:
        """
        Get the required dataset fields for this strategy.

        Returns:
            List of required field names
        """
        ...
