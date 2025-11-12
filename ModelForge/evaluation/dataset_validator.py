"""
Dataset validation utilities.
Validates datasets before training to catch errors early.
"""
from datasets import load_dataset
from typing import Dict, List

from ..exceptions import DatasetValidationError
from ..logging_config import logger


class DatasetValidator:
    """Validator for training datasets."""

    # Required fields for each task type
    TASK_FIELD_REQUIREMENTS = {
        "text-generation": ["input", "output"],
        "summarization": ["document", "summary"],
        "extractive-question-answering": ["context", "question", "answers"],
    }

    # Required fields for each training strategy
    STRATEGY_FIELD_REQUIREMENTS = {
        "sft": {},  # Task-specific
        "rlhf": ["prompt", "chosen", "rejected"],
        "dpo": ["prompt", "chosen", "rejected"],
        "qlora": {},  # Task-specific
    }

    @classmethod
    def validate_dataset(
        cls,
        dataset_path: str,
        task: str,
        strategy: str = "sft",
        min_examples: int = 10,
    ) -> bool:
        """
        Validate a dataset for training.

        Args:
            dataset_path: Path to dataset file
            task: Task type
            strategy: Training strategy
            min_examples: Minimum number of examples required

        Returns:
            True if dataset is valid

        Raises:
            DatasetValidationError: If dataset is invalid
        """
        logger.info(f"Validating dataset: {dataset_path} for task={task}, strategy={strategy}")

        try:
            # Load dataset
            dataset = load_dataset("json", data_files=dataset_path, split="train")

            # Check minimum size
            if len(dataset) < min_examples:
                raise DatasetValidationError(
                    f"Dataset too small: {len(dataset)} examples. "
                    f"Minimum required: {min_examples}"
                )

            # Check required fields based on strategy
            if strategy in ["rlhf", "dpo"]:
                required_fields = cls.STRATEGY_FIELD_REQUIREMENTS[strategy]
            else:
                # For SFT and QLoRA, use task-specific fields
                required_fields = cls.TASK_FIELD_REQUIREMENTS.get(task, [])

            # Validate fields
            missing_fields = [
                field for field in required_fields
                if field not in dataset.column_names
            ]

            if missing_fields:
                raise DatasetValidationError(
                    f"Dataset missing required fields: {missing_fields}. "
                    f"Required for {strategy}/{task}: {required_fields}. "
                    f"Available fields: {dataset.column_names}"
                )

            # Additional validation: check for empty examples
            if len(required_fields) > 0:
                for i, example in enumerate(dataset.select(range(min(10, len(dataset))))):
                    for field in required_fields:
                        if not example.get(field):
                            logger.warning(
                                f"Example {i} has empty field '{field}'"
                            )

            logger.info(
                f"Dataset validated successfully: {len(dataset)} examples, "
                f"fields: {dataset.column_names}"
            )
            return True

        except FileNotFoundError as e:
            raise DatasetValidationError(
                f"Dataset file not found: {dataset_path}"
            ) from e

        except Exception as e:
            raise DatasetValidationError(
                f"Error validating dataset: {str(e)}"
            ) from e

    @classmethod
    def get_required_fields(cls, task: str, strategy: str = "sft") -> List[str]:
        """
        Get required fields for a task/strategy combination.

        Args:
            task: Task type
            strategy: Training strategy

        Returns:
            List of required field names
        """
        if strategy in ["rlhf", "dpo"]:
            return cls.STRATEGY_FIELD_REQUIREMENTS[strategy]
        else:
            return cls.TASK_FIELD_REQUIREMENTS.get(task, [])
