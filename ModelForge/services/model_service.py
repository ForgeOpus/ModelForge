"""
Model service for managing fine-tuned models.
Handles model CRUD operations and validation.
"""
from typing import List, Dict, Optional

from ..database.database_manager import DatabaseManager
from ..providers.provider_factory import ProviderFactory
from ..exceptions import ModelAccessError
from ..logging_config import logger


class ModelService:
    """Service for managing models."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize model service.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        logger.info("Model service initialized")

    def get_all_models(self) -> List[Dict]:
        """
        Get all fine-tuned models.

        Returns:
            List of model dictionaries
        """
        logger.info("Fetching all models")
        return self.db_manager.get_all_models()

    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """
        Get a model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Model dictionary if found, None otherwise
        """
        logger.info(f"Fetching model: {model_id}")
        return self.db_manager.get_model_by_id(model_id)

    def get_models_by_task(self, task: str) -> List[Dict]:
        """
        Get all models for a specific task.

        Args:
            task: Task type

        Returns:
            List of model dictionaries
        """
        logger.info(f"Fetching models for task: {task}")
        return self.db_manager.get_models_by_task(task)

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model.

        Args:
            model_id: Model identifier

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Deleting model: {model_id}")
        return self.db_manager.delete_model(model_id)

    def validate_model_access(
        self,
        repo_name: str,
        model_class: str = "AutoModelForCausalLM",
        provider_name: str = "huggingface",
    ) -> Dict:
        """
        Validate that a model is accessible.

        Args:
            repo_name: Model repository name
            model_class: Model class name
            provider_name: Provider to use for validation

        Returns:
            Dictionary with validation result

        Raises:
            ModelAccessError: If model is not accessible
        """
        logger.info(f"Validating model access: {repo_name}")

        try:
            provider = ProviderFactory.create_provider(provider_name)
            is_valid = provider.validate_model_access(repo_name, model_class)

            if is_valid:
                return {
                    "valid": True,
                    "message": f"Model {repo_name} is accessible",
                }
            else:
                return {
                    "valid": False,
                    "message": f"Model {repo_name} is not accessible",
                }

        except ModelAccessError as e:
            logger.error(f"Model access validation failed: {e}")
            raise

        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return {
                "valid": False,
                "message": f"Validation error: {str(e)}",
            }
