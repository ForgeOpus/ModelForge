from finetuning_strategies.tasks import FinetuningTask
from typing import Dict, Any, Type, List
from pydantic import BaseModel, Field

class TaskResponse(BaseModel):
    name: str = Field(..., description="The display name of the fine-tuning task.")
    slug: str = Field(..., description="The unique slug identifier for the fine-tuning task.")
    description: str = Field(..., description="A detailed description of the fine-tuning task, including its use cases and requirements.")
    model_class: str = Field(..., description="The model class associated with this fine-tuning task, typically a subclass of AutoModel.")

class TaskModels:
    """
    This module provides a collection of different supported task types associated with language modeling.
    These task types control the model heads and the expected dataset formats for the different SFT technniques.
    """

    @classmethod
    def get_task_class(cls, slug: str) -> Type[FinetuningTask]:
        """Get a fine-tuning task class by its slug identifier.

        Args:
            slug (str): The unique slug identifier for the desired fine-tuning task.

        Returns:
            Type[FinetuningTask]: The fine-tuning task class associated with the provided slug.

        Raises:
            ValueError: If no task is found for the given slug.
        """
        if slug in FinetuningTask._registry:
            return FinetuningTask._registry[slug]
        else:
            raise ValueError(f"No task found for slug '{slug}'")

    @classmethod
    def list_tasks(cls) -> List[Dict[str, Any]]:
        """
        List all available fine-tuning tasks with their details.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the name, slug, description,
            and model class for a fine-tuning task.
        """
        return [
            TaskResponse(
                name=task_cls.name,
                slug=task_cls.slug,
                description=task_cls.description,
                model_class=task_cls.model_class.__name__ if task_cls.model_class is not None else None
            )
            for task_cls in FinetuningTask._registry.values()
        ]
