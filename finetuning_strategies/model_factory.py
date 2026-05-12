from finetuning_strategies.tasks import FinetuningTask
from typing import Dict, Any, Type, List
from transformers import AutoModel

class TaskModels:
    """
    This module provides a collection of different supported task types associated with language modeling.
    These task types control the model heads and the expected dataset formats for the different SFT technniques.
    """

    _registry: Dict[str, Type[AutoModel]] = {}

    @classmethod
    def register(cls, task_class: Type[FinetuningTask]):
        def decorator(task_class: Type[FinetuningTask]):
            if not issubclass(task_class, FinetuningTask):
                raise ValueError(f"Task type '{task_class.__name__}' must be inherited from the base FinetuningTask")
            if task_class.slug in cls._registry:
                raise ValueError(f"Task of type '{task_class.slug}' already registered. Please choose a unique slug")
        return task_class
    
    @classmethod
    def get_task_class(cls, slug: str) -> Type[AutoModel]:
        if not slug in cls._registry:
            raise ValueError(f"Unsupported task type '{slug}'. Please choose one of '{list(cls._registry.keys())}'")
        return cls._registry[slug]
    
    @classmethod
    def list_task_slugs(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def list_tasks_display_info(cls) -> List[Dict[str, str]]:
        return [
            {"name": task_class.name, 
            "description": task_class.description} 
            for task_class in cls._registry.values()
        ]

    @classmethod
    def get_task_display_info(cls, slug: str) -> List[Dict[str, str]]:
        task_class: type[Any] = cls.get_task_class(slug)
        return {
            "name": task_class.name,
            "description": task_class.description
        }

    @classmethod
    def clear_registry(cls):
        cls._registry.clear()
