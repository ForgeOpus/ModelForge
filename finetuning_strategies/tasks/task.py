"""
Base class for finetuning tasks. Each task should inherit from this class and implement the required attributes and methods.
Maintains a registry of all defined tasks, allowing for easy retrieval and management of tasks by their slug. Each task must 
define a unique slug, a name, a description, and specify the model class it is associated with. The tasks are auto-registered to the
registry upon subclassing, ensuring that all tasks are easily accessible and organized.

All task access operations can be performed using the model_factory interface, which provides a consistent way to retrieve and manage 4
tasks based on their slugs.
"""

from abc import ABC
from transformers import AutoModel

class FinetuningTask(ABC):

    _registry = {}
    
    name: str = None
    slug: str = None
    description: str = None

    model_class: AutoModel = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'slug') and cls.slug is not None and isinstance(cls.slug, str):
            if cls.slug in FinetuningTask._registry:
                raise ValueError(f"Duplicate slug '{cls.slug}' found for task class '{cls.__name__}'. Please ensure that each task class has a unique slug.")
            FinetuningTask._registry[cls.slug] = cls
        else:
            raise ValueError(f"Task class '{cls.__name__}' must define a 'slug' attribute")

        if cls.name is None or not isinstance(cls.name, str):
            raise ValueError(f"Task class '{cls.__name__}' must define a 'name' attribute of type string")
        if cls.description is None or not isinstance(cls.description, str):
            raise ValueError(f"Task class '{cls.__name__}' must define a 'description' attribute of type string")
        if cls.model_class is None or not issubclass(cls.model_class, AutoModel):
            raise ValueError(f"Task class '{cls.__name__}' must define a 'model_class' attribute that is a subclass of AutoModel")
