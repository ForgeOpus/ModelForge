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
