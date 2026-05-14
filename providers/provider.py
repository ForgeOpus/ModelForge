from abc import ABC, abstractmethod

class Provider(ABC):
    
    _registry = {}
    
    name: str = None
    description: str = None
    slug: str = None


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not issubclass(cls, Provider):
            raise ValueError(f"Provider class '{cls.__name__}' must be inherited from the base Provider class")
        if not isinstance(cls.name, str):
            raise ValueError(f"Provider class '{cls.__name__}' must have a 'name' of type str")
        if not isinstance(cls.description, str):
            raise ValueError(f"Provider class '{cls.__name__}' must have a 'description' of type str")
        if not isinstance(cls.slug, str):
            raise ValueError(f"Provider class '{cls.__name__}' must have a 'slug' of type str")
            
        if cls.slug in cls._registry:
            raise ValueError(f"Provider with slug '{cls.slug}' already exists. Please choose a unique slug")
        cls._registry[cls.slug] = cls


    

    
