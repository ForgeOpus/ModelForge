from typing import Dict, List
from pydantic import BaseModel, Field
from utilities.training_objective import TrainingObjective


class DataFormatSpec(BaseModel):
    """User-facing description of what data to prepare for a given (use case, objective) pair."""
    user_facing_description: str
    column_aliases: Dict[str, str]
    # Some trainer formats support a useful canonical column that is not
    # required for every valid record (DPO's shared prompt is one example).
    # Keep those aliases separate so normalization does not mistake an optional
    # convenience for a required source field.
    optional_column_aliases: Dict[str, str] = Field(default_factory=dict)


class LanguageUseCase:
    """Abstract base class for language use cases. Subclasses auto-register to _registry on definition.

    Class-level attributes (not instance attributes — do NOT use __init__):
        name: Display name (e.g. "Summarization")
        slug: Unique identifier (e.g. "summarization")
        description: User-facing explanation
        task_slug: The FinetuningTask slug this use case maps to (validated lazily at call site)
        hub_task_filter: HuggingFace pipeline_tag string (e.g. "summarization")
        compatible_objectives: List[TrainingObjective] — which training objectives are valid
        data_formats: Dict[TrainingObjective, DataFormatSpec] — one entry per compatible objective
    """

    _registry: Dict[str, type] = {}

    name: str = None
    slug: str = None
    description: str = None
    task_slug: str = None
    hub_task_filter: str = None
    compatible_objectives: List[TrainingObjective] = None
    data_formats: Dict[TrainingObjective, DataFormatSpec] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate required string attributes
        for attr in ("name", "slug", "description", "task_slug", "hub_task_filter"):
            val = getattr(cls, attr, None)
            if not val or not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"LanguageUseCase subclass '{cls.__name__}' must define a non-empty "
                    f"string '{attr}' class attribute."
                )

        # Validate compatible_objectives
        objs = getattr(cls, "compatible_objectives", None)
        if not objs or not isinstance(objs, list) or len(objs) == 0:
            raise ValueError(
                f"LanguageUseCase subclass '{cls.__name__}' must define a non-empty "
                f"list 'compatible_objectives'."
            )
        for obj in objs:
            if not isinstance(obj, TrainingObjective):
                raise ValueError(
                    f"LanguageUseCase subclass '{cls.__name__}': all entries in "
                    f"'compatible_objectives' must be TrainingObjective instances, got {obj!r}."
                )

        # Validate data_formats — must have exactly one entry per compatible objective
        fmts = getattr(cls, "data_formats", None)
        if not fmts or not isinstance(fmts, dict):
            raise ValueError(
                f"LanguageUseCase subclass '{cls.__name__}' must define a 'data_formats' dict."
            )
        for obj in objs:
            if obj not in fmts:
                raise ValueError(
                    f"LanguageUseCase subclass '{cls.__name__}': 'data_formats' must have "
                    f"an entry for each compatible objective. Missing: {obj!r}."
                )
            if not isinstance(fmts[obj], DataFormatSpec):
                raise ValueError(
                    f"LanguageUseCase subclass '{cls.__name__}': data_formats[{obj!r}] "
                    f"must be a DataFormatSpec instance."
                )

        # Uniqueness check
        if cls.slug in LanguageUseCase._registry:
            raise ValueError(
                f"A LanguageUseCase with slug '{cls.slug}' is already registered. "
                f"'{cls.__name__}' cannot be registered."
            )

        LanguageUseCase._registry[cls.slug] = cls

    @classmethod
    def get_use_cases(cls) -> List[type]:
        """Return a list of all registered use case slugs."""
        return list(cls._registry.keys())

    @classmethod
    def list_use_case_classes(cls) -> List[str]:
        """Return a list of registered use case slugs."""
        return list(cls._registry.keys())

    @classmethod
    def get_use_case_by_slug(cls, slug: str) -> type:
        """Look up a registered use case by slug. Raises ValueError if not found."""
        if slug not in cls._registry:
            registered = sorted(cls._registry.keys())
            raise ValueError(
                f"No LanguageUseCase found with slug '{slug}'. "
                f"Registered slugs: {registered}"
            )
        return cls._registry[slug]

    @classmethod
    def is_valid_use_case(cls, slug: str) -> bool:
        """Return True if the given slug corresponds to a registered use case."""
        return slug in cls._registry

    @classmethod
    def get_use_case(cls, slug: str) -> type:
        """Look up a registered use case by slug. Raises ValueError if not found."""
        if slug not in cls._registry:
            registered = sorted(cls._registry.keys())
            raise ValueError(
                f"No LanguageUseCase found with slug '{slug}'. "
                f"Registered slugs: {registered}"
            )
        return cls._registry[slug]

    def __str__(self) -> str:
        return f"{self.slug}({self.name}: {self.description})"
