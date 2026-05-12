from abc import ABC, abstractmethod
from transformers import AutoModel
from typing import Type

class FinetuningTask(ABC):

    name: str = None
    slug: str = None
    description: str = None

    model_class: AutoModel = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Define the display name for the task"""
        raise NotImplementedError("Please add a name for task", self.name)

    @property
    @abstractmethod
    def slug(self) -> str:
        """Define a unique task alub for the task"""
        raise NotImplementedError("Please add a unique slub for task", self.name)

    @property
    @abstractmethod
    def description(self) -> str:
        """Give a brief description about the task, the expected dataset format, the use cases
        and any other relevant information"""
        raise NotImplementedError("Please add a description for task", self.name)

    @property
    @abstractmethod
    def model_class(self) -> Type[AutoModel]:
        """Define the AutoModel class associated with the task type to create model of the task
        for finetuning"""
        raise NotImplementedError(f"Please add a model class for the task {self.name}")
