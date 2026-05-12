from finetuning_strategies.tasks.task import FinetuningTask
from transformers import AutoModelForCausalLM, AutoModel
from typing import Type

class CausalLM(FinetuningTask):

    @property
    def name(self) -> str:
        self.name = "Text Generation"
        return self.name

    @property
    def description(self) -> str:
        self.description = "Generate sequences of text on any given topic"
        return self.description

    @property
    def slug(self) -> str:
        self.slug = "causal_language_modeling"
        return self.slug
    
    @property
    def model_class(self) -> Type[AutoModel]:
        self.model_class = AutoModelForCausalLM
        return self.model_class
    