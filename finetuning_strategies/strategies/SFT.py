'''
Defines the Supervised Fine-Tuning (SFT) strategy for model fine-tuning. 
This strategy involves training a model on a labeled dataset using supervised 
learning techniques. The SFT class inherits from the Finetuner base class and 
implements the finetune method to perform the fine-tuning process. The provider 
attribute indicates the specific fine-tuning provider or framework being used for 
this strategy.
'''
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from finetuning_strategies.strategy_factory import FinetuningStrategies
from transformers import TokenizersBackend, SentencePieceBackend
from finetuning_strategies.model_factory import TaskModels
from typing import List, Literal, Any, Union
from providers import SUPPORTED_PROVIDERS
from .strategy import FinetuningStrategy
from trl import SFTTrainer, SFTConfig
from datasets import IterableDataset
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch import nn
import torch


@FinetuningStrategies.register()
class SFT(FinetuningStrategy):
    
    supported_providers = ("HuggingFace", "Unsloth")
    optimizer = None  # Placeholder for the optimizer to be used in fine-tuning

    # def __init__(self, provider: str):
    #     if provider not in self.supported_providers:
    #         raise ValueError(f"Provider '{provider}' is not supported for SFT. Supported providers: {self.supported_providers}")
    #     self.provider = provider

    @property
    def name(self) -> str:
        self.name = "Supervised Fine-Tuning"
        return self.name

    @property
    def slug(self) -> str:
        return "sft"
    
    @property
    def device(self) -> str:
        # Select backend-appropriate device for training
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        return self.device

    @property
    def description(self) -> str:
        return ("Supervised Fine-Tuning (SFT) is a strategy that involves training a model on a labeled dataset using supervised learning techniques. "
                "In this approach, the model learns to make predictions based on the input data and the corresponding labels, allowing it to generalize to new, unseen data. "
                "SFT is used in LLMs to adapt a pre-trained model to specific tasks or domains by fine-tuning it on task-specific datasets, or"
                "to make it follow instructions and output formats to be able to hold a proper conversation with users. This strategy is effective for improving the performance of "
                "models on specific tasks.")

    @property
    def supported_providers(self) -> List[str]:
        self.supported_providers = [
            "HuggingFace", 
            "Unsloth"
        ]
        return self.supported_providers
        

    @staticmethod
    def validate_data_format(iterable_dataset: IterableDataset) -> bool:
        # Implement validation logic for the data format required by SFT
        columns = list(iterable_dataset.features.keys())

        ## Check for SFT specific columns
        required_columns = ["prompt", "completion"]  # Example required columns for SFT

        for col in required_columns:
            if col not in columns:
                raise ValueError(f"Required column '{col}' is missing from the dataset.")

        return True

    @staticmethod
    def apply_data_format(prompt:str, completion:str, type_: Literal["conversational", "language_modeling", "prompt", "conversational_prompt"] = "prompt"):

        if not isinstance(prompt, str):
            raise ValueError(f"Prompt must be a string.")
        if not isinstance(completion, str):
            raise ValueError(f"Completion must be a string.")
        
        if type_ == "conversational":
            return {
                "messages": [{"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}]
            }
        elif type_ == "language_modeling":
            return {
                "text": prompt
            }
        elif type_ == "prompt":
            return {
                "prompt":prompt,
                "completion": completion
            }
        elif type_ == "conversational_prompt":
            return {
                "prompt": [{"role": "user", "content": prompt}],
                "completion": [{"role": "assistant", "content": completion}]
            }
        else:
            raise ValueError(f"Chat template format {type_} is in valid. Please choose one of 'conersational', 'language_modeling', 'prompt', or 'conversational_prompt'.")

    def build_sft_config(
        self, 
    ) -> None:
        pass

    def finetune(self, model_name, dataset, task_slug):
        # Implement the fine-tuning logic using the specified provider
        # This may involve preparing the data, setting up the training loop, 
        # and updating the model weights
        
        # Validate data format
        self.validate_data_format(dataset)
        task_class: type[Any] = TaskModels.get_task_class(task_slug)
        
        #Todo: add any extra params to pass, as needed
        model = task_class.from_pretrained(model_name)

        tokenizer: Union[SentencePieceBackend, TokenizersBackend, None] = AutoTokenizer.from_pretrained(model)
        has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
        
