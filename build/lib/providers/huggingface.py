from providers.provider import Provider
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from typing import Any


class HuggingFaceProvider(Provider):

    name = "HuggingFace"
    description = "Fine-tunes models using the Hugging Face Transformers and TRL libraries."
    slug = "huggingface"

    def load_model(self, model_name: str, model_class: Any, **kwargs) -> PreTrainedModel:
        return model_class.from_pretrained(model_name, **kwargs)

    def load_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
