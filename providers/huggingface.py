from providers.provider import Provider
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase
from typing import Any


class HuggingFaceProvider(Provider):

    name = "HuggingFace"
    description = "Fine-tunes models using the Hugging Face Transformers and TRL libraries."
    slug = "huggingface"

    def load_model(self, model_name: str, model_class: Any, **kwargs) -> PreTrainedModel:
        # Accept saved plans from older ModelForge versions, but never forward
        # the removed Transformers load_in_*bit keywords to model constructors.
        four_bit = kwargs.pop("load_in_4bit", False)
        eight_bit = kwargs.pop("load_in_8bit", False)
        quantization = kwargs.get("quantization_config")
        if four_bit and eight_bit:
            raise ValueError("Choose either 4-bit or 8-bit quantization, not both.")
        if quantization is not None and (four_bit or eight_bit):
            raise ValueError("Quantization was specified in both legacy flags and quantization_config.")
        if four_bit or eight_bit:
            quantization = {"load_in_4bit": four_bit, "load_in_8bit": eight_bit}
            if four_bit:
                quantization.update(bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        if isinstance(quantization, dict):
            kwargs["quantization_config"] = BitsAndBytesConfig(**quantization)
        return model_class.from_pretrained(model_name, **kwargs)

    def load_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
