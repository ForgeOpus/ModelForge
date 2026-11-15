from finetuning_strategies.tasks.task import FinetuningTask
from transformers import AutoModelForCausalLM

class CausalLM(FinetuningTask):

    name = "Causal Language Modeling"
    slug = "causal_language_modeling"
    description = """
    Causal Language Modeling (CLM) is a fine-tuning strategy where a pre-trained language model is further trained to 
    predict the next token in a sequence, given all the previous tokens. This approach is particularly effective for 
    tasks that involve generating coherent and contextually relevant text, such as story generation, dialogue systems, 
    and code generation. In CLM, the model learns to understand the structure and semantics of language by being exposed 
    to large amounts of text data, allowing it to generate fluent and meaningful responses based on the input it receives.
    """
    model_class = AutoModelForCausalLM
