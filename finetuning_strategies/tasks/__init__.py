from finetuning_strategies.tasks.task import FinetuningTask
from finetuning_strategies.tasks.causal_language_modeling import CausalLM
from finetuning_strategies.tasks.seq2seq_modeling import Seq2SeqLM
from finetuning_strategies.tasks.token_classification import TokenClassification
from finetuning_strategies.tasks.question_answering import QuestionAnswering

# A list of all registered task slugs, available for Literal type annotations
# and engine-level validation.
SUPPORTED_TASKS: list[str] = list(FinetuningTask._registry.keys())

__all__ = ["FinetuningTask", "CausalLM", "Seq2SeqLM", "TokenClassification", "QuestionAnswering", "SUPPORTED_TASKS"]
