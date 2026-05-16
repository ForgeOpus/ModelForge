from finetuning_strategies.tasks.task import FinetuningTask
from transformers import AutoModelForTokenClassification


class TokenClassification(FinetuningTask):
    """Token-level classification task (e.g. NER, extractive QA span prediction).

    Each token in the input receives a label. Used for named entity recognition,
    part-of-speech tagging, and extractive question answering (answer span prediction).
    """

    name = "Token Classification"
    slug = "token_classification"
    description = (
        "Token classification assigns a label to each token in an input sequence. "
        "Applications include named entity recognition (NER), part-of-speech tagging, "
        "and extractive question answering where the model predicts answer span boundaries."
    )
    model_class = AutoModelForTokenClassification
