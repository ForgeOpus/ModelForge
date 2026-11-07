from finetuning_strategies.tasks.task import FinetuningTask
from transformers import AutoModelForQuestionAnswering


class QuestionAnswering(FinetuningTask):
    """Extractive question answering with start/end span prediction."""

    name = "Extractive Question Answering"
    slug = "question_answering"
    description = "Predicts the start and end token positions of an answer in a context passage."
    model_class = AutoModelForQuestionAnswering
