from recommendation_engine.language_use_cases.use_case import LanguageUseCase, DataFormatSpec
from utilities.training_objective import TrainingObjective


class SummarizationUseCase(LanguageUseCase):
    """Use case for abstractive document summarization."""

    name = "Summarization"
    slug = "summarization"
    description = (
        "Train a model to produce concise summaries of longer documents. "
        "Covers news summarization, meeting notes, scientific abstracts, and more."
    )
    task_slug = "seq2seq_modeling"
    hub_task_filter = "summarization"
    compatible_objectives = [TrainingObjective.SUPERVISED]
    data_formats = {
        TrainingObjective.SUPERVISED: DataFormatSpec(
            user_facing_description=(
                "A dataset with a 'document' column (the source text to summarize) and a "
                "'summary' column (the target summary)."
            ),
            column_aliases={
                "document": "prompt",
                "summary": "completion",
            },
        ),
    }
