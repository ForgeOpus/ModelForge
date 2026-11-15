from recommendation_engine.language_use_cases.use_case import LanguageUseCase, DataFormatSpec
from utilities.training_objective import TrainingObjective


class ExtractiveQAUseCase(LanguageUseCase):
    """Use case for extractive question answering (span prediction)."""

    name = "Extractive QA"
    slug = "extractive_qa"
    description = (
        "Train a model to identify and extract answer spans from a context passage given a question. "
        "Suitable for reading comprehension and document-grounded QA tasks."
    )
    task_slug = "question_answering"
    strategy_slug = "qa"
    hub_task_filter = "question-answering"
    compatible_objectives = [TrainingObjective.SUPERVISED]
    data_formats = {
        TrainingObjective.SUPERVISED: DataFormatSpec(
            user_facing_description=(
                "A dataset with a 'context' column (the passage to search), a 'question' column "
                "(the question to answer), and an 'answers' column (dict with 'text' and "
                "'answer_start' keys indicating the answer span within the context)."
            ),
            column_aliases={
                "context": "context",
                "question": "question",
                "answers": "answers",
            },
        ),
    }
