from recommendation_engine.language_use_cases.use_case import LanguageUseCase, DataFormatSpec
from utilities.training_objective import TrainingObjective


class TextGenerationUseCase(LanguageUseCase):
    """Use case for open-ended text generation from a prompt."""

    name = "Text Generation"
    slug = "text_generation"
    description = (
        "Train a model to generate coherent, contextually relevant text given an input prompt. "
        "Suitable for creative writing, story continuation, and general text completion tasks."
    )
    task_slug = "causal_language_modeling"
    hub_task_filter = "text-generation"
    compatible_objectives = [TrainingObjective.SUPERVISED]
    data_formats = {
        TrainingObjective.SUPERVISED: DataFormatSpec(
            user_facing_description=(
                "A dataset with a 'prompt' column (the input text the model receives) and a "
                "'completion' column (the target text the model should generate)."
            ),
            column_aliases={
                "input_text": "prompt",
                "generated_text": "completion",
            },
        ),
    }
