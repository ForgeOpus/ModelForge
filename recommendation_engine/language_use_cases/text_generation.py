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
    compatible_objectives = [
        TrainingObjective.SUPERVISED,
        TrainingObjective.PREFERENCE,
        TrainingObjective.REINFORCEMENT,
    ]
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
        TrainingObjective.PREFERENCE: DataFormatSpec(
            user_facing_description=(
                "A dataset with a 'chosen' column (the preferred continuation) and a 'rejected' "
                "column (a less-preferred continuation). Include a shared 'prompt' column (or "
                "'input_text' alias) when your continuations do not already contain the context; "
                "DPO also accepts implicit-prompt pairs."
            ),
            column_aliases={
                "preferred_continuation": "chosen",
                "rejected_continuation": "rejected",
            },
            optional_column_aliases={"input_text": "prompt"},
        ),
        TrainingObjective.REINFORCEMENT: DataFormatSpec(
            user_facing_description=(
                "A dataset with a 'prompt' column (the text prefix) and an 'answer' column "
                "(the exact desired continuation). ModelForge's built-in reward is 1 only "
                "when a generated continuation matches the answer after trimming whitespace."
            ),
            column_aliases={
                "input_text": "prompt",
                "ideal_continuation": "answer",
            },
        ),
    }
