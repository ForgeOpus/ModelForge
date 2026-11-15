from recommendation_engine.language_use_cases.use_case import LanguageUseCase, DataFormatSpec
from utilities.training_objective import TrainingObjective


class InstructionFollowingUseCase(LanguageUseCase):
    """Use case for instruction-following / chat assistants."""

    name = "Instruction Following"
    slug = "instruction_following"
    description = (
        "Train a model to follow natural-language instructions and produce helpful, accurate "
        "responses. Covers chatbots, assistants, and any task requiring instruction adherence."
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
                "A dataset with a 'prompt' column (the instruction or user message) and a "
                "'completion' column (the ideal assistant response)."
            ),
            column_aliases={
                "instruction": "prompt",
                "response": "completion",
            },
        ),
        TrainingObjective.PREFERENCE: DataFormatSpec(
            user_facing_description=(
                "A dataset with a 'chosen' column (the preferred response) and a 'rejected' "
                "column (the less preferred response). Include a shared 'prompt' column (or "
                "'instruction' alias) when your responses do not already contain the context; "
                "DPO also accepts implicit-prompt pairs."
            ),
            column_aliases={
                "preferred_response": "chosen",
                "dispreferred_response": "rejected",
            },
            optional_column_aliases={"instruction": "prompt"},
        ),
        TrainingObjective.REINFORCEMENT: DataFormatSpec(
            user_facing_description=(
                "A dataset with a 'prompt' column (the instruction) and an 'answer' column "
                "(the exact ideal response). ModelForge awards a reward of 1 only when a "
                "generated response exactly matches the answer, ignoring surrounding whitespace."
            ),
            column_aliases={
                "instruction": "prompt",
                "ideal_answer": "answer",
            },
        ),
    }
