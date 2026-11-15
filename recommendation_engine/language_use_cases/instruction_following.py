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
    compatible_objectives = [TrainingObjective.SUPERVISED, TrainingObjective.PREFERENCE]
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
                "A dataset with a 'prompt' column (the instruction), a 'chosen' column "
                "(the preferred response), and a 'rejected' column (the less preferred response)."
            ),
            column_aliases={
                "instruction": "prompt",
                "preferred_response": "chosen",
                "dispreferred_response": "rejected",
            },
        ),
    }
