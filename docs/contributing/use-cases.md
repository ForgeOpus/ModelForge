# Adding a use case

A use case is the goal a user picks first, such as "Summarization". It decides which Hub models are searched, which model head is loaded, which objectives can be trained and what the dataset columns are called. This page covers the `LanguageUseCase` interface, how a use case is registered and resolved, and what to test and document when you add one.

!!! note "There are no model config files in v4"
    ModelForge v3 recommended models from hand-written JSON files (`model_configs/low_end.json`, `mid_range.json`, `high_end.json`). Those files and the "model configs" contributor page are gone. v4 searches the Hugging Face Hub at plan time for the most-downloaded models with the use case's `pipeline_tag` that fit the detected memory, so adding a model to ModelForge needs no code change. Adding a use case is the closest equivalent.

## The base class

`LanguageUseCase` lives in `recommendation_engine/language_use_cases/use_case.py`. A use case is a subclass with **class attributes only**. Do not define `__init__`; the class is never instantiated by the workflow.

| Attribute | Type | Required | Meaning |
|---|---|---|---|
| `name` | `str` | yes | Display name shown in every interface, e.g. `"Summarization"` |
| `slug` | `str` | yes | Unique ID used in the API, CLI and database, e.g. `"summarization"` |
| `description` | `str` | yes | One or two sentences shown next to the name |
| `task_slug` | `str` | yes | A registered `FinetuningTask` slug: `causal_language_modeling`, `seq2seq_modeling`, `token_classification` or `question_answering` |
| `hub_task_filter` | `str` | yes | Hugging Face Hub `pipeline_tag` used to search for base models, e.g. `"text-generation"` |
| `compatible_objectives` | `list[TrainingObjective]` | yes | Any of `SUPERVISED`, `PREFERENCE`, `REINFORCEMENT` from `utilities/training_objective.py` |
| `data_formats` | `dict[TrainingObjective, DataFormatSpec]` | yes | One `DataFormatSpec` per compatible objective |
| `strategy_slug` | `str` | no | Force one strategy for every objective, bypassing the objective map |

`DataFormatSpec` is a Pydantic model in the same file:

| Field | Type | Meaning |
|---|---|---|
| `user_facing_description` | `str` | Plain description of the columns to prepare. Shown in the UI and returned by `POST /api/datasets/validate`. |
| `column_aliases` | `dict[str, str]` | Required columns, as `{user_column: trainer_column}` |
| `optional_column_aliases` | `dict[str, str]` | Columns that are renamed if present but not required. Default `{}`. |

### What is validated at definition time

`LanguageUseCase.__init_subclass__` runs when the class body is executed and raises `ValueError` if:

- `name`, `slug`, `description`, `task_slug` or `hub_task_filter` is missing, not a string or blank;
- `compatible_objectives` is missing, empty, not a list, or contains anything other than `TrainingObjective` members;
- `data_formats` is not a dict, lacks an entry for a compatible objective, or holds something other than a `DataFormatSpec`;
- another use case already registered the same `slug`.

`task_slug` and the strategy are **not** checked here. They are checked when the use case is resolved (below), so a typo there shows up as a hidden objective or a planning error, not an import error.

## Registration

Defining the subclass registers it in `LanguageUseCase._registry` under its slug. The module has to be imported for that to happen. Add an import to `recommendation_engine/language_use_cases/__init__.py` (and to `__all__`):

```python
from recommendation_engine.language_use_cases.translation import TranslationUseCase
```

`workflow/registries.py:load_core_registries()` imports that package, and `WorkflowService()`, `assemble_and_train()` and the CLI all call it. Once imported, the use case appears in:

- `GET /api/system` and `GET /api/use-cases`, and therefore the browser's **New run** page, which reads its tiles from `/api/system`;
- `modelforge list-tasks`, `modelforge recommend --use-case <slug>`, the terminal wizard and `modelforge-nb`.

No frontend change is needed to make a use case selectable.

## How a use case is resolved

`RecommendationEngine._resolve_use_case_and_strategy()` in `utilities/recommendation_engine.py` turns `(use_case_slug, objective)` into a task and a strategy:

1. Look up the use case. Unknown slug: `ValueError`.
2. Check the objective is in `compatible_objectives`.
3. Pick the strategy: the use case's `strategy_slug` if set, otherwise `OBJECTIVE_STRATEGY_MAP`:

    | Objective | Strategy |
    |---|---|
    | `supervised` | `sft` |
    | `preference` | `dpo` |
    | `reinforcement` | `grpo` |

4. Check the strategy is registered and lists `task_slug` in its `supported_tasks`.
5. Return `data_formats[objective]`.

Current task support per strategy:

| Strategy | `supported_tasks` |
|---|---|
| `sft` | `causal_language_modeling`, `seq2seq_modeling` |
| `dpo` | `causal_language_modeling` |
| `grpo` | `causal_language_modeling` |
| `qa` | `question_answering` |

`WorkflowService.list_use_cases()` runs the same check for each objective. Objectives with no registered, task-compatible strategy are dropped from `compatible_objectives` and listed in `unavailable_objectives` with a reason. If none remain, the whole use case is left out of the selection APIs and reported under `unavailable_use_cases` in `GET /api/system`. So a summarization use case that lists `PREFERENCE` would silently lose it, because `dpo` does not support `seq2seq_modeling`.

`token_classification` is a registered task, but no strategy supports it yet, so a use case built on it would be hidden.

## Choosing `task_slug` and `hub_task_filter`

The task decides the `AutoModel*` class and LoRA task type. `hub_task_filter` decides which models the planner finds. Keep them consistent:

| `task_slug` | Model class | Hub `pipeline_tag` values a typed-in model may have |
|---|---|---|
| `causal_language_modeling` | `AutoModelForCausalLM` | `text-generation` |
| `seq2seq_modeling` | `AutoModelForSeq2SeqLM` | `summarization`, `translation`, `text2text-generation` |
| `token_classification` | `AutoModelForTokenClassification` | `token-classification` |
| `question_answering` | `AutoModelForQuestionAnswering` | `question-answering` |

The right-hand column is `expected_pipeline_tags` in `WorkflowService.validate_model_id()`. A model ID a user types in is rejected if its tag is outside that set for the task, so pick a `hub_task_filter` from it.

The Hub search keeps only models with safetensors parameter counts, skips repos named `tiny`, `random` or `dummy` and `internal-testing` owners, and sorts by downloads. The extra filter that keeps only chat and instruct checkpoints is hard-coded to the slug `instruction_following` (`instruction_only=(use_case_slug == "instruction_following")`); a new use case does not get it.

## Column aliases

Trainers read fixed column names. Users' datasets rarely use them. `column_aliases` bridges the two: the key is the name you document to users, the value is the name the trainer reads.

`DatasetFactory.normalize_columns()` in `dataset_manager/dataset_factory.py` applies the mapping in two places, so validation and training see the same data:

- `WorkflowService.validate_dataset()`, before calling the strategy's `validate_data_format()`;
- `workflow/training.py:assemble_and_train()`, before training.

Rules it enforces:

- A required source column may be missing only if its trainer column is already present. A dataset that already uses `prompt` and `completion` passes a use case that documents `document` and `summary`.
- A dataset that has both an alias and its trainer column (for example `document` and `prompt`) is rejected as ambiguous.
- Optional aliases are renamed when present and never reported missing.
- An identity mapping such as `{"context": "context"}` simply marks the column required.

After renaming, the strategy's `validate_data_format()` must accept the columns:

| Strategy | Trainer columns after renaming |
|---|---|
| `sft` (causal) | `prompt` and `completion`, or `messages`, or `text` |
| `sft` (seq2seq) | `prompt` and `completion` |
| `dpo` | `chosen` and `rejected`, optional `prompt` |
| `grpo` | `prompt` and `answer`; no streaming datasets |
| `qa` | `context`, `question`, `answers` (`{"text": [...], "answer_start": [...]}`) |

So each `DataFormatSpec.column_aliases` should map onto exactly one of those column sets for the strategy that objective resolves to.

## Example

A translation use case on the existing seq2seq task and SFT strategy. It is an illustration, not a built-in.

```python title="recommendation_engine/language_use_cases/translation.py"
from recommendation_engine.language_use_cases.use_case import LanguageUseCase, DataFormatSpec
from utilities.training_objective import TrainingObjective


class TranslationUseCase(LanguageUseCase):
    """Use case for sentence-level machine translation."""

    name = "Translation"
    slug = "translation"
    description = (
        "Train a model to translate text from one language to another. "
        "Provide aligned source and target sentences."
    )
    task_slug = "seq2seq_modeling"
    hub_task_filter = "translation"
    compatible_objectives = [TrainingObjective.SUPERVISED]
    data_formats = {
        TrainingObjective.SUPERVISED: DataFormatSpec(
            user_facing_description=(
                "A dataset with a 'source' column (text in the input language) and a "
                "'target' column (its translation)."
            ),
            column_aliases={
                "source": "prompt",
                "target": "completion",
            },
        ),
    }
```

Then register it in `recommendation_engine/language_use_cases/__init__.py`. Supervised resolves to `sft`, which supports `seq2seq_modeling`, and `translation` is an accepted seq2seq pipeline tag, so nothing else is needed.

If a use case needs a strategy other than the objective default, set `strategy_slug`, as `ExtractiveQAUseCase` does with `strategy_slug = "qa"`. A new strategy or task is a separate change; see [Custom strategies](../strategies/custom-strategies.md). A new task also needs entries in the task-keyed maps in `workflow/jobs.py`, `workflow/training.py`, `workflow/service.py`, `workflow/playground.py` and `web/playground.py`.

## Tests

Add a test file in `tests/unit_tests/`, for example `translation_use_case_test.py`. Keep it offline: patch `RecommendationEngine._query_hub` rather than calling the Hub.

```python
from unittest.mock import patch

from datasets import Dataset

from dataset_manager.dataset_factory import DatasetFactory
from recommendation_engine.language_use_cases import LanguageUseCase
from recommendation_engine.validation_models.recommendation import RecommendedModel
from utilities.hardware_detector import HardwareDetector
from utilities.recommendation_engine import RecommendationEngine
from utilities.training_objective import TrainingObjective
from workflow.service import WorkflowService


def _hardware(vram_gb: float) -> HardwareDetector.HardwareSpecs:
    return HardwareDetector.HardwareSpecs(
        physical_core_count=8, ram_gb=32, disk_storage_gb=500,
        available_disk_storage_gb=200,
        gpus=[{"name": "GPU", "total_memory_gb": vram_gb}],
    )


def test_translation_resolves_to_seq2seq_sft():
    _, strategy, task_slug, strategy_slug, _ = RecommendationEngine._resolve_use_case_and_strategy(
        "translation", TrainingObjective.SUPERVISED
    )
    assert task_slug == "seq2seq_modeling"
    assert strategy_slug == "sft"


def test_translation_is_listed_with_supervised_only():
    use_cases = {u["slug"]: u for u in WorkflowService().list_use_cases()}
    assert use_cases["translation"]["compatible_objectives"] == ["supervised"]


def test_translation_aliases_normalize_to_trainer_columns():
    spec = LanguageUseCase.get_use_case_by_slug("translation").data_formats[TrainingObjective.SUPERVISED]
    dataset = Dataset.from_dict({"source": ["hello"], "target": ["bonjour"]})
    normalized = DatasetFactory.normalize_columns(
        dataset, spec.column_aliases, spec.optional_column_aliases
    )
    assert set(normalized.column_names) == {"prompt", "completion"}


def test_translation_recommendation_without_hub():
    model = RecommendedModel(model_id="org/translator", num_parameters=60_000_000, downloads=10)
    with patch.object(RecommendationEngine, "_query_hub", return_value=([model], False, [])):
        rec = RecommendationEngine.recommend("translation", TrainingObjective.SUPERVISED, _hardware(16))
    assert rec.task_slug == "seq2seq_modeling"
    assert rec.recommended_models[0].model_id == "org/translator"
```

Also:

- Add a flow to `FLOWS` in `scripts/run_training_flow_tests.py` with your documented column names and the matching `model_kind` (`causal`, `seq2seq` or `qa`), then run `python scripts/run_training_flow_tests.py --flow <name>`. This trains a tiny local model end to end through `assemble_and_train()`.
- Never define a throwaway `LanguageUseCase` subclass in a test without removing it from `LanguageUseCase._registry` afterwards. The registry is global and rejects duplicate slugs, so a leftover class breaks later tests.
- Run the whole suite: `uv run pytest`.

## Documentation

A new use case is a user-facing change, so update both doc sets in the same pull request (see [Keeping the two doc sets in sync](contributing.md#keeping-the-two-doc-sets-in-sync)):

| What | Read the Docs | In-app (`web/frontend/src/docs/content/`) |
|---|---|---|
| Use case table, task, objectives | [Use cases and tasks](../configuration/training-tasks.md) | `basics.tsx`, slug `use-cases-and-tasks` |
| Columns and aliases | [Dataset formats](../configuration/dataset-formats.md) | `basics.tsx`, slug `preparing-a-dataset` |
| `task_slug` and `strategy_slug` values, if new | [Responses](../api-reference/responses.md) | `reference.tsx`, slug `recommendation-fields` |

Also update `TRAINING_WORKFLOWS.md` at the repository root if the use case adds a new data contract; it describes the columns for each objective.

## See also

- [Use cases and tasks](../configuration/training-tasks.md)
- [Dataset formats](../configuration/dataset-formats.md)
- [How planning works](../configuration/configuration-guide.md)
- [Custom strategies](../strategies/custom-strategies.md)
- [Architecture](architecture.md)
