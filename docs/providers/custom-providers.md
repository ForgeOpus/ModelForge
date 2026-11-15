# Adding a provider

A provider loads base models and tokenizers for training. This page shows how to write one, register it, let strategies use it, select it for a run, and test it. It assumes a source checkout of ModelForge; see [Development setup](../contributing/development.md).

!!! warning "Code changes required"
    ModelForge 4 has no plugin entry point for providers, and the planner always sets `provider_slug = "huggingface"`. Adding a provider means editing ModelForge's source in three places, described below. Most needs (a local model folder, a private Hub repo, a different cache) are already covered by the [Hugging Face provider](huggingface.md), because any path or ID that `from_pretrained` accepts works as a model ID.

## The interface

`providers/provider.py`:

```python
class Provider(ABC):
    _registry = {}

    name: str = None
    description: str = None
    slug: str = None

    def __init_subclass__(cls, **kwargs): ...   # validates and registers

    @abstractmethod
    def load_model(self, model_name: str, model_class: Any, **kwargs) -> Any: ...

    @abstractmethod
    def load_tokenizer(self, model_name: str) -> Any: ...
```

| Member | Requirement |
|---|---|
| `name` | Display name. Must be a `str`. |
| `description` | One-line description. Must be a `str`. |
| `slug` | Unique identifier used in plans. Must be a `str`. |
| `load_model` | Return a model the strategy's trainer accepts (in practice a Transformers `PreTrainedModel`). |
| `load_tokenizer` | Return a tokenizer with a pad token. |

Defining a subclass registers it immediately in `Provider._registry`. A missing or non-string `name`, `description` or `slug` raises `ValueError` at class definition, and so does a slug that is already registered.

### What callers pass

The provider is instantiated with no arguments (`provider_cls()` in `workflow/training.py`), so `__init__` must not require any.

Strategies then call:

```python
model = provider.load_model(model_id, task_class.model_class, **model_init_kwargs)
tokenizer = provider.load_tokenizer(model_id)
```

- `model_id` is the run's model ID: a Hub ID or a local path.
- `model_class` is the task's Transformers auto class: `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM` or `AutoModelForQuestionAnswering`.
- `model_init_kwargs` comes from the plan's `strategy_config`. It is empty for full fine-tuning and LoRA bf16. For QLoRA it is `{"quantization_config": {...}}` as a **plain dictionary**, because plans are stored as JSON. Your provider must turn it into a `BitsAndBytesConfig`, or the model won't be quantized.
- DPO with full fine-tuning calls `load_model(model_id, model_class)` a second time with no kwargs, for the frozen reference model.

Strategies set a pad token on the tokenizer if it has none and add a fallback chat template when needed. Chat training needs an EOS token.

## Step 1: write the provider

Create `providers/local_mirror.py`. This example loads models from a local directory tree when a copy exists there, and from the Hub otherwise.

```python
"""Load models from a local mirror directory, falling back to the Hub."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from providers.provider import Provider


class LocalMirrorProvider(Provider):

    name = "Local mirror"
    description = "Loads models from a local directory tree, falling back to the Hugging Face Hub."
    slug = "local_mirror"

    def __init__(self) -> None:
        # Instantiated with no arguments. Read settings from the environment.
        # LOCAL_MIRROR_ROOT is this example's own variable, not a ModelForge setting.
        self.root = Path(os.environ.get("LOCAL_MIRROR_ROOT", "/models"))

    def _resolve(self, model_name: str) -> str:
        candidate = self.root / model_name
        return str(candidate) if candidate.is_dir() else model_name

    def load_model(self, model_name: str, model_class: Any, **kwargs) -> PreTrainedModel:
        quantization = kwargs.get("quantization_config")
        if isinstance(quantization, dict):
            kwargs["quantization_config"] = BitsAndBytesConfig(**quantization)
        return model_class.from_pretrained(self._resolve(model_name), **kwargs)

    def load_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(self._resolve(model_name))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
```

If you also want to accept plans from older ModelForge versions, handle top-level `load_in_4bit` / `load_in_8bit` flags the way `providers/huggingface.py` does, and never forward them to `from_pretrained`.

## Step 2: register it at startup

A provider exists only once its module has been imported. Add the import to `load_core_registries()` in `workflow/registries.py`, which the web server, terminal wizard, notebook CLI and `assemble_and_train` all call:

```python
def load_core_registries() -> None:
    import recommendation_engine.language_use_cases  # noqa: F401
    import finetuning_strategies.tasks  # noqa: F401
    import finetuning_strategies.strategies.sft  # noqa: F401
    import finetuning_strategies.strategies.dpo  # noqa: F401
    import finetuning_strategies.strategies.grpo  # noqa: F401
    import finetuning_strategies.strategies.qa  # noqa: F401
    import providers.huggingface  # noqa: F401
    import providers.local_mirror  # noqa: F401
```

Don't import it from `providers/__init__.py`. That module is imported by every strategy, and the registry tests expect importing the base class alone to leave the registry empty.

Check that it registered:

```bash
python -c "from workflow.registries import load_core_registries; load_core_registries(); from providers.provider_factory import ProviderFactory; print(ProviderFactory.get_available_providers())"
```

```text
['huggingface', 'local_mirror']
```

It also appears under `providers` in `GET /api/system`.

## Step 3: allow strategies to use it

Each strategy declares which providers it supports. A run whose plan names an unsupported provider is rejected with `Recommendation provider is not supported by this strategy.` Add your slug to every strategy that should accept it:

```python
# finetuning_strategies/strategies/sft.py
class SFT(FinetuningStrategy):
    supported_providers = ["huggingface", "local_mirror"]
```

Do the same in `dpo.py`, `grpo.py` and `qa.py` as needed. `GET /api/system` shows each strategy's `supported_providers`.

## Step 4: select it for a run

The planner sets `provider_slug = "huggingface"` on every plan (step 6 in `RecommendationEngine.recommend`, `utilities/recommendation_engine.py`). There are two ways to use another provider.

=== "Per run, through the API"

    Generate a plan, change `provider_slug`, and send it with `POST /api/runs`. Use the plan's top model as `model_id`: if you train a different model, the plan is rebuilt and `provider_slug` goes back to `huggingface`.

    ```python
    import requests

    base = "http://127.0.0.1:8000/api"
    headers = {"Authorization": "Bearer <startup token>"}
    plan = requests.post(f"{base}/recommendations", headers=headers, json={
        "use_case_slug": "instruction_following", "objective": "supervised",
    }).json()
    plan["provider_slug"] = "local_mirror"

    run = requests.post(f"{base}/runs", headers=headers, json={
        "use_case_slug": "instruction_following",
        "objective": "supervised",
        "model_id": plan["recommended_models"][0]["model_id"],
        "dataset_path": "./data/train.jsonl",
        "output_dir": "./modelforge_output/mirror-run",
        "recommendation": plan,
    }).json()
    ```

=== "Always, by changing the planner"

    Edit the provider line in `utilities/recommendation_engine.py`:

    ```python
    # Step 6 — Provider
    provider_slug = "local_mirror"
    ```

    Every plan from the console, terminal wizard, notebook CLI and API then uses your provider. Make sure every strategy lists it in `supported_providers`.

The playground doesn't use providers. It loads trained output folders with Transformers directly.

## Step 5: test it

The registry is a class attribute shared by the whole test process, so save and restore it around each test, as `tests/unit_tests/provider_test.py` does. Avoid network access by passing a mock model class (the pattern in `tests/unit_tests/quantized_loading_test.py`) and by building a tiny tokenizer on disk.

Create `tests/unit_tests/local_mirror_provider_test.py`:

```python
from unittest.mock import Mock

import pytest
from transformers import BitsAndBytesConfig, PreTrainedTokenizerFast

from providers.provider import Provider
from providers.provider_factory import ProviderFactory


@pytest.fixture(autouse=True)
def clean_provider_registry():
    original = dict(Provider._registry)
    yield
    Provider._registry.clear()
    Provider._registry.update(original)


@pytest.fixture
def provider_cls():
    import importlib
    import providers.local_mirror as module

    # Re-define the class inside the clean registry for each test.
    Provider._registry.pop("local_mirror", None)
    return importlib.reload(module).LocalMirrorProvider


def test_registers_under_its_slug(provider_cls):
    assert ProviderFactory.get_provider("local_mirror") is provider_cls


def test_quantization_dict_becomes_bitsandbytes_config(provider_cls):
    model_class = Mock()
    provider_cls().load_model("org/model", model_class,
                              quantization_config={"load_in_8bit": True})
    passed = model_class.from_pretrained.call_args.kwargs
    assert isinstance(passed["quantization_config"], BitsAndBytesConfig)
    assert passed["quantization_config"].load_in_8bit is True


def test_prefers_local_copy(provider_cls, tmp_path, monkeypatch):
    (tmp_path / "org" / "model").mkdir(parents=True)
    monkeypatch.setenv("LOCAL_MIRROR_ROOT", str(tmp_path))
    model_class = Mock()
    provider_cls().load_model("org/model", model_class)
    assert model_class.from_pretrained.call_args.args[0] == str(tmp_path / "org" / "model")


def test_falls_back_to_hub_id(provider_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_MIRROR_ROOT", str(tmp_path))
    model_class = Mock()
    provider_cls().load_model("org/missing", model_class)
    assert model_class.from_pretrained.call_args.args[0] == "org/missing"


def test_tokenizer_gets_pad_token(provider_cls, tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    source = tmp_path / "org" / "tok"
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                            eos_token="[EOS]").save_pretrained(source)
    monkeypatch.setenv("LOCAL_MIRROR_ROOT", str(tmp_path))
    tokenizer = provider_cls().load_tokenizer("org/tok")
    assert tokenizer.pad_token == tokenizer.eos_token == "[EOS]"
```

Run it, then the whole suite:

```bash
python -m pytest tests/unit_tests/local_mirror_provider_test.py -q
python -m pytest -q
```

To check a real training step end to end, follow `test_actual_quantized_qwen2_dpo_step` in `tests/unit_tests/quantized_loading_test.py`. It builds a tiny Qwen2 model on disk and calls `strategy.finetune(..., provider=HuggingFaceProvider(), ...)` directly. Pass your provider instead. The quantized variant needs CUDA and is skipped without it.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Unsupported provider type: local_mirror` | The module was never imported. Check `workflow/registries.py`. |
| `Provider with slug 'local_mirror' already exists` | The module was imported twice under different names, or two classes share a slug. |
| `Provider class '...' must have a 'slug' of type str` | A required class attribute is missing or not a string. |
| `Recommendation provider is not supported by this strategy.` | The strategy's `supported_providers` doesn't list your slug. |
| Run uses `huggingface` anyway | The plan was rebuilt, because `model_id` wasn't the plan's top model, or the plan came from the planner unchanged. |
| QLoRA run isn't quantized, or fails on an unexpected `quantization_config` | `load_model` passed the plan's dictionary straight to `from_pretrained`. Convert it to `BitsAndBytesConfig`. |

## See also

- [Provider overview](overview.md)
- [Hugging Face provider](huggingface.md)
- [Custom strategies](../strategies/custom-strategies.md)
- [Architecture](../contributing/architecture.md)
- [Development setup](../contributing/development.md)
