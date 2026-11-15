# Hugging Face provider

The Hugging Face provider (`providers/huggingface.py`, slug `huggingface`) loads every base model and tokenizer ModelForge trains, using Transformers. This page covers how it loads models, how authentication and gated models work, where files are cached, and how ModelForge uses the Hub for model search and publishing.

## Model loading

`HuggingFaceProvider.load_model(model_name, model_class, **kwargs)` calls `model_class.from_pretrained(model_name, **kwargs)`.

- `model_name` is a Hub ID (`org/model`) or a local folder that `from_pretrained` accepts.
- `model_class` comes from the task: `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM` or `AutoModelForQuestionAnswering`. See [Use cases and tasks](../configuration/training-tasks.md).
- `kwargs` are the plan's `strategy_config.model_init_kwargs`.

### Quantization

For QLoRA tiers the plan stores quantization as a plain dictionary so it survives JSON:

=== "qlora_8bit"

    ```json
    {"quantization_config": {"load_in_8bit": true}}
    ```

=== "qlora_4bit"

    ```json
    {"quantization_config": {"load_in_4bit": true, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": true}}
    ```

The provider turns a `quantization_config` dictionary into a `transformers.BitsAndBytesConfig` before calling `from_pretrained`. Full fine-tuning and LoRA bf16 plans have empty `model_init_kwargs`.

Plans saved by older ModelForge versions may use top-level `load_in_4bit` / `load_in_8bit` flags instead. The provider still accepts them and converts them to a `BitsAndBytesConfig` (4-bit uses NF4 with double quantization). It never forwards those removed keywords to Transformers. It raises `ValueError` if both flags are set, or if the flags and a `quantization_config` are both present.

bitsandbytes quantization needs CUDA. The planner never chooses a QLoRA tier on Apple MPS, and strips quantization from CPU-only plans. See [LoRA and QLoRA](../strategies/qlora.md).

### Tokenizer

`load_tokenizer(model_name)` calls `AutoTokenizer.from_pretrained(model_name)`. If the tokenizer has no pad token, the provider sets it to the EOS token. Chat templates are added later by the strategy when needed. See [Dataset formats](../configuration/dataset-formats.md#chat-formatting).

### Remote code

ModelForge never sets `trust_remote_code`, so Transformers' default (off) applies to both the model and the tokenizer. Models that need custom code from their repository generally can't be loaded for training.

## Authentication and gated models

Public models and datasets need no login. For gated models (for example Llama), accept the licence on the model's Hub page, then log in on the machine that runs ModelForge:

```bash
hf auth login
```

Or set a token in the environment of the process that starts ModelForge:

```bash
export HF_TOKEN=hf_...
modelforge run
```

Everything ModelForge does on the Hub (model search, model ID validation, downloads, dataset loading, pushing) uses `huggingface_hub`'s standard token lookup, so a cached login or `HF_TOKEN` applies everywhere.

!!! note "`huggingface-cli` no longer works"
    Older guides use `huggingface-cli login`. With `huggingface-hub` 1.x, which ModelForge 4 requires, that command only prints a deprecation notice. Use `hf auth login`.

The terminal wizard (`modelforge cli`) checks for a token at startup. If it finds neither `HF_TOKEN` nor a cached login, it offers to log in with a pasted token, or to continue with public models and datasets only.

When a hand-entered model ID isn't visible, validation returns `No model with that ID is visible on the Hub...`. The Hub answers 401 rather than 404 for an unknown repo when you are not logged in, so this message covers misspellings, private repos and gated models you haven't been granted.

## Cache

The provider doesn't set a cache directory. Base models download to the standard Hugging Face cache, and Hub datasets to the `datasets` cache. Both follow the usual Hugging Face environment variables, for example:

| Variable | Effect |
|---|---|
| `HF_HOME` | Root for all Hugging Face caches and the saved token |
| `HF_HUB_CACHE` | Where model repositories are stored |
| `HF_DATASETS_CACHE` | Where processed datasets are stored |
| `HF_HUB_OFFLINE=1` | Use only cached files |

Set them before starting ModelForge. Trained models are not written to the cache; they go to each run's output directory.

!!! warning "Offline use"
    Planning searches the Hub, and validating a model ID calls the Hub API. Both fail offline. You can still train a model that is already cached if you send a complete plan to `POST /api/runs`.

## Hub search

Model recommendations come from `HfApi().list_models`, filtered by the use case's `pipeline_tag` and sorted by downloads. Only models that publish a parameter count in their safetensors metadata can be planned. See [How planning works](../configuration/configuration-guide.md#step-5-hub-model-search) for the full filter and ranking.

Hand-entered IDs are checked with `HfApi().model_info` for visibility, pipeline tag and parameter count. See [Using your own model ID](../configuration/configuration-guide.md#using-your-own-model-id).

## Push to Hub

A completed model can be uploaded through the HTTP API. There is no push button in the browser console.

```bash
curl -s -X POST http://127.0.0.1:8000/api/models/3/push \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"repo_id": "your-name/my-model", "private": true}'
```

```json
{"pushed": true, "repo_id": "your-name/my-model", "url": "https://huggingface.co/your-name/my-model"}
```

- `3` is the model's ID from `GET /api/models`, not the run ID.
- The repository is created if it doesn't exist (`private` defaults to `false`), then the whole output folder is uploaded.
- The token must have write access to the namespace.
- A missing output folder returns 409. A Hub error returns 502 with the Hub's message.

!!! warning "The whole folder is uploaded"
    That includes `checkpoint-*` folders, TensorBoard logs and `modelforge_config.json`. Delete what you don't want to publish first.

For LoRA and QLoRA runs the output folder holds only the adapter and tokenizer. Anyone loading it needs access to the base model too.

## Platform support

| Platform | Full fine-tuning | LoRA bf16 | QLoRA 8-bit / 4-bit | Precision |
|---|---|---|---|---|
| NVIDIA CUDA | Yes | Yes | Yes | bf16 on compute capability 8.0+, else fp16 |
| Apple MPS | Yes | Yes | No | fp32 |
| CPU-only | No | LoRA on fp32 weights | No | fp32 |

See [Hardware and memory](../configuration/hardware-profiles.md).

## See also

- [Provider overview](overview.md)
- [Adding a provider](custom-providers.md)
- [Post-installation setup](../installation/post-installation.md)
- [Common issues](../troubleshooting/common-issues.md)
