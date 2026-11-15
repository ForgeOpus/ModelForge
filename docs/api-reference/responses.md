# Response shapes

This page documents the JSON bodies the ModelForge API returns: recommendations, runs and their lifecycle, training events over REST and WebSocket, checkpoints, trained models, Playground replies, validation results and errors. Endpoints and request bodies are in the [REST API](rest-api.md) reference.

The examples are real output shapes from ModelForge 4.0.0. Long strings are shortened with `...` where marked.

## Recommendation

Returned by `POST /api/recommendations`, accepted back by `POST /api/runs` as `recommendation`, and stored on each run as `config_json`.

This one was produced for Instruction Following, supervised, on a single 24 GB RTX 4090 with `liger-kernel` installed:

```json
{
  "recommended_models": [
    {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "num_parameters": 8030261248, "downloads": 7123456},
    {"model_id": "Qwen/Qwen2.5-7B-Instruct", "num_parameters": 7615616512, "downloads": 9876543},
    {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "num_parameters": 1543714304, "downloads": 6543210},
    {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "num_parameters": 494032768, "downloads": 5432109}
  ],
  "task_slug": "causal_language_modeling",
  "strategy_slug": "sft",
  "peft_approach": "lora_bf16",
  "provider_slug": "huggingface",
  "strategy_config": {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": true,
    "packing": true,
    "packing_strategy": "bfd",
    "max_length": 1024,
    "use_liger_kernel": true,
    "activation_offloading": false,
    "bf16": true,
    "fp16": false,
    "model_init_kwargs": {},
    "num_train_epochs": 3.0,
    "learning_rate": 2e-05,
    "per_device_eval_batch_size": 8,
    "weight_decay": 0.0,
    "warmup_ratio": 0.0,
    "lr_scheduler_type": "linear",
    "optim": "adamw_torch",
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 500,
    "seed": 42
  },
  "peft_config": {"r": 32, "lora_alpha": 64, "target_modules": "all-linear"},
  "data_format": {
    "user_facing_description": "A dataset with a 'prompt' column (the instruction or user message) and a 'completion' column (the ideal assistant response).",
    "column_aliases": {"instruction": "prompt", "response": "completion"},
    "optional_column_aliases": {}
  },
  "parallelism_strategy": "data_parallel",
  "rationale": {
    "vram_summary": "24.0 GB across 1 GPU(s). Adjusted VRAM (after strategy overhead): 23.99 GB. ",
    "use_case_resolution": "Use case 'instruction_following' with objective 'supervised' → task_slug='causal_language_modeling', strategy_slug='sft'. Compatible because: Instruction Following supports supervised, and sft supports causal_language_modeling.",
    "peft_decision": "Selected tier: lora_bf16 (P_max=10.94B). Compared popular candidates across all feasible tiers; selected the largest shortlisted model and the least quantized training method that fits it. Memory estimates reserve activations and model copies; adapter replicas fit the smallest GPU. Per-tier P_max:\n  full_finetune: P_max=1.22B params\n  lora_bf16: P_max=10.94B params\n  qlora_8bit: P_max=21.89B params\n  qlora_4bit: P_max=43.78B params",
    "parallelism_decision": "DATA_PARALLEL selected: single GPU detected, FSDP requires ≥2 GPUs.",
    "model_filter": "Queried Hub: pipeline_tag='text-generation', sort=downloads desc, search_max_params=43.78B, selected_tier_max_params=10.94B. Found 4 model(s).",
    "warnings": [
      "3 models were skipped due to missing parameter metadata on the Hub."
    ]
  },
  "vram_estimate": {
    "raw_vram_gb": 23.98828125,
    "adjusted_vram_gb": 23.98828125,
    "activation_budget_gb": 2.0,
    "adapter_cost_gb": 0.1,
    "bytes_per_param": 2.0,
    "max_params_by_tier": {
      "full_finetune": 1221571180.5555556,
      "lora_bf16": 10944140625.0,
      "qlora_8bit": 21888281250.0,
      "qlora_4bit": 43776562500.0
    }
  }
}
```

The Hub search returned four models under the 4-bit ceiling. The engine took the largest (8.0B), then the least quantized tier that fits it (LoRA bf16, ceiling 10.94B), and kept only candidates under that tier's ceiling, largest first.

### Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `recommended_models` | list | Up to `n` candidates, each `{model_id, num_parameters, downloads}`. Ordered by parameter count, then downloads, largest first. The first entry is the model `strategy_config` and `peft_config` were built for. Empty when nothing on the Hub fits. |
| `task_slug` | string | `causal_language_modeling`, `seq2seq_modeling` or `question_answering`. |
| `strategy_slug` | string | `sft`, `dpo`, `grpo` or `qa`. |
| `peft_approach` | string | `full_finetune`, `lora_bf16`, `qlora_8bit` or `qlora_4bit`. |
| `provider_slug` | string | Always `huggingface`. |
| `strategy_config` | object or null | Trainer arguments for the first model. `null` when `recommended_models` is empty. Keys per strategy: [Training config](training-config.md). |
| `peft_config` | object or null | `{r, lora_alpha, target_modules}`. `null` for `full_finetune` or when no model was found. |
| `data_format` | object | What the dataset must contain for this use case and objective. `column_aliases` maps your column names to the trainer's (source to destination); a column already named like the destination is accepted too. `optional_column_aliases` are renamed when present but not required. |
| `parallelism_strategy` | string | `data_parallel` or `fsdp`. |
| `rationale` | object | Human-readable reasoning, below. |
| `vram_estimate` | object or null | The numbers behind the tier choice, below. |

`downloads` is the Hub's download count, used for ranking candidates. It is `0` for a model you named yourself with `model_id`.

### rationale

| Field | Content |
|---|---|
| `vram_summary` | Detected VRAM and GPU count, and the adjusted figure after strategy overhead. `CPU-only (no GPU)` or `Apple MPS (8 GB effective)` on those machines. Ends with `Multi-GPU summed.` when more than one GPU is used. Shown in the console as **VRAM estimate**. |
| `use_case_resolution` | How the use case and objective mapped to a task and strategy. |
| `peft_decision` | Selected tier, its parameter ceiling, and the ceiling of every tier. Shown as **PEFT decision**. |
| `parallelism_decision` | Why `data_parallel` or `fsdp` was chosen. |
| `model_filter` | Hub pipeline tag, sort order, search ceiling, tier ceiling and number of models found. Notes a relaxed retry. |
| `warnings` | List of strings. Shown under the plan. |

Warnings the engine can emit:

| Warning (prefix) | When |
|---|---|
| `N models were skipped due to missing parameter metadata on the Hub.` | Hub models without a safetensors parameter count can't be sized and are skipped. |
| `DPO aligns existing behavior. For instruction following, choose an instruction/chat-trained checkpoint...` | Every preference plan. |
| `CPU-only training detected. Training will be extremely slow...` | No GPU. |
| `Apple MPS detected. bitsandbytes 4-bit and 8-bit quantisation are not supported on MPS...` | Apple Silicon. |
| `Multi-GPU detected. FSDP auto-configured across N GPUs...` | Two or more GPUs and FSDP chosen. |
| `Multi-GPU detected (N GPUs). Data parallel selected. To use FSDP for full fine-tuning, set parallelism_strategy_override='fsdp'...` | Two or more GPUs and data parallel chosen. |
| `QLoRA 4-bit was selected because the chosen model exceeds the unquantized and 8-bit training budgets.` | 4-bit chosen although LoRA bf16 had room for some model. |
| `No single-GPU tier is viable for this hardware. Even QLoRA 4-bit may OOM...` | Every ceiling is zero. |
| `No models found on the Hub matching task '...' with at most XB parameters. No strategy config was generated...` | Empty `recommended_models`. |
| `Initial Hub query returned 0 results. Retried with a 50% relaxed parameter upper bound.` | First search found nothing. |
| `Available disk space (X GB) may be insufficient for the largest recommended model...` | Free disk is below the largest candidate's weight size at the chosen tier. |

### vram_estimate

| Field | Meaning |
|---|---|
| `raw_vram_gb` | Total VRAM of the planned GPUs in GiB. Apple MPS counts as 8.0, CPU-only as 0. |
| `adjusted_vram_gb` | After the strategy's extra model copies and batch multiplier: `(raw − (batch_multiplier − 1) × 2.0) / (1 + extra_model_copies)`. For adapter tiers on multiple GPUs this is the smallest GPU's adjusted figure, because each data-parallel replica must fit on one card. |
| `activation_budget_gb` | Always `2.0`. |
| `adapter_cost_gb` | Always `0.1`. |
| `bytes_per_param` | Weight cost at the chosen tier: 18, 2, 1 or 0.5. |
| `max_params_by_tier` | Parameter ceiling for each tier, in parameters (not billions). |

On CPU-only machines every tier except `qlora_4bit` is 0, and `qlora_4bit` is the smaller of 500M and RAM ÷ 3 bytes. On Apple MPS the two quantized tiers are 0.

## Runs

Returned by `GET /api/runs`, `GET /api/runs/{id}`, `POST /api/runs`, `POST /api/runs/{id}/cancel` and `POST /api/runs/{id}/resume`.

```json
{
  "id": 7,
  "status": "running",
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "use_case": "instruction_following",
  "objective": "supervised",
  "task_slug": "causal_language_modeling",
  "strategy": "sft",
  "peft": "full_finetune",
  "dataset": "./data/train.jsonl",
  "output_dir": "./runs/qwen-sft-01",
  "config_json": "{\"recommended_models\":[{\"model_id\":\"Qwen/Qwen2.5-0.5B-Instruct\",...}],...}",
  "tensorboard_log_dir": "runs/qwen-sft-01/tensorboard/run-7",
  "error": null,
  "created_at": "2026-11-15T03:12:44.581203",
  "updated_at": "2026-11-15T03:12:45.020917",
  "launch_json": "{\"gpu_indices\": [], \"dataset_remote\": false}",
  "tensorboard_url": "http://127.0.0.1:6006/"
}
```

| Field | Type | Meaning |
|---|---|---|
| `id` | int | Run ID. |
| `status` | string | See [lifecycle](#run-lifecycle). |
| `model_id` | string | Base model. |
| `use_case` | string | Use case slug. |
| `objective` | string | `supervised`, `preference` or `reinforcement`. |
| `task_slug`, `strategy`, `peft` | string | From the plan actually used. |
| `dataset` | string | `dataset_path` as sent. |
| `output_dir` | string | As sent. |
| `config_json` | string | The full recommendation used, as a JSON **string**. Parse it to read `strategy_config`. |
| `tensorboard_log_dir` | string or null | This run's own log folder, `<log dir>/run-<id>`. `null` when TensorBoard is off. |
| `error` | string or null | The exception message for failed runs, and sometimes for cancelled ones. |
| `created_at`, `updated_at` | string | ISO 8601, UTC, no zone suffix. |
| `launch_json` | string | JSON string with the `gpu_indices` and `dataset_remote` used, replayed on resume. |
| `tensorboard_url` | string | Only on `GET /api/runs` and `GET /api/runs/{id}`, and only when TensorBoard is on. Where TensorBoard is served on the server machine (default `http://127.0.0.1:6006/`). |

### Run lifecycle

| Status | Meaning |
|---|---|
| `queued` | Created; the training thread hasn't started yet. Usually lasts a fraction of a second. |
| `running` | The trainer is loading or training. |
| `completed` | Training finished and the model was saved. A [model record](#models) is created if the output folder exists. |
| `cancelled` | Stopped after a cancel request. Checkpoints written so far remain. Not registered as a model. |
| `failed` | An exception ended the run. `error` holds the message. |

```text
queued ──► running ──► completed
                  ├──► cancelled   (POST /cancel, stops after the current step)
                  └──► failed      (exception, or the server stopped mid-run)
```

When the server starts, it marks every `queued` or `running` run in the database as `failed` with:

```text
ModelForge stopped before this run reached a terminal state.
```

Resuming always creates a new run. The old one keeps its status.

## Events

Training events have one shape on both `GET /api/runs/{id}/events` and the WebSocket:

```json
{
  "run_id": 7,
  "type": "log",
  "message": "Training step 120",
  "payload": {
    "step": 120,
    "max_steps": 1875,
    "logs": {"loss": 1.4213, "grad_norm": 0.82, "learning_rate": 1.87e-05, "mean_token_accuracy": 0.641, "epoch": 0.19}
  },
  "ts": 1794712589.204
}
```

| Field | Type | Meaning |
|---|---|---|
| `run_id` | int | |
| `type` | string | See below. |
| `message` | string | Human-readable; may be empty. |
| `payload` | object | Type-specific; `{}` when there is nothing to add. |
| `ts` | float | Unix time in **seconds** from the server clock. Multiply by 1000 for JavaScript `Date`. |

`GET /api/runs/{id}/events` wraps them as `{"events": [...]}`, oldest first.

### Event types

| `type` | `message` | `payload` |
|---|---|---|
| `status` | `Training started.` | `{}` |
| `log` | `Training step N` | `step`, `max_steps`, `logs`: whatever the trainer logged (`loss`, `grad_norm`, `learning_rate`, `epoch`, and strategy metrics such as `mean_token_accuracy` for SFT or reward statistics for GRPO). The final log of a run holds summary values such as `train_runtime` and `train_loss` instead of a per-step loss. |
| `log` | `Output directory not found, skipping registration: <path>`, `Could not write modelforge_config.json: <error>`, `Could not register the trained model: <error>` | `{}`. Housekeeping notices after training; the run still completes. |
| `checkpoint` | `Checkpoint saved at step N` | `path`, `step` |
| `cancel` | `Cancellation requested.` (from the API) or `Cancellation requested; stopping after current step.` (from the trainer) | `{}` |
| `completed` | `Training completed.` | `output_dir` |
| `cancelled` | `Training cancelled.` | `output_dir`, or `error` if stopping raised an exception |
| `error` | The exception message | `error`: the same message |
| `heartbeat` | absent | absent. WebSocket only; sent after one second with no other event. Has only `run_id` and `type`. |

A run ends with exactly one of `completed`, `cancelled` or `error`.

## Checkpoints

`GET /api/runs/{id}/checkpoints`:

```json
{
  "checkpoints": [
    {"id": 4, "run_id": 7, "path": "runs/qwen-sft-01/checkpoint-1000", "step": 1000, "created_at": "2026-11-15T03:41:10.118024"},
    {"id": 3, "run_id": 7, "path": "runs/qwen-sft-01/checkpoint-500", "step": 500, "created_at": "2026-11-15T03:26:52.700311"}
  ]
}
```

Highest step first. `path` is `<output_dir>/checkpoint-<step>`, built from the `output_dir` you sent (normalized, so `./runs/x` becomes `runs/x`). It is relative if that was relative, and relative paths resolve against the server's working directory. Pass it unchanged to `/resume` or `/api/playground/infer`.

## Models

`GET /api/models` returns `{"models": [...]}`; `GET /api/models/{id}` returns one:

```json
{
  "id": 2,
  "run_id": 7,
  "name": "Qwen2.5-0.5B-Instruct-7",
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "task_slug": "causal_language_modeling",
  "use_case": "instruction_following",
  "peft_approach": "full_finetune",
  "model_path": "runs/qwen-sft-01",
  "size_bytes": 3967402311,
  "pushed_to_hub": false,
  "hub_repo_id": null,
  "created_at": "2026-11-15T04:02:31.554110"
}
```

| Field | Meaning |
|---|---|
| `name` | Last part of the base model ID plus the run ID. |
| `model_path` | The run's output folder. Unique: a later completed run that writes to the same folder updates this record (same `id`) instead of adding a new one. |
| `size_bytes` | Total size of the folder when registered, including checkpoints and logs. `null` if it couldn't be measured. |
| `pushed_to_hub`, `hub_repo_id` | Set by `POST /api/models/{id}/push`. |

Other model endpoints return:

```json
{"deleted": true, "files_removed": false}
```

```json
{"pushed": true, "repo_id": "your-name/qwen-sft", "url": "https://huggingface.co/your-name/qwen-sft"}
```

Each output folder also contains `modelforge_config.json`, which is what the Playground uses to reload it:

```json
{
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "task_slug": "causal_language_modeling",
  "use_case": "instruction_following",
  "objective": "supervised",
  "strategy": "sft",
  "peft_approach": "full_finetune",
  "model_class": "AutoModelForCausalLM",
  "pipeline_task": "text-generation",
  "modelforge_version": "4.0.0"
}
```

## Playground

### POST /api/playground/infer

```json
{
  "reply": "LoRA freezes the base weights and trains two small matrices beside each layer.",
  "task": "causal_language_modeling",
  "input_tokens": 38,
  "context_limit": 4096,
  "device": "cuda:0",
  "training_active": false
}
```

| Field | Meaning |
|---|---|
| `reply` | Generated text. `(No output)` when the model produced nothing. For extractive QA, the answer span from the context, or `(No answer span)`. For token classification, one `token: label` line per token. |
| `task` | The run's task slug. |
| `input_tokens` | Tokens in the conversation sent to the model. |
| `context_limit` | The limit that applied: the smallest of the requested window, the model's maximum positions and the tokenizer's limit. |
| `device` | Where the model ran: a single device such as `cuda:0` or `cpu`, or a combination such as `GPU + CPU` or `CPU + disk`. |
| `training_active` | Whether a run was training, in which case GPUs were not used. |

### Streaming sessions

`GET /api/playground/status`:

```json
{"available": true, "reason": null, "session_id": null, "model_path": null}
```

`POST /api/playground/sessions`:

```json
{"session_id": "4f0c2a9e6d1b4b5e8a7c3f2d1e0b9a8c", "model_path": "./runs/qwen-sft-01", "base_model": "Qwen/Qwen2.5-0.5B-Instruct", "note": null}
```

`DELETE /api/playground/sessions/{id}`:

```json
{"closed": true}
```

Chat WebSocket messages from the server: `{"type": "start"}`, then any number of `{"type": "token", "text": "..."}`, then `{"type": "done"}`, or `{"type": "error", "message": "..."}` on failure. See [REST API: chat WebSocket](rest-api.md#ws-apiplaygroundsession_idchat).

## Validation

### Dataset validation

`POST /api/datasets/validate`, valid:

```json
{
  "valid": true,
  "path": "./data/train.jsonl",
  "remote": false,
  "columns": ["prompt", "completion"],
  "format_description": "A dataset with a 'prompt' column (the instruction or user message) and a 'completion' column (the ideal assistant response).",
  "error": null,
  "row_count": 2
}
```

Invalid:

```json
{
  "valid": false,
  "path": "./data/train.jsonl",
  "remote": false,
  "columns": [],
  "format_description": "A dataset with a 'prompt' column (the instruction or user message) and a 'completion' column (the ideal assistant response).",
  "error": "Dataset is missing required column(s): instruction, response.",
  "row_count": null
}
```

| Field | Meaning |
|---|---|
| `columns` | Column names after aliases were applied (for example `instruction` already renamed to `prompt`). Empty when invalid. |
| `format_description` | The use case's description of the expected data. `null` without `use_case_slug`. |
| `row_count` | Rows in the first split. `null` on failure. |
| `error` | Why it failed, or `null`. |

The missing-column message names the **alias** columns of the use case, not the trainer's names. `missing ... instruction, response` is satisfied by either `instruction`/`response` or `prompt`/`completion`. Common `error` values are listed in [Common issues](../troubleshooting/common-issues.md#dataset-validation-errors).

### Model validation

`POST /api/models/validate`, valid:

```json
{"valid": true, "model_id": "Qwen/Qwen2.5-0.5B-Instruct", "num_parameters": 494032768, "pipeline_tag": "text-generation", "error": null}
```

Invalid responses always have `valid: false` and an `error`. The possible errors:

| `error` | Cause |
|---|---|
| `Enter a model ID, for example org/model.` | Empty `model_id`. |
| ``No model with that ID is visible on the Hub. Check the spelling, or run `hf auth login` if it is private or gated.`` | The Hub answered 401, 403 or 404. |
| `The Hub returned an error (<status>).` | Any other Hub HTTP error. |
| `This model is tagged '<tag>', which is incompatible with task '<task>'. Expected one of: <tags>.` | Wrong pipeline tag for the task. This response also includes `num_parameters` and `pipeline_tag`. |
| `Could not reach the Hugging Face Hub. Check your connection.` | Network or other failure. |

`valid: true` with `num_parameters: null` means the model exists but publishes no safetensors parameter count. Planning and training refuse such models.

## System

`GET /api/system` (strategy descriptions and the use-case list shortened):

```json
{
  "name": "ModelForge",
  "package": "modelforge-finetuning",
  "version": "4.0.0",
  "providers": ["huggingface"],
  "strategies": [
    {"name": "Supervised Fine-Tuning", "slug": "sft", "description": "...", "supported_providers": ["huggingface"], "supported_tasks": ["causal_language_modeling", "seq2seq_modeling"], "extra_model_copies": 0, "batch_multiplier": 1.0},
    {"name": "Direct Preference Optimization", "slug": "dpo", "description": "...", "supported_providers": ["huggingface"], "supported_tasks": ["causal_language_modeling"], "extra_model_copies": 1, "batch_multiplier": 2.0},
    {"name": "Group Relative Policy Optimization", "slug": "grpo", "description": "...", "supported_providers": ["huggingface"], "supported_tasks": ["causal_language_modeling"], "extra_model_copies": 0, "batch_multiplier": 4.0},
    {"name": "Extractive Question Answering", "slug": "qa", "description": "...", "supported_providers": ["huggingface"], "supported_tasks": ["question_answering"], "extra_model_copies": 0, "batch_multiplier": 1.0}
  ],
  "use_cases": ["... same entries as GET /api/use-cases ..."],
  "unavailable_use_cases": []
}
```

`unavailable_use_cases` lists registered use cases that have no runnable objective, as `{name, slug, reasons}`, where `reasons` maps each objective to why it can't run.

## Use cases

One entry of `GET /api/use-cases`:

```json
{
  "name": "Text Generation",
  "slug": "text_generation",
  "description": "Train a model to generate coherent, contextually relevant text given an input prompt. ...",
  "task_slug": "causal_language_modeling",
  "hub_task_filter": "text-generation",
  "compatible_objectives": ["supervised", "preference", "reinforcement"],
  "unavailable_objectives": {},
  "data_formats": {
    "supervised": {
      "user_facing_description": "A dataset with a 'prompt' column (the input text the model receives) and a 'completion' column (the target text the model should generate).",
      "column_aliases": {"input_text": "prompt", "generated_text": "completion"},
      "optional_column_aliases": {}
    },
    "preference": {
      "user_facing_description": "A dataset with a 'chosen' column (the preferred continuation) and a 'rejected' column (a less-preferred continuation). ...",
      "column_aliases": {"preferred_continuation": "chosen", "rejected_continuation": "rejected"},
      "optional_column_aliases": {"input_text": "prompt"}
    },
    "reinforcement": {
      "user_facing_description": "A dataset with a 'prompt' column (the text prefix) and an 'answer' column (the exact desired continuation). ...",
      "column_aliases": {"input_text": "prompt", "ideal_continuation": "answer"},
      "optional_column_aliases": {}
    }
  }
}
```

`compatible_objectives` lists only objectives that can actually run. `unavailable_objectives` maps any declared but unrunnable objective to the reason.

## Hardware

`GET /api/hardware`:

```json
{
  "physical_core_count": 16,
  "disk_storage_gb": 1862.5,
  "available_disk_storage_gb": 912.3,
  "ram_gb": 62.7,
  "gpus": [{"name": "NVIDIA GeForce RTX 4090", "total_memory_gb": 23.98828125}]
}
```

Sizes are GiB. NVIDIA GPUs are read through NVML. On Apple Silicon, `gpus` is `[{"name": "Apple Silicon GPU", "total_memory_gb": "N/A"}]`. With no usable GPU it is `[]`.

## Errors

Every error is `{"detail": ...}`. ModelForge's own errors have a string:

```json
{"detail": "Run not found."}
```

Request validation errors (`422`) have FastAPI's list form:

```json
{
  "detail": [
    {
      "type": "literal_error",
      "loc": ["body", "context_window"],
      "msg": "Input should be 1024, 4096, 8016 or 32768",
      "input": 2048,
      "ctx": {"expected": "1024, 4096, 8016 or 32768"}
    }
  ]
}
```

Unhandled server errors return `500` with a plain-text `Internal Server Error` body. The traceback is in the terminal running `modelforge run`.

## See also

- [REST API](rest-api.md)
- [Training config](training-config.md)
- [Checkpoints](../user-guide/checkpoints.md)
- [Common issues](../troubleshooting/common-issues.md)
