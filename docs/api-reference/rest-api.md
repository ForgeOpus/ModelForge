# REST API

This page covers every HTTP and WebSocket endpoint served by `modelforge run`: authentication, request bodies, status codes and examples. The browser console uses this same API, so anything the console does can be scripted. Response bodies are documented field by field in [Response shapes](responses.md).

## Base URL

The API is served by the same process and port as the console:

```text
http://<host>:<port>/api
```

`modelforge run` binds `0.0.0.0:8000` by default. Change it with `--host` and `--port` (see [CLI reference](cli.md)). The interactive OpenAPI pages that FastAPI generates are also available at `/docs` and `/openapi.json`. They are not behind the token.

## Authentication

A random token is generated each time the server starts and printed with the console link:

```text
$ modelforge run
ModelForge v4 browser UI
Serving on: http://0.0.0.0:8000
Open:       http://127.0.0.1:8000/app?token=3q2-7Yx...
The startup token is required for API and WebSocket access.
```

Pass `--token <value>` to keep the token fixed across restarts, which is what you want for scripts and CI.

Every `/api/...` endpoint except `GET /api/health` requires the token. HTTP endpoints accept it in any of three places:

| Where | Form |
|---|---|
| `Authorization` header | `Authorization: Bearer <token>` |
| Custom header | `X-ModelForge-Token: <token>` |
| Query string | `?token=<token>` |

If more than one is present, a `Bearer` header wins, then the query string, then `X-ModelForge-Token`. The token is compared in constant time. A missing or wrong token returns:

```json
HTTP/1.1 401 Unauthorized
{"detail": "Invalid or missing ModelForge startup token."}
```

WebSockets only accept the query string form (`?token=<token>`). A bad token closes the socket with code `1008` (policy violation) before it is accepted.

Not authenticated: `GET /api/health`, the console itself (`/`, `/app`, `/app/...`), its static files under `/assets`, and FastAPI's `/docs`, `/redoc` and `/openapi.json`.

!!! warning "The server listens on every interface by default"
    With the default `--host 0.0.0.0`, anyone on your network who has the token can start runs, read file paths and push models to the Hub with your cached login. Bind to `127.0.0.1` on shared networks and reach remote machines through an SSH tunnel. There is no TLS; put a reverse proxy in front if you need HTTPS.

### CORS

The app installs FastAPI's CORS middleware with `allow_origins=["*"]`, all methods and all headers, and no credentials. A page on another origin can call the API, but only if it has the token. The token is the only access control.

## Conventions

- Request and response bodies are JSON. Send `Content-Type: application/json` on POST requests.
- Path IDs (`run_id`, `model_id`) are integers. A non-integer ID returns `422`.
- Timestamps in run, checkpoint and model records are ISO 8601 strings in UTC without a zone suffix, for example `2026-11-15T03:12:44.581203`. Event `ts` values are Unix seconds (floats), not milliseconds.

### Error format

Errors raised by ModelForge use FastAPI's standard shape with a string `detail`:

```json
{"detail": "A training run is already active. Cancel or wait for it before starting another."}
```

Request bodies that fail validation return `422` with a list in `detail`:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "output_dir"],
      "msg": "Field required",
      "input": {"use_case_slug": "instruction_following", "model_id": "Qwen/Qwen2.5-0.5B-Instruct", "dataset_path": "data.jsonl"}
    }
  ]
}
```

### Status codes

| Code | Meaning in ModelForge |
|---|---|
| `200` | Success. Every successful call returns `200`, including ones that create runs. |
| `400` | The request was well formed but can't be acted on: a recommendation that doesn't match the use case, an unknown checkpoint, a GPU index out of range, a refused delete. |
| `401` | Missing or wrong token. |
| `404` | Run, model, checkpoint or active job not found. |
| `409` | Conflict with current state: a run is already training, the Playground is holding model resources, a model folder is gone, not enough memory for inference. |
| `422` | Body validation failed, or the recommendation engine rejected the request (unknown use case, incompatible objective, unusable model ID). |
| `500` | Unexpected server error, including Hub network failures during a recommendation and inference failures. |
| `502` | Hub upload failed (`POST /api/models/{id}/push`). |
| `503` | The Playground can't verify resource use or reach the offload directory. |

## Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| GET | [`/api/health`](#get-apihealth) | Liveness and version. No auth. |
| GET | [`/api/system`](#get-apisystem) | Version, providers, strategies, use cases. |
| GET | [`/api/hardware`](#get-apihardware) | Detected GPUs, RAM and disk. |
| GET | [`/api/use-cases`](#get-apiuse-cases) | Use cases with objectives and data formats. |
| POST | [`/api/recommendations`](#post-apirecommendations) | Build a training plan. |
| POST | [`/api/datasets/validate`](#post-apidatasetsvalidate) | Check a dataset against an objective. |
| POST | [`/api/models/validate`](#post-apimodelsvalidate) | Check a Hub model ID. |
| POST | [`/api/runs`](#post-apiruns) | Start a training run. |
| GET | [`/api/runs`](#get-apiruns) | List runs. |
| GET | [`/api/runs/{run_id}`](#get-apirunsrun_id) | One run. |
| POST | [`/api/runs/{run_id}/cancel`](#post-apirunsrun_idcancel) | Stop an active run after the current step. |
| GET | [`/api/runs/{run_id}/events`](#get-apirunsrun_idevents) | Recorded event history. |
| WS | [`/api/runs/{run_id}/events`](#ws-apirunsrun_idevents) | Live event stream. |
| GET | [`/api/runs/{run_id}/checkpoints`](#get-apirunsrun_idcheckpoints) | Saved checkpoints. |
| POST | [`/api/runs/{run_id}/resume`](#post-apirunsrun_idresume) | Start a new run from a checkpoint. |
| GET | [`/api/models`](#get-apimodels) | Registered trained models. |
| GET | [`/api/models/{model_id}`](#get-apimodelsmodel_id) | One model. |
| DELETE | [`/api/models/{model_id}`](#delete-apimodelsmodel_id) | Remove from the registry, optionally deleting files. |
| POST | [`/api/models/{model_id}/push`](#post-apimodelsmodel_idpush) | Upload the model folder to the Hub. |
| GET | [`/api/playground/status`](#get-apiplaygroundstatus) | Whether a streaming session can open. |
| POST | [`/api/playground/sessions`](#post-apiplaygroundsessions) | Load a model into a streaming session. |
| DELETE | [`/api/playground/sessions/{session_id}`](#delete-apiplaygroundsessionssession_id) | Unload it. |
| WS | [`/api/playground/{session_id}/chat`](#ws-apiplaygroundsession_idchat) | Stream generated tokens. |
| POST | [`/api/playground/infer`](#post-apiplaygroundinfer) | One-shot inference against a run or checkpoint. |

The examples below assume:

```bash
export MF=http://127.0.0.1:8000
export TOKEN=my-fixed-token   # started with: modelforge run --token my-fixed-token
```

## System and planning

### GET /api/health

Liveness check. No token required.

```bash
curl $MF/api/health
```

```json
{"status": "healthy", "version": "4.0.0"}
```

### GET /api/system

Package name and version, registered providers, every registered strategy, the selectable use cases, and use cases that are registered but have no runnable objective (with the reason).

```bash
curl -H "X-ModelForge-Token: $TOKEN" $MF/api/system
```

See [Response shapes: system](responses.md#system) for the full body.

### GET /api/hardware

What the server detected: physical CPU cores, total and free disk, total RAM, and each GPU's name and total VRAM in GiB. Apple Silicon reports one GPU named `Apple Silicon GPU` with `total_memory_gb: "N/A"`. A CPU-only machine returns an empty `gpus` list.

```json
{
  "physical_core_count": 16,
  "disk_storage_gb": 1862.5,
  "available_disk_storage_gb": 912.3,
  "ram_gb": 62.7,
  "gpus": [{"name": "NVIDIA GeForce RTX 4090", "total_memory_gb": 23.99}]
}
```

Disk figures are for the filesystem root (`/`; on Windows, the root of the current drive), not the output directory.

### GET /api/use-cases

```json
{"use_cases": [ { "slug": "instruction_following", "...": "..." } ]}
```

Each entry has `name`, `slug`, `description`, `task_slug`, `hub_task_filter`, `compatible_objectives`, `unavailable_objectives` and `data_formats` (one per objective). A use case with no runnable objective is left out of this list and shows up under `unavailable_use_cases` in `/api/system`. See [Response shapes: use cases](responses.md#use-cases).

The built-in slugs are `instruction_following`, `text_generation`, `summarization` and `extractive_qa`.

### POST /api/recommendations

Builds a plan from the detected hardware, a use case and an objective. This queries the Hugging Face Hub, so it needs network access.

Request body (`RecommendationRequest`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `use_case_slug` | string | required | One of the slugs from `/api/use-cases`. |
| `objective` | string | `"supervised"` | `supervised`, `preference` or `reinforcement`. |
| `model_id` | string or null | `null` | Plan for this exact Hub model instead of searching. It is validated first (see [`/api/models/validate`](#post-apimodelsvalidate)) and must publish a parameter count. |
| `gpu_indices` | list of int or null | `null` | Plan against a subset of GPUs. `null` or `[]` uses all detected GPUs. |
| `n` | int | `5` | Maximum number of candidate models. Must be positive. |

```bash
curl -H "X-ModelForge-Token: $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"use_case_slug":"instruction_following","objective":"supervised"}' \
     $MF/api/recommendations
```

Returns a recommendation object. See [Response shapes: recommendation](responses.md#recommendation).

| Status | When |
|---|---|
| `200` | Plan built. It may still contain no models; check `recommended_models` and `rationale.warnings`. |
| `422` | Unknown use case, an objective the use case doesn't support, `n <= 0`, a GPU index out of range (`GPU indices must be between 0 and N.`), an unusable `model_id` (the validation error text is returned), `Selected model exceeds the available training memory budget.`, or `Strategy '<slug>' requires more VRAM than is available on this hardware for any viable model size.` |
| `500` | The Hub search itself failed, for example with no network. |

### POST /api/datasets/validate

Loads a dataset the same way training will, applies the use case's column aliases, and runs the strategy's format check.

Request body (`DatasetValidationRequest`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `path` | string | required | A local file or directory, or a Hub dataset ID when `remote` is true. |
| `remote` | bool | `false` | Load from the Hugging Face Hub. |
| `use_case_slug` | string or null | `null` | When set, columns are checked against this use case and objective. When omitted, only loading is checked. |
| `objective` | string | `"supervised"` | Used with `use_case_slug`. |

```bash
curl -H "X-ModelForge-Token: $TOKEN" -H "Content-Type: application/json" \
     -d '{"path":"./data/train.jsonl","use_case_slug":"instruction_following","objective":"supervised"}' \
     $MF/api/datasets/validate
```

Always returns `200`; problems are reported in the body with `valid: false` and an `error` string. The one exception is an unknown `use_case_slug`, which is not caught and returns `500`. See [Response shapes: dataset validation](responses.md#dataset-validation).

Local paths are resolved on the server, relative to the directory `modelforge run` was started from.

### POST /api/models/validate

Checks a hand-entered Hub model ID before a run is committed to it.

| Field | Type | Default | Notes |
|---|---|---|---|
| `model_id` | string | required | For example `Qwen/Qwen2.5-0.5B-Instruct`. |
| `task_slug` | string or null | `null` | When set, the model's `pipeline_tag` must match the task. |

Task to accepted pipeline tags:

| `task_slug` | Accepted `pipeline_tag` |
|---|---|
| `causal_language_modeling` | `text-generation` |
| `seq2seq_modeling` | `summarization`, `translation`, `text2text-generation` |
| `question_answering` | `question-answering` |
| `token_classification` | `token-classification` |

A model with no `pipeline_tag` on the Hub is accepted.

```bash
curl -H "X-ModelForge-Token: $TOKEN" -H "Content-Type: application/json" \
     -d '{"model_id":"facebook/bart-large-cnn","task_slug":"seq2seq_modeling"}' \
     $MF/api/models/validate
```

Always returns `200` with `valid` true or false. See [Response shapes: model validation](responses.md#model-validation) for every error message.

## Runs

### POST /api/runs

Starts a training run in a background thread and returns the new run record.

Request body (`TrainingRunRequest`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `use_case_slug` | string | required | |
| `objective` | string | `"supervised"` | `supervised`, `preference` or `reinforcement`. |
| `model_id` | string | required | Hub ID of the base model. Must not be blank. |
| `dataset_path` | string | required | Local path on the server, or Hub dataset ID. Must not be blank. |
| `dataset_remote` | bool | `false` | Load `dataset_path` from the Hub. |
| `output_dir` | string | required | Where checkpoints and the final model go. Must not be blank. Give each run its own. |
| `gpu_indices` | list of int | `[]` | Train on these GPUs. Empty uses all detected GPUs. One process is launched per GPU. |
| `tensorboard` | bool | `true` | Write TensorBoard logs. |
| `tensorboard_log_dir` | string or null | `null` | Where to write them. Defaults to `<output_dir>/tensorboard`. Each run writes into its own `run-<id>` subfolder. |
| `recommendation` | object or null | `null` | A full recommendation object to train from. When omitted, the server builds one for `model_id`. See [Training config: overriding](training-config.md#overriding-the-plan). |
| `resume_from_checkpoint` | string or null | `null` | Passed to the trainer's `train(resume_from_checkpoint=...)`. Prefer [`/resume`](#post-apirunsrun_idresume), which checks the checkpoint belongs to the run. |
| `parallelism_strategy_override` | string or null | `null` | `data_parallel` or `fsdp`. FSDP is only accepted with `full_finetune`. |

A minimal request that lets the server plan for the given model:

```bash
curl -H "X-ModelForge-Token: $TOKEN" -H "Content-Type: application/json" \
     -d '{
           "use_case_slug": "instruction_following",
           "objective": "supervised",
           "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
           "dataset_path": "./data/train.jsonl",
           "output_dir": "./runs/qwen-sft-01"
         }' \
     $MF/api/runs
```

Training from a plan you fetched and edited:

```bash
curl -s -H "X-ModelForge-Token: $TOKEN" -H "Content-Type: application/json" \
     -d '{"use_case_slug":"instruction_following","objective":"supervised"}' \
     $MF/api/recommendations > plan.json

# Edit plan.json, e.g. lower strategy_config.max_length, then:
MODEL=$(jq -r '.recommended_models[0].model_id' plan.json)
jq -n --slurpfile plan plan.json --arg model "$MODEL" '{
  use_case_slug: "instruction_following",
  objective: "supervised",
  model_id: $model,
  dataset_path: "./data/train.jsonl",
  output_dir: "./runs/sft-edited",
  recommendation: $plan[0]
}' | curl -H "X-ModelForge-Token: $TOKEN" -H "Content-Type: application/json" -d @- $MF/api/runs
```

Edits are only kept when `model_id` equals `recommendation.recommended_models[0].model_id`. Otherwise the plan is rebuilt for the chosen model. The full rule is in [Training config](training-config.md#overriding-the-plan).

Returns the run record, usually with `status: "queued"` (the training thread may already have moved it to `running`). See [Response shapes: runs](responses.md#runs). This response does not include `tensorboard_url`; fetch the run again to get it.

| Status | When |
|---|---|
| `200` | Run created. Later failures (download errors, out of memory, bad trainer arguments) show up as `status: "failed"` and an `error` event, not as an HTTP error. |
| `400` | The recommendation could not be built or doesn't fit the request: invalid `recommendation` object, `Recommendation does not match the selected use case and objective.`, `Recommendation provider is not supported by this strategy.`, `Unknown fine-tuning approach.`, `Cannot safely configure this model without valid parameter metadata.`, FSDP with a non-full tier, a GPU index out of range, or any error from planning for `model_id`. |
| `409` | `A training run is already active. Cancel or wait for it before starting another.` or `Playground inference is using model resources. Wait for it to finish before starting training.` |
| `422` | Body validation, for example a blank `output_dir` (`value cannot be empty`) or an unknown `objective`. |

Only one run trains at a time per server process. An open Playground streaming session also blocks new runs until it is closed.

### GET /api/runs

| Query | Type | Default |
|---|---|---|
| `limit` | int | `100` |

```bash
curl -H "X-ModelForge-Token: $TOKEN" "$MF/api/runs?limit=20"
```

```json
{"runs": [ { "id": 7, "status": "running", "...": "..." } ]}
```

Runs are ordered by `updated_at`, most recent first. Runs with TensorBoard enabled include `tensorboard_url`.

### GET /api/runs/{run_id}

One run record, with `tensorboard_url` when TensorBoard is enabled. `404` with `Run not found.` if it doesn't exist.

### POST /api/runs/{run_id}/cancel

Asks an active run to stop. The trainer stops after the step in progress, so the run moves to `cancelled` a moment later, not immediately. No body.

Returns the run record as it stands (normally still `running`). `404` with `No active job found for run <id>.` if the run isn't active in this server process, including runs that already finished.

Steps since the last checkpoint are lost. Cancel shortly after a checkpoint if you plan to resume.

### GET /api/runs/{run_id}/events

Everything already emitted for the run, oldest first:

```json
{"events": [ {"run_id": 7, "type": "status", "message": "Training started.", "payload": {}, "ts": 1794712364.58} ]}
```

While the server is up, it keeps the last 5,000 events per run in memory for the 20 most recent runs. When a run finishes, its timeline is also written to the database, so finished runs keep their history across restarts. A run that was interrupted by a server stop has no archived events. Unknown run IDs return an empty list, not `404`.

### WS /api/runs/{run_id}/events

Live events for a run, from the moment you connect. Load the REST history first, then attach, and de-duplicate if you need a complete timeline.

```text
ws://127.0.0.1:8000/api/runs/7/events?token=<token>
```

Each message is one JSON event (`run_id`, `type`, `message`, `payload`, `ts`). When no event arrives for one second, the server sends a heartbeat:

```json
{"run_id": 7, "type": "heartbeat"}
```

Heartbeats carry no `message`, `payload` or `ts`. The socket never closes on its own when a run ends; close it after a `completed`, `cancelled` or `error` event. Several clients can watch the same run and each receives every event. Event types and payloads are listed in [Response shapes: events](responses.md#events).

A minimal Python client (needs the `websockets` package, which ModelForge already depends on):

```python
import asyncio, json, websockets

async def follow(run_id, token):
    url = f"ws://127.0.0.1:8000/api/runs/{run_id}/events?token={token}"
    async with websockets.connect(url) as ws:
        async for raw in ws:
            event = json.loads(raw)
            if event["type"] == "heartbeat":
                continue
            print(event["type"], event.get("message", ""))
            if event["type"] in {"completed", "cancelled", "error"}:
                break

asyncio.run(follow(7, "my-fixed-token"))
```

### GET /api/runs/{run_id}/checkpoints

```json
{"checkpoints": [ {"id": 3, "run_id": 7, "path": "runs/qwen-sft-01/checkpoint-1000", "step": 1000, "created_at": "..."} ]}
```

Ordered by step, highest first. Checkpoints are recorded each time the trainer saves (every `save_steps`, 500 by default). Unknown run IDs return an empty list.

### POST /api/runs/{run_id}/resume

Starts a **new** run that continues from one of this run's registered checkpoints, with the same plan, model, dataset, output directory, GPUs and TensorBoard setting.

| Field | Type | Notes |
|---|---|---|
| `checkpoint_path` | string | Must match a `path` returned by `/checkpoints` for this run. |

```bash
curl -H "X-ModelForge-Token: $TOKEN" -H "Content-Type: application/json" \
     -d '{"checkpoint_path":"runs/qwen-sft-01/checkpoint-1000"}' \
     $MF/api/runs/7/resume
```

Returns the new run record. The original run is left as it was.

| Status | When |
|---|---|
| `400` | `No run found with id <id>.` or `Checkpoint is not registered for run <id>: <path>` |
| `409` | Another run is active, or the Playground holds model resources. |

The API doesn't check the original run's status. The console only offers **Resume** on failed or cancelled runs.

## Trained models

A model is registered when a run completes and its output directory exists. Cancelled and failed runs are not registered.

### GET /api/models

```json
{"models": [ { "id": 2, "name": "Qwen2.5-0.5B-Instruct-7", "...": "..." } ]}
```

Newest first, up to 200. See [Response shapes: models](responses.md#models).

### GET /api/models/{model_id}

One model. `404` with `Model not found.`

### DELETE /api/models/{model_id}

| Query | Type | Default | Notes |
|---|---|---|---|
| `delete_files` | bool | `false` | Also delete the model's output folder from disk. |

```bash
curl -X DELETE -H "X-ModelForge-Token: $TOKEN" "$MF/api/models/2?delete_files=true"
```

```json
{"deleted": true, "files_removed": true}
```

Only the folder recorded in the registry is ever deleted, never a path from the request. It refuses (`400`, `Refusing to remove a filesystem or workspace root.`) when that folder is `/`, your home directory, or the server's working directory. `404` if the model doesn't exist. The run record and its checkpoint list are not touched.

### POST /api/models/{model_id}/push

Uploads the model's output folder to a Hugging Face model repository, creating it if needed. Uses the Hugging Face login cached on the server machine (`hf auth login`, or `HF_TOKEN`).

| Field | Type | Default |
|---|---|---|
| `repo_id` | string | required, for example `your-name/qwen-sft` |
| `private` | bool | `false` |

```json
{"pushed": true, "repo_id": "your-name/qwen-sft", "url": "https://huggingface.co/your-name/qwen-sft"}
```

| Status | When |
|---|---|
| `404` | `Model not found.` |
| `409` | `The files for this model are no longer at <path>.` |
| `502` | `Upload failed: <Hub error>`, for example no login or no write access. |

!!! warning "The whole folder is uploaded"
    That includes every `checkpoint-*` folder and the `tensorboard` logs if they are inside the output directory. Delete what you don't want to publish first.

## Playground

There are two independent inference paths. The streaming session API keeps one model loaded and refuses to open while training is active. The one-shot `infer` endpoint loads a model per request and can run on CPU and disk while training. See [Playground](../user-guide/playground.md) for how the console uses them.

### GET /api/playground/status

```json
{"available": false, "reason": "A training run is using the GPU. Loading a model beside it would run the card out of memory, so the playground stays closed until the run finishes.", "session_id": null, "model_path": null}
```

`available` is true when no run is training and no other inference holds the resource lock, or when a session is already open. `session_id` and `model_path` describe the open session, if any.

### POST /api/playground/sessions

Loads a saved model or adapter folder into the single session slot.

| Field | Type | Notes |
|---|---|---|
| `model_path` | string | A directory on the server: a run's output folder or a `checkpoint-*` folder. |

```json
{"session_id": "4f0c2a9e6d1b4b5e8a7c3f2d1e0b9a8c", "model_path": "runs/qwen-sft-01", "base_model": "Qwen/Qwen2.5-0.5B-Instruct", "note": null}
```

`note` is `"No GPU detected. Responses will be slow."` when CUDA isn't available. Opening the same path again returns the existing session. Opening a different path closes the old session first.

The model is loaded with `device_map="auto"`. If the folder contains `adapter_config.json` it is loaded as a PEFT adapter on top of its recorded base model. The task (and so the model class) is read from `modelforge_config.json` in the folder, defaulting to causal LM. Checkpoint folders don't contain that file, so a checkpoint of a summarization or QA model is treated as a causal LM here and usually fails to load. Use `/api/playground/infer` for those, which reads the task from the run record.

`409` with the reason when the session can't open: a run is training, the resource lock is held, `Nothing found at <path>.`, `The saved model has unsupported task '<task>'.`, or `Could not load this model: <error>`.

While a session is open it holds the resource lock, so `POST /api/runs` and `/api/playground/infer` return `409` until you close it.

### DELETE /api/playground/sessions/{session_id}

Unloads the model and frees GPU memory. Returns `{"closed": true}`, or `{"closed": false}` if that session isn't the open one.

### WS /api/playground/{session_id}/chat

```text
ws://127.0.0.1:8000/api/playground/<session_id>/chat?token=<token>
```

Client to server, one message per turn:

```json
{"messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}
```

| Field | Default | Notes |
|---|---|---|
| `messages` | `[]` | Full conversation so far, as `role`/`content` objects. Causal LMs use the tokenizer's chat template (or a plain `User:`/`Assistant:` fallback). Other tasks use only the last message. |
| `max_new_tokens` | `256` | |
| `temperature` | `0.7` | `0` switches to greedy decoding. |
| `top_p` | `0.9` | |

Send `{"type": "stop"}` at any time to end generation at the next token.

Server to client:

| Message | Meaning |
|---|---|
| `{"type": "start"}` | Generation started. |
| `{"type": "token", "text": "..."}` | Next chunk of text. |
| `{"type": "done"}` | Generation finished or was stopped. |
| `{"type": "error", "message": "..."}` | Generation failed. The socket stays open for the next turn. |

If the session has already closed, the server sends `{"type": "error", "message": "That playground session has closed."}` and closes the socket.

For extractive QA models, send the last message's `content` as JSON: `{"question": "...", "context": "..."}`. Token classification models reply with one `token: label` line per token.

### POST /api/playground/infer

One-shot inference: loads the model, answers, and unloads it. Nothing is kept between requests.

Request body (`InferenceRequest`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `run_id` | int | required | Must be > 0. |
| `checkpoint_path` | string or null | `null` | A checkpoint registered for this run. When omitted, the run must be `completed` and its output folder is used. |
| `context_window` | int | `4096` | One of `1024`, `4096`, `8016`, `32768`. |
| `max_new_tokens` | int | `256` | 1 to 1024. |
| `messages` | list | required | 1 to 256 messages with `role` (`user` or `assistant`) and `content` (1 to 100,000 characters). Must alternate, starting and ending with `user`. At most 500,000 characters in total. |

```bash
curl -H "X-ModelForge-Token: $TOKEN" -H "Content-Type: application/json" \
     -d '{"run_id":7,"messages":[{"role":"user","content":"Summarise LoRA in one sentence."}]}' \
     $MF/api/playground/infer
```

```json
{"reply": "LoRA trains two small matrices beside each frozen weight ...", "task": "causal_language_modeling", "input_tokens": 38, "context_limit": 4096, "device": "cuda:0", "training_active": false}
```

Weights are loaded in float32 and placed by Accelerate across free GPU memory, CPU RAM and disk, within memory budgets checked before loading. While a run is training, GPUs are excluded. Decoding is greedy. The effective `context_limit` is the smallest of `context_window`, the model's maximum positions and the tokenizer's limit; the conversation plus `max_new_tokens` must fit under it. Set `MODELFORGE_OFFLOAD_DIR` to choose where disk offload goes (default: the system temp directory).

| Status | `detail` |
|---|---|
| `400` | `This run's model task is not supported by the playground.`, `Cannot estimate model memory safely...`, `Cannot estimate this model's attention memory safely.`, `Cannot plan model placement safely: ...`, `Cannot read adapter configuration: ...` |
| `404` | `Run not found.`, `Checkpoint is not registered for this run.`, `The saved model directory is unavailable.` |
| `409` | `Choose a completed run or a registered checkpoint.`, `Model resources are busy. Try again when the current request finishes.`, `Not enough available RAM for safe inference.` (or `... while training is active.`), `Cannot safely use a GPU while training is active.`, `Not enough memory to execute the model, even with disk offload.`, `Not enough disk space for safe model offload.` |
| `422` | `This conversation needs N tokens, exceeding the M-token limit. Start a new chat or choose a larger supported context.`, `Send JSON with nonempty "question" and "context" strings.` (QA models), or body validation such as `Messages must alternate user and assistant, ending with a user message.` |
| `500` | `Inference failed: <error>` |
| `503` | `Cannot verify training resource use; inference is unavailable.`, `Cannot access the model offload directory: ...` |

Models that need `trust_remote_code` are not loaded by this endpoint.

## Scripting example

Plan, start, follow and fetch the result from Python with `requests`:

```python
import time
import requests

MF = "http://127.0.0.1:8000/api"
H = {"Authorization": "Bearer my-fixed-token"}

plan = requests.post(f"{MF}/recommendations", headers=H, json={
    "use_case_slug": "instruction_following", "objective": "supervised",
}).json()
model = plan["recommended_models"][0]["model_id"]

check = requests.post(f"{MF}/datasets/validate", headers=H, json={
    "path": "./data/train.jsonl", "use_case_slug": "instruction_following",
}).json()
assert check["valid"], check["error"]

run = requests.post(f"{MF}/runs", headers=H, json={
    "use_case_slug": "instruction_following", "objective": "supervised",
    "model_id": model, "dataset_path": "./data/train.jsonl",
    "output_dir": "./runs/scripted-01", "recommendation": plan,
})
run.raise_for_status()
run_id = run.json()["id"]

while True:
    status = requests.get(f"{MF}/runs/{run_id}", headers=H).json()
    if status["status"] in {"completed", "failed", "cancelled"}:
        break
    time.sleep(30)

print(status["status"], status["error"])
```

## See also

- [Response shapes](responses.md): every response body and event type.
- [Training config](training-config.md): what's inside `strategy_config` and how to override it.
- [CLI reference](cli.md): `modelforge run` flags.
- [Common issues](../troubleshooting/common-issues.md): 401s, 409s and failed runs.
