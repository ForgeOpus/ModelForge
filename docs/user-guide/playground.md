# Playground

The Playground lets you chat with a trained model, or with any checkpoint of a run, from the browser console. This page covers choosing a model, the two inference modes, context windows, how memory and disk offload are planned, extractive QA input, and what happens to your conversations.

Open it from **Playground** in the sidebar (`/app/playground`).

## Choosing what to talk to

| Control | What it does |
|---|---|
| **Run** | Any run in the run history, shown as `#id · model · status`. |
| **Model output** | **Final model** for a completed run, or a checkpoint listed as **Step N**. For runs that are not completed the default entry is **Choose a checkpoint**. |
| **Context window** | One-shot mode only. 1,024, 4,096 (default), 8,016 or 32,768 tokens. |

A run's final model is only available once the run is `completed`. Checkpoints work for any run that has them, including runs that are still training, failed or were cancelled. Changing the run or the checkpoint starts a fresh conversation. The message box stays disabled until a completed model or a checkpoint is selected.

## Inference modes

A segmented control switches between **One-shot** (the default) and **Streaming**.

| | One-shot | Streaming |
|---|---|---|
| Model lifetime | Loaded for each message, then unloaded. | Loaded once by **Open streaming session**, kept until **Close session**. |
| Works while training | Yes, on CPU RAM and disk only. | No. Sessions are refused while a run is active. |
| Output | Whole reply at once. | Tokens streamed as they are generated, with **Stop generation**. |
| Decoding | Greedy (no sampling), up to 256 new tokens. | Sampling with temperature 0.7 and top-p 0.9, up to 256 new tokens. |
| Precision and placement | float32, placed by an explicit memory plan (below). | Saved dtype (`dtype="auto"`), `device_map="auto"`. |
| Context window control | Yes. | No. |
| Export and New chat | **Export Markdown**, **New chat**. | Not available. |

One-shot is the safer choice on a busy machine, because it checks memory before loading anything. Streaming is faster for a longer conversation, because the model is not reloaded for every message.

## One-shot mode

### Context windows

The limit that applies is the smallest of:

- the window you choose,
- the model's maximum positions (`max_position_embeddings` or `n_positions` in its config),
- the tokenizer's `model_max_length`.

For generating tasks (text generation, instruction following and summarization) each request reserves 256 tokens for the reply on top of the input. If the input plus that reservation does not fit, the request is refused with an error like:

```text
This conversation needs 4410 tokens, exceeding the 4096-token limit. Start a new chat or choose a larger supported context.
```

Earlier messages are never silently dropped. Start a new chat or pick a larger window. Token-classification and extractive QA models produce no new tokens, so nothing is reserved for them.

Below each reply the console shows the input token count, the limit that applied, the device placement, and `training active` if a run was training at the time:

```text
812 input tokens · 4096 token limit · CUDA:0
```

What counts as input depends on the task:

| Task | Input sent to the model |
|---|---|
| Causal language models | The whole conversation, formatted with the tokenizer's chat template (or a plain `User:` / `Assistant:` template when the tokenizer has none). |
| Seq2seq (summarization) | The last message only. |
| Token classification | The last message. The limit check counts the whole conversation. |
| Extractive QA | The last message, parsed as JSON (see [below](#extractive-qa-models)). |

### Device placement and offload

Weights are loaded in float32. ModelForge first builds an empty copy of the model to measure it, then uses Accelerate to plan a device map inside explicit memory budgets. The estimate includes the weights, a float32 attention matrix and KV cache for the chosen context window, 1 GB of runtime overhead, and four times the adapter file size.

| Budget | When no run is training | While a run is training |
|---|---|---|
| Each CUDA GPU | Free memory, minus the larger of 1 GB or 10% of total, minus the runtime estimate. Used only if at least 256 MB remains. | Not used. |
| CPU RAM | Available RAM, minus a reserve of the larger of 2 GB or a quarter of available RAM, minus the runtime estimate. | Same, but the reserve is the larger of 8 GB or half of available RAM. |
| Disk | Anything that does not fit above. | Anything that does not fit in RAM. |

The resulting placement is shown as `CUDA:0`, `CPU`, or a combination such as `GPU + CPU + DISK`. The request is refused, rather than started and crashed, when:

- the CPU budget is under 256 MB (*Not enough available RAM for safe inference*),
- the plan would put the whole model on disk,
- a GPU would be used while a run is training,
- disk offload is needed and the offload drive has less than twice the model and adapter size plus 1 GB free.

Offloaded weights go to a temporary directory that is deleted after the reply, including on errors. Set `MODELFORGE_OFFLOAD_DIR` to an existing directory before starting the server to put it on a larger or faster drive; otherwise the system temporary directory is used.

```bash
MODELFORGE_OFFLOAD_DIR=/mnt/fast-ssd/offload modelforge run
```

!!! warning "Disk offload is slow"
    A reply can take minutes when most of the model lives on disk. CPU inference also competes with a CPU-bound training run.

!!! note "Budgets are estimates"
    The checks run once, before loading. The operating system, the training job or another process can allocate memory afterwards. The coordination only applies inside one ModelForge server process.

Models that need `trust_remote_code` are not loaded. For a LoRA or QLoRA output, the base model named in `adapter_config.json` is loaded and the adapter applied on top. The tokenizer is taken from the output folder when it contains one, otherwise from the base model.

## Streaming mode

Press **Open streaming session** to load the selected model or checkpoint. It stays loaded until you press **Close session**, switch run or checkpoint, or leave the page. Only one session exists at a time; opening a different model closes the previous one. When no GPU is available, the console notes that responses will be slow.

Each message opens a WebSocket that streams tokens as they are generated. **Stop generation** ends the reply at the next decoding step.

Streaming sessions reject requests while a run is training: loading a second model beside a training job is the quickest way to run the GPU out of memory. The message is *A training run is using the GPU…* and the session stays closed until the run finishes.

!!! note "Streaming and checkpoints"
    Streaming mode reads the task type from `modelforge_config.json`, which is written to the final output folder only when a run completes. Checkpoint folders do not have it, so streaming loads a checkpoint as a causal language model. To test checkpoints of summarization or extractive QA runs, use one-shot mode, which takes the task from the run record.

For extractive QA in streaming mode, the context is truncated to fit the smallest of the tokenizer limit, the model's maximum positions and 4,096 tokens.

## Extractive QA models

Send a JSON message with a non-empty `question` and `context`:

```json
{"question": "Where is Paris?", "context": "Paris is in France."}
```

The reply is a span copied out of the context, up to 30 tokens long, or `(No answer span)`. Anything that is not this JSON is refused with *Send JSON with nonempty "question" and "context" strings.*

Token-classification models reply with one `token: label` line per token.

## Conversations

Conversations exist only in the browser tab. **New chat**, switching run or checkpoint, or leaving the page discards them. The server keeps no conversation history.

In one-shot mode, **Export Markdown** downloads the conversation as `modelforge-run-<id>-chat.md`. One-shot replies are greedy, so the same conversation gives the same reply.

Limits on a one-shot request: up to 256 messages, each up to 100,000 characters and 500,000 characters in total, alternating user and assistant and ending with a user message.

## Playground and training

Training and the Playground share one admission lock inside the server:

- While a one-shot reply is being generated, or while a streaming session is open, starting a training run is refused with *Playground inference is using model resources. Wait for it to finish before starting training.* Close the streaming session first.
- While a run is training, one-shot inference keeps off the GPUs and streaming sessions are refused.
- Two one-shot requests cannot run at once. The second is refused with *Model resources are busy.*

## API

The console uses these endpoints, described in the [REST API reference](../api-reference/rest-api.md):

| Endpoint | Mode |
|---|---|
| `POST /api/playground/infer` | One-shot. Body: `run_id`, optional `checkpoint_path`, `context_window`, `max_new_tokens` (1 to 1024, default 256), `messages`. |
| `GET /api/playground/status` | Whether a streaming session can be opened, and the current one. |
| `POST /api/playground/sessions` | Open a streaming session for a `model_path`. |
| `DELETE /api/playground/sessions/{session_id}` | Close it. |
| `WS /api/playground/{session_id}/chat` | Stream a reply; send `{"type": "stop"}` to stop. |

API clients can raise `max_new_tokens` up to 1,024 for one-shot requests; the console always uses the default of 256.

## See also

- [Checkpoints and resuming](checkpoints.md)
- [Browser console](browser-console.md)
- [Performance and memory](../troubleshooting/performance.md)
- [Responses and events](../api-reference/responses.md)
