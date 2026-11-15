# Checkpoints and Resuming

Every 500 steps a run saves a checkpoint with the weights (or adapter), optimizer state and scheduler position. If a run fails, is cancelled, or the server stops, you can continue from a checkpoint instead of starting over. This page covers where checkpoints go, how to resume from the console and the API, what a server restart does, and how to test a checkpoint before deciding to finish a run.

## When checkpoints are saved

The interval is the `save_steps` training argument, 500 for every strategy (SFT, DPO, GRPO and extractive QA). It is part of the plan's **Training config** and is read-only in the console. Each save is a normal Transformers Trainer checkpoint, counted in optimizer steps. The Trainer also saves one at the final step when a run reaches the end, so a completed run has a checkpoint even if it was shorter than 500 steps. A run that fails or is cancelled before step 500 has none.

No `save_total_limit` is set, so every checkpoint is kept. On long runs with large models, or full fine-tunes, this can use a lot of disk. Delete old `checkpoint-*` folders you no longer need. The free-disk warning in the plan counts only the model's weights, not checkpoints.

See [Training configuration](../api-reference/training-config.md) for the other keys.

## Where checkpoints go

Checkpoints are written to:

```text
<output dir>/checkpoint-<step>/
```

For runs started from the browser console, the REST API or a [notebook](notebooks.md), each save is also recorded in the run database and emitted as a `checkpoint` event (`payload` has `path` and `step`). The run's detail page lists them under **Checkpoints**, newest first.

Runs from the [terminal wizard](terminal-cli.md) also write `checkpoint-<step>` folders, but they are not recorded in the run database, so they cannot be resumed or opened from the console.

## Resuming from the console

1. Open the run under **Runs**.
2. In **Checkpoints**, press **Resume** beside the step to continue from.
3. ModelForge starts a new run with the same use case, objective, model, dataset, GPU selection, output directory, TensorBoard setting and stored plan, and continues training from that checkpoint. The console takes you to the new run.

**Resume** is shown only on runs that are neither active (`queued` or `running`) nor `completed`, which in practice means `failed` and `cancelled` runs.

The resumed run is a new run with its own ID. The original keeps its status and its checkpoint list. The new run writes into the same output directory, so its later checkpoints and final weights land beside the old ones, and its TensorBoard logs go to a new `run-<new id>` folder next to the original's.

!!! note "Only one run trains at a time"
    Resuming is refused if another run is active, with *A training run is already active.* It is also refused while the Playground is using model resources.

## Resuming through the API

List a run's checkpoints, then post one of the returned paths:

```bash
TOKEN=...   # the startup token printed by modelforge run
curl -s -H "X-ModelForge-Token: $TOKEN" \
  http://127.0.0.1:8000/api/runs/7/checkpoints

curl -s -X POST -H "X-ModelForge-Token: $TOKEN" -H "Content-Type: application/json" \
  -d '{"checkpoint_path": "./modelforge_output/checkpoint-1500"}' \
  http://127.0.0.1:8000/api/runs/7/resume
```

The path must exactly match one registered for that run; any other path is refused with `400`. A `409` means a run is already active or the Playground holds model resources. The response is the new run record; follow it with `GET /api/runs/{id}` or the `/api/runs/{id}/events` WebSocket.

`POST /api/runs` also accepts a `resume_from_checkpoint` field, but the resume endpoint is the supported route. See the [REST API reference](../api-reference/rest-api.md).

## If the server stops

Runs execute inside the server process. If the server exits or crashes, active training stops with it. On the next start, every run still marked `queued` or `running` is marked `failed` with:

```text
ModelForge stopped before this run reached a terminal state.
```

Its checkpoints are untouched and still listed, so resume it as above. Steps after the last checkpoint are lost.

!!! warning "Notebook sessions mark runs too"
    The same startup check runs whenever a notebook session creates its job manager (`modelforge-nb`, option 1 or 2). Starting one while the browser server is training marks that server's active run as `failed` in the database, although training carries on and the status is overwritten when it finishes. Avoid using both at the same time.

## Testing a checkpoint in the Playground

Any listed checkpoint can be loaded in the [Playground](playground.md), including one from a run that is still training. Choose the run, then pick **Step N** under **Model output**. This is a quick way to decide whether a run is worth finishing.

- Use **One-shot** mode while training is active. It keeps the model off the GPUs, using CPU RAM and disk, so the training run keeps its VRAM.
- **Streaming** mode is refused while a run is training.

## Cancelling

**Cancel run** on the run's detail page (or `POST /api/runs/{id}/cancel`) asks the trainer to stop at the end of the current step. The run ends as `cancelled` and is not registered as an artifact. No extra checkpoint is written on cancel, so steps since the last save are lost. If you plan to resume, cancel shortly after a checkpoint event appears in the training log.

## See also

- [Browser console](browser-console.md)
- [Playground](playground.md)
- [Responses and events](../api-reference/responses.md)
- [Performance and memory](../troubleshooting/performance.md)
