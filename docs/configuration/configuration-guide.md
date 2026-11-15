# How planning works

ModelForge does not ask you to write a training config. It builds a **plan** (a recommendation) from three inputs: the use case, the training objective, and the hardware it detects. This page walks through every step the planner takes, what it can and cannot be told to do, and how to read the result.

The planner lives in `utilities/recommendation_engine.py` (`RecommendationEngine.recommend`). The browser console, the terminal wizard, the notebook CLI and the HTTP API all call it.

## The pipeline at a glance

```text
use case + objective ──► task + strategy ──► data format
hardware ──► raw VRAM ──► adjusted VRAM ──► per-tier parameter ceilings
ceilings ──► Hub search ──► shortlist ──► chosen model ──► tier
chosen model + tier + hardware ──► strategy_config, peft_config
GPU count + tier + model size ──► parallelism (data parallel or FSDP)
everything ──► rationale + warnings
```

| Step | Input | Output | Where to read it |
|---|---|---|---|
| 1. Resolve use case | `use_case_slug`, `objective` | `task_slug`, `strategy_slug`, `data_format` | `rationale.use_case_resolution` |
| 2. Detect hardware | The machine | GPUs, RAM, disk | `GET /api/hardware` |
| 3. Raw and adjusted VRAM | GPUs, strategy | Usable memory budget | `vram_estimate`, `rationale.vram_summary` |
| 4. Tier ceilings | Adjusted VRAM | Largest model per tier | `vram_estimate.max_params_by_tier` |
| 5. Model search | Hub pipeline tag, ceiling | Shortlist of up to `n` models | `recommended_models`, `rationale.model_filter` |
| 6. Pick model and tier | Shortlist | `peft_approach` | `rationale.peft_decision` |
| 7. Build configs | Tier, hardware, model | `strategy_config`, `peft_config` | Plan panel, "Training config" |
| 8. Parallelism | GPU count, tier, model size | `parallelism_strategy` | `rationale.parallelism_decision` |
| 9. Warnings | Everything above | `rationale.warnings` | Plan panel notices |

## Step 1: use case and objective

A **use case** is the goal in plain terms (for example `summarization`). It fixes the model head, the Hub pipeline tag searched, and which objectives are allowed. The **objective** picks the training method:

| Objective | Strategy |
|---|---|
| `supervised` | `sft` |
| `preference` | `dpo` |
| `reinforcement` | `grpo` |

A use case may override that mapping. Extractive QA sets `strategy_slug = "qa"`, so its supervised objective runs the native QA trainer instead of SFT.

| Use case | Task | Hub pipeline tag | Objectives | Strategy |
|---|---|---|---|---|
| `text_generation` | `causal_language_modeling` | `text-generation` | supervised, preference, reinforcement | `sft`, `dpo`, `grpo` |
| `instruction_following` | `causal_language_modeling` | `text-generation` | supervised, preference, reinforcement | `sft`, `dpo`, `grpo` |
| `summarization` | `seq2seq_modeling` | `summarization` | supervised | `sft` |
| `extractive_qa` | `question_answering` | `question-answering` | supervised | `qa` |

The planner rejects the request (HTTP 422 from `POST /api/recommendations`) when:

- the use case slug is not registered;
- the objective is not in the use case's `compatible_objectives`;
- the resolved strategy is not registered, or does not list the use case's task in `supported_tasks`.

The resolved pair also selects the **data format** your dataset must follow. See [Dataset formats](dataset-formats.md) and [Use cases and tasks](training-tasks.md).

## Step 2: hardware detection

`HardwareDetector.detect_hardware()` reports physical CPU cores, total RAM, total and free disk space, and a list of GPUs:

- **NVIDIA (CUDA):** every GPU NVML reports, with its name and total memory in GB.
- **Apple Silicon (MPS):** a single entry named `Apple Silicon GPU` whose memory is `"N/A"`.
- **Neither:** an empty list, which means CPU-only.

Details, including how precision is chosen, are in [Hardware and memory](hardware-profiles.md).

By default every detected GPU is used. The HTTP API accepts `gpu_indices` (zero-based) on `POST /api/recommendations` and `POST /api/runs` to plan and train on a subset. The terminal wizard asks which GPUs to use when it finds more than one. The browser console always plans for all GPUs.

## Step 3: raw and adjusted VRAM

**Raw VRAM** is the sum of the selected GPUs' memory. MPS counts as a flat 8 GB. CPU-only is 0.

**Adjusted VRAM** accounts for strategies that hold more than one model or more than one sequence in memory:

```text
adjusted = (raw − (batch_multiplier − 1) × 2.0 GB) / (1 + extra_model_copies)
```

| Strategy | `extra_model_copies` | `batch_multiplier` | Adjusted VRAM on 24 GB |
|---|---|---|---|
| `sft` | 0 | 1.0 | 24 GB |
| `dpo` | 1 | 2.0 | 11 GB |
| `grpo` | 0 | 4.0 | 18 GB |
| `qa` | 0 | 1.0 | 24 GB |

If the adjusted figure is negative on a GPU machine, the plan fails with `Strategy '<slug>' requires more VRAM than is available on this hardware for any viable model size.` GRPO reserves 6 GB of extra activations, so it fails on GPUs under 6 GB. DPO fails under 2 GB.

## Step 4: tier ceilings

A **tier** is how the base weights are held during training. Each tier has a fixed cost per parameter, plus a 2.0 GB activation budget and, for adapter tiers, 0.1 GB for the adapters:

| Tier (`peft_approach`) | Bytes per parameter | Ceiling (billions of parameters) |
|---|---|---|
| `full_finetune` | 18 | (adjusted − 2.0) / 18 |
| `lora_bf16` | 2 | (adjusted − 2.1) / 2 |
| `qlora_8bit` | 1 | (adjusted − 2.1) / 1 |
| `qlora_4bit` | 0.5 | (adjusted − 2.1) / 0.5 |

On more than one CUDA GPU the full fine-tuning ceiling uses the **summed** budget, because FSDP can shard it. The three adapter tiers are recomputed from the **smallest** selected GPU, because each data-parallel process keeps a whole copy of the model.

Special cases:

- **CPU-only:** only `qlora_4bit` has a ceiling, `min(500M, RAM_GB × 10⁹ / 3)` parameters. Quantization is switched off in the generated config, so it trains a LoRA adapter on full-precision weights.
- **MPS:** `qlora_8bit` and `qlora_4bit` are set to 0, because bitsandbytes does not support MPS.

The worked numbers for common GPUs are in [Hardware and memory](hardware-profiles.md#worked-examples).

## Step 5: Hub model search

The planner searches the Hugging Face Hub with the largest ceiling across all tiers as the size limit (usually the `qlora_4bit` ceiling):

1. List models with the use case's `pipeline_tag`, sorted by downloads, fetching up to `n × 20` (100 by default).
2. Skip test fixtures: owners containing `internal-testing`, and names containing `tiny`, `random` or `dummy` as a separate word.
3. For `instruction_following` only, skip models that have neither the `conversational` tag nor `instruct`, `instruction`, `chat` or `it` as a word in the name.
4. Skip models without a parameter count in their safetensors metadata. The plan warns with the number skipped.
5. Keep models at or under the size limit, stopping after `n` (default 5).

If nothing is found, the search runs once more with a relaxed bound.

!!! note "Hub access"
    The search uses `huggingface_hub.HfApi` with whatever token is cached on the machine. It needs network access. See [Hugging Face provider](../providers/huggingface.md#authentication-and-gated-models).

## Step 6: picking the model and tier

The shortlist from step 5 is the most popular models that fit at **some** tier. The planner then:

1. Sorts the shortlist by parameter count, largest first, with downloads as the tie-breaker.
2. Takes the largest model.
3. Gives it the least quantized tier that fits, in the order `full_finetune`, `lora_bf16`, `qlora_8bit`, `qlora_4bit`.

So the plan prefers **model capacity first**, then the highest-fidelity method for that model. A tiny model that could be fully fine-tuned does not hide a larger model that fits as an adapter.

!!! example "24 GB GPU, SFT"
    Ceilings: full 1.22B, LoRA bf16 10.95B, QLoRA 8-bit 21.9B, QLoRA 4-bit 43.8B.

    - Largest shortlisted model is 7B: `lora_bf16` (7 ≤ 10.95).
    - Largest is 14B: `qlora_8bit` (14 ≤ 21.9).
    - Largest is 32B: `qlora_4bit` (32 ≤ 43.8), with a warning that 4-bit was needed.
    - Largest is 1B: `full_finetune` (1 ≤ 1.22).

`recommended_models` lists every shortlisted model that fits the chosen tier, in the same size-first order. The training config is built for the first one. Picking a different model in the console, the notebook CLI or the API **re-plans** for that model, so a smaller model can come back as `full_finetune`.

## Step 7: training and adapter configs

The strategy's `build_recommended_config(peft_tier, hardware, chosen_model)` returns two typed objects:

- `strategy_config`: trainer settings. Batch size is 1 for SFT, DPO and GRPO, with gradient accumulation, `max_length` and activation offloading scaled to total VRAM, precision from the GPU's compute capability, and quantization in `model_init_kwargs` for QLoRA tiers.
- `peft_config`: LoRA rank, alpha (always 2 × rank) and `target_modules: "all-linear"`. `null` for full fine-tuning.

The VRAM-scaled values are listed in [Hardware and memory](hardware-profiles.md#vram-scaled-defaults). Per-strategy defaults are on [SFT](../strategies/sft.md), [DPO](../strategies/dpo.md), [GRPO](../strategies/grpo.md) and [LoRA and QLoRA](../strategies/qlora.md). Every field is listed in the [training config reference](../api-reference/training-config.md).

If the Hub returned no models, no config is built: `strategy_config` and `peft_config` are `null` and the plan cannot be trained.

The provider is always `huggingface`. See [Provider overview](../providers/overview.md).

## Step 8: parallelism


`parallelism_strategy` is `data_parallel` unless all of these hold, in which case it is `fsdp`:

- two or more CUDA GPUs are selected;
- the tier is `full_finetune`;
- a full replica (parameters × 18 bytes + 2 GB) does not fit on the smallest GPU.

FSDP cannot shard LoRA adapters, so it is never chosen for adapter tiers, and an override that combines FSDP with an adapter tier is rejected. See [Multi-GPU training](../strategies/multi-gpu.md).

## Step 9: plan warnings

Warnings appear under the plan in the console and in `rationale.warnings`. Each one comes from a specific check:

| Warning (abridged) | When | What to do |
|---|---|---|
| `N models were skipped due to missing parameter metadata on the Hub.` | Some popular models don't publish a parameter count. | Nothing. Those models can't be planned, even by ID. |
| `DPO aligns existing behavior...` | Every preference plan. | Start DPO from an instruct or chat checkpoint, or run SFT first. |
| `CPU-only training detected...` | No GPU found. | Expect very slow training. Quantization is disabled. |
| `Apple MPS detected... Capped at LoRA bf16.` | Apple Silicon. | Nothing. 8-bit and 4-bit are unavailable on MPS. |
| `Multi-GPU detected. FSDP auto-configured across N GPUs...` | FSDP chosen. | Nothing. |
| `Multi-GPU detected (N GPUs). Data parallel selected...` | More than one GPU, data parallel. | Set `parallelism_strategy_override` to `fsdp` for full fine-tuning if you want sharding. |
| `QLoRA 4-bit was selected because the chosen model exceeds the unquantized and 8-bit training budgets.` | Largest shortlisted model only fits at 4-bit. | Pick a smaller model if you want less quantization. |
| `No single-GPU tier is viable for this hardware...` | Every ceiling is 0. | Free GPU memory, add GPUs, or use a lighter objective. |
| `No models found on the Hub matching task...` | Empty shortlist. | No config was generated. Enter a model ID, or add VRAM. |
| `Initial Hub query returned 0 results. Retried...` | The relaxed retry ran. | See the previous row. |
| `Available disk space (X GB) may be insufficient...` | Free disk at `/` is below the largest model's weight size at the chosen tier. | Free disk space. Checkpoints need more on top. |

## Using your own model ID

You can train any Hub model, not only the shortlist. In the console, type it into **Or enter a different model ID**. In the API, pass `model_id` to `POST /api/recommendations` to plan it, or to `POST /api/runs`.

The model is checked by `WorkflowService.validate_model_id` (also exposed as `POST /api/models/validate` with `{model_id, task_slug}`):

1. It must be visible on the Hub with your login. A 401, 403 or 404 returns `No model with that ID is visible on the Hub...`. Log in with `hf auth login` for private or gated models.
2. If it has a pipeline tag, the tag must match the task:

    | Task | Accepted pipeline tags |
    |---|---|
    | `causal_language_modeling` | `text-generation` |
    | `seq2seq_modeling` | `summarization`, `translation`, `text2text-generation` |
    | `question_answering` | `question-answering` |

    A model with no pipeline tag is accepted.

3. It must publish a parameter count in its safetensors metadata.
4. The parameter count must fit under the largest tier ceiling, or the plan fails with `Selected model exceeds the available training memory budget.`

When the run starts, a model without parameter metadata is refused with `Cannot safely configure this model without valid parameter metadata.`

## What you can configure

### In the browser console

The console is deliberately small. You choose the use case, objective, dataset (local or Hub), base model (shortlist or any ID), output directory and whether to write TensorBoard logs. The plan, including the training config, is **read-only**. GPU selection and the parallelism override are not exposed.

### Through the HTTP API

| Field | Endpoint | Effect |
|---|---|---|
| `model_id` | `POST /api/recommendations` | Plan a specific model instead of searching. |
| `n` | `POST /api/recommendations` | Shortlist size (default 5, must be positive). |
| `gpu_indices` | `POST /api/recommendations`, `POST /api/runs` | Plan and train on a subset of GPUs. Train on the same subset you planned for. |
| `recommendation` | `POST /api/runs` | Train from a plan you supply, for example with an edited `strategy_config`. |
| `parallelism_strategy_override` | `POST /api/runs` | Force `data_parallel` or `fsdp`. FSDP requires `full_finetune`. |

An edited `recommendation` is checked before the run is recorded:

- its `task_slug` and `strategy_slug` must match the use case and objective;
- its `provider_slug` must be supported by the strategy;
- its `peft_approach` must be one of the four tiers;
- FSDP must not be combined with an adapter tier.

!!! warning "Edits are kept only for the top model"
    The supplied plan is used as sent only when `model_id` equals `recommendation.recommended_models[0].model_id`. If you train a different model, the planner rebuilds the plan for that model and your edits are discarded (a `parallelism_strategy_override` is still applied).

Values in `strategy_config` are passed to the trainer's config builder, so for SFT, DPO and GRPO any field of the corresponding TRL config (`SFTConfig`, `DPOConfig`, `GRPOConfig`) is accepted. The QA strategy reads only the keys it knows. `null` values are dropped.

Example: plan, lower `max_length`, then start.

```bash
TOKEN=...   # the startup token printed by `modelforge run`
curl -s -X POST http://127.0.0.1:8000/api/recommendations \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"use_case_slug": "instruction_following", "objective": "supervised"}' > plan.json

python - <<'EOF'
import json
plan = json.load(open("plan.json"))
plan["strategy_config"]["max_length"] = 512
body = {
    "use_case_slug": "instruction_following",
    "objective": "supervised",
    "model_id": plan["recommended_models"][0]["model_id"],
    "dataset_path": "./data/train.jsonl",
    "output_dir": "./modelforge_output/my-run",
    "recommendation": plan,
}
json.dump(body, open("run.json", "w"))
EOF

curl -s -X POST http://127.0.0.1:8000/api/runs \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d @run.json
```

### What can't be configured

- There is no switch to force a tier. The planner always takes the least quantized tier that fits the chosen model. You can send an edited plan with a different `peft_approach`, but nothing then checks that it fits, and `peft_approach`, `peft_config` and `strategy_config.model_init_kwargs` must agree.
- The memory constants (2.0 GB activations, 0.1 GB adapters, bytes per parameter) are fixed in code.
- The provider is always Hugging Face.
- GRPO's reward is the built-in exact-match reward. Custom reward functions are not supported.
- System RAM is never counted as extra VRAM, and CPU offloading for training is not configured.

## Command-line equivalents

```bash
modelforge recommend --use-case instruction_following --objective supervised --n 5
modelforge list-tasks
```

`modelforge recommend` prints the task, strategy, tier, shortlist and warnings for all detected GPUs. See [CLI reference](../api-reference/cli.md).

## See also

- [Hardware and memory](hardware-profiles.md): the memory model with worked examples.
- [Dataset formats](dataset-formats.md): what each plan expects from your data.
- [Recommendation fields](../api-reference/responses.md): every field of the plan.
- [REST API](../api-reference/rest-api.md): request bodies for the endpoints above.
- [Performance and out-of-memory](../troubleshooting/performance.md).
