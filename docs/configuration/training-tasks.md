# Use cases and tasks

A **use case** is what you want the model to do, in plain terms. Choosing one fixes which kind of model ModelForge searches the Hub for, which model head it loads, which objectives you can train, and which trainer runs. This page maps each use case to its pipeline tag, task, strategies and trainers.

## Three layers

| Layer | What it is | Defined in | Example |
|---|---|---|---|
| Use case | The goal, plus the data format for each objective | `recommendation_engine/language_use_cases/` | `summarization` |
| Task | The model head (Transformers `AutoModel` class) | `finetuning_strategies/tasks/` | `seq2seq_modeling` → `AutoModelForSeq2SeqLM` |
| Strategy | The training method and trainer | `finetuning_strategies/strategies/` | `sft` |

You pick the use case and the objective. ModelForge derives the task and the strategy. See [How planning works](configuration-guide.md#step-1-use-case-and-objective).

## Available use cases

| Use case | Slug | Hub pipeline tag | Task | Objectives |
|---|---|---|---|---|
| Text Generation | `text_generation` | `text-generation` | `causal_language_modeling` | supervised, preference, reinforcement |
| Instruction Following | `instruction_following` | `text-generation` | `causal_language_modeling` | supervised, preference, reinforcement |
| Summarization | `summarization` | `summarization` | `seq2seq_modeling` | supervised |
| Extractive QA | `extractive_qa` | `question-answering` | `question_answering` | supervised |

List them on your machine:

```bash
modelforge list-tasks
```

Or call `GET /api/use-cases`, which also returns each use case's data formats and any objectives that are unavailable.

## From use case to trainer

| Use case · objective | Strategy | Model class | Trainer |
|---|---|---|---|
| Text generation or instruction following · supervised | `sft` | `AutoModelForCausalLM` | TRL `SFTTrainer` |
| Text generation or instruction following · preference | `dpo` | `AutoModelForCausalLM` | TRL `DPOTrainer` |
| Text generation or instruction following · reinforcement | `grpo` | `AutoModelForCausalLM` | TRL `GRPOTrainer` with the built-in exact-match reward |
| Summarization · supervised | `sft` | `AutoModelForSeq2SeqLM` | Transformers `Seq2SeqTrainer` |
| Extractive QA · supervised | `qa` | `AutoModelForQuestionAnswering` | Transformers `Trainer` with start/end span labels |

Summarization goes through the `sft` strategy, but because TRL's SFT trainer is built for causal models, the strategy routes the `seq2seq_modeling` task to Transformers' native encoder-decoder trainer. SFT-only options such as packing are not passed to it.

When a run uses LoRA or QLoRA, the adapter's PEFT task type follows the task:

| Task | PEFT `task_type` |
|---|---|
| `causal_language_modeling` | `CAUSAL_LM` |
| `seq2seq_modeling` | `SEQ_2_SEQ_LM` |
| `question_answering` | `QUESTION_ANS` |

## Objectives per strategy

| Objective | Default strategy | Supports tasks | Extra model copies | Batch multiplier |
|---|---|---|---|---|
| supervised | `sft` | `causal_language_modeling`, `seq2seq_modeling` | 0 | 1.0 |
| preference | `dpo` | `causal_language_modeling` | 1 | 2.0 |
| reinforcement | `grpo` | `causal_language_modeling` | 0 | 4.0 |
| supervised (Extractive QA) | `qa` | `question_answering` | 0 | 1.0 |

The last two columns feed the memory budget. See [Hardware and memory](hardware-profiles.md#per-strategy-adjustment).

An objective is only offered when its strategy is registered and supports the use case's task. Objectives that fail this check are listed under `unavailable_objectives` with a reason, and a use case with no executable objective is hidden from the console and from `GET /api/use-cases` (it appears under `unavailable_use_cases` in `GET /api/system`). With the built-in strategies every objective listed above is available.

## Text generation

Open-ended continuation of a prompt: story continuation, completion in a house style, structured text in a fixed format.

- **Models searched:** `text-generation`, the most downloaded with known parameter counts.
- **Columns:** `prompt` and `completion`, or the aliases `input_text` and `generated_text`.
- **Preference and reinforcement:** use `preferred_continuation` / `rejected_continuation` and `ideal_continuation` aliases.

## Instruction following

Chat assistants and anything that follows a natural-language instruction.

- **Models searched:** `text-generation`, restricted to chat or instruct models: those tagged `conversational`, or with `instruct`, `instruction`, `chat` or `it` as a word in the name.
- **Columns:** `prompt` and `completion`, or the aliases `instruction` and `response`.
- **Preference and reinforcement:** use `preferred_response` / `dispreferred_response` and `ideal_answer` aliases.

### Instruction following vs. text generation

Both train the same kind of model with the same trainers. They differ in two ways:

1. Instruction following searches only chat and instruct checkpoints. Text generation searches every `text-generation` model, including base models.
2. The column aliases describe instructions and responses rather than input text and continuations.

Pick whichever matches how your data is already labelled and which kind of starting model you want. For preference training (DPO) on chat behaviour, start from an instruct model: DPO aligns existing behaviour and does not teach a base model to chat.

## Summarization

Abstractive summaries of longer documents with an encoder-decoder model such as T5 or BART.

- **Models searched:** `summarization`. A model ID you type in may also be tagged `translation` or `text2text-generation`.
- **Columns:** `document` and `summary`, or `prompt` and `completion`. Both must be strings.
- Both input and target are truncated to the plan's `max_length` tokens.

## Extractive QA

The model doesn't write an answer. It predicts the start and end of the answer span inside a passage.

- **Models searched:** `question-answering`.
- **Columns:** `context`, `question`, `answers` in the SQuAD layout. See [Extractive QA answers](dataset-formats.md#extractive-qa-answers).
- Long passages are split into overlapping windows (`max_length` 384 tokens, or 256 below 12 GB of VRAM, with a 128-token stride).
- The tokenizer must have a CLS token.
- In the [Playground](../user-guide/playground.md) you send a question and a context and get back a span.

## Registered tasks without a use case

`token_classification` (`AutoModelForTokenClassification`) is registered as a task, but no use case or strategy uses it yet. You can't train it from the console, CLI or API. Adding one is described in [Adding a use case](../contributing/use-cases.md).

## Choosing a use case

| You want | Use case | Objective |
|---|---|---|
| A chat model that follows your format or tone | Instruction Following | supervised |
| To prefer some correct answers over others | Instruction Following or Text Generation | preference |
| Answers that can be checked exactly (numbers, labels) | Instruction Following or Text Generation | reinforcement |
| Continuations in a particular style | Text Generation | supervised |
| Short summaries of long documents | Summarization | supervised |
| Answers copied out of a passage | Extractive QA | supervised |

Start with supervised training. It is the cheapest and most forgiving objective. See [Strategies overview](../strategies/overview.md).

## What gets saved

A run started from the browser console, the notebook CLI or the API writes `modelforge_config.json` to its output directory when it completes. The file names the base model, task, use case, objective, strategy, PEFT approach, model class and pipeline task, and the playground uses it to reload the model. The terminal wizard (`modelforge cli`) does not write this file.

| Task | `model_class` | `pipeline_task` |
|---|---|---|
| `causal_language_modeling` | `AutoModelForCausalLM` | `text-generation` |
| `seq2seq_modeling` | `AutoModelForSeq2SeqLM` | `summarization` |
| `question_answering` | `AutoModelForQuestionAnswering` | `question-answering` |

## See also

- [Dataset formats](dataset-formats.md): columns and complete examples for every combination.
- [Strategies overview](../strategies/overview.md)
- [Adding a use case](../contributing/use-cases.md)
- [Custom strategies](../strategies/custom-strategies.md)
