# Dataset formats

ModelForge reads local files, local folders and Hugging Face Hub datasets. Before validating or training, it renames documented column aliases to the names each trainer expects, then runs the trainer's own format check. This page lists every accepted source, the columns each use case and objective needs, and a complete example file for each combination.

## Sources and file types

| Source | How to give it | Notes |
|---|---|---|
| Single file | A path ending in `.parquet`, `.arrow`, `.jsonl`, `.json`, `.csv` or `.txt` | The extension picks the loader. |
| Folder | A path to a directory | Loads every file of one format together, see below. |
| Hub dataset | An ID such as `org/dataset`, with **Hub** selected (console), option 2 (terminal), or `"remote": true` / `"dataset_remote": true` (API) | Downloaded with `datasets.load_dataset`. |

Local paths are resolved on the machine running ModelForge, relative to the directory you started it from.

### File types

| Extension | Loader | Notes |
|---|---|---|
| `.jsonl` | `json` | One JSON object per line. Recommended. |
| `.json` | `json` | A top-level array of row objects, or an object holding one list of rows, such as `{"data": [...]}`. The list field is found automatically. |
| `.csv` | `csv` | Flat string columns only. Can't hold the nested `answers` field for Extractive QA or message lists. |
| `.parquet` | `parquet` | Supports nested fields. |
| `.arrow` | `arrow` | Supports nested fields. |
| `.txt` | `text` | One `text` column, one row per line. No current use case accepts a `text`-only dataset, so `.txt` files fail validation. |

Keep each column a single type across all rows. A `prompt` that is a string in some rows and a list of messages in others can't be loaded into one Arrow column.

### Folders

A folder is scanned (not recursively) for supported files. The first format present in this priority order wins, and every file of that format is loaded as one dataset, in name order:

1. `.parquet`
2. `.arrow`
3. `.jsonl`
4. `.json`
5. `.csv`
6. `.txt`

Files of other formats in the same folder are ignored. A folder with none of these raises `No supported dataset files found in directory: <path>`. Shards of `.json` in a folder are read as flat JSON, without the nested-field detection used for single files.

### Hub datasets

The dataset is loaded with its default configuration and all splits are downloaded. ModelForge uses the **first split** in the dataset (for almost every dataset that is `train`) for validation and for training. Other splits are ignored, and no evaluation split is used.

You can't choose a split or a configuration name from the console, the terminal wizard or the API. Datasets that require a configuration name fail to load. Download and save the split you want as a local file instead:

```python
from datasets import load_dataset

load_dataset("org/dataset", "config_name", split="validation").to_json("data.jsonl")
```

Gated or private datasets need a Hugging Face login on the machine running ModelForge. See [Hugging Face provider](../providers/huggingface.md#authentication-and-gated-models).

## Required columns

Each (use case, objective) pair declares the columns it needs and the aliases it accepts. Canonical names are what the trainer reads. Aliases are renamed to them.

| Use case · objective | Trainer columns | Accepted aliases (alias → column) | Optional |
|---|---|---|---|
| Text generation · supervised | `prompt`, `completion` | `input_text` → `prompt`, `generated_text` → `completion` | |
| Text generation · preference | `chosen`, `rejected` | `preferred_continuation` → `chosen`, `rejected_continuation` → `rejected` | `prompt` (alias `input_text`) |
| Text generation · reinforcement | `prompt`, `answer` | `input_text` → `prompt`, `ideal_continuation` → `answer` | |
| Instruction following · supervised | `prompt`, `completion` | `instruction` → `prompt`, `response` → `completion` | |
| Instruction following · preference | `chosen`, `rejected` | `preferred_response` → `chosen`, `dispreferred_response` → `rejected` | `prompt` (alias `instruction`) |
| Instruction following · reinforcement | `prompt`, `answer` | `instruction` → `prompt`, `ideal_answer` → `answer` | |
| Summarization · supervised | `prompt`, `completion` | `document` → `prompt`, `summary` → `completion` | |
| Extractive QA · supervised | `context`, `question`, `answers` | none | |

For every row, either the alias or the canonical name works. A summarization file with `document` and `summary` trains exactly like one with `prompt` and `completion`.

Extra columns are allowed and are left alone.

### The ambiguity rule

A dataset that contains **both** an alias and the column it maps to is rejected, because ModelForge would have to guess which one to train on:

```text
Dataset contains both an alias and its canonical training column: 'instruction' -> 'prompt'. Remove or rename one explicitly.
```

Drop or rename one of the two columns.

### Where normalization happens

Aliases are renamed in one shared place (`DatasetFactory.normalize_columns`), both when you press **Validate** and when training starts, in the browser console, terminal wizard, notebook CLI and API. Validation and training therefore see the same columns.

## Chat formatting

For causal language models (text generation and instruction following), ModelForge converts rows to role/content messages before the trainer applies the tokenizer's chat template:

| Objective | Conversion |
|---|---|
| Supervised | A string `prompt` becomes `[{"role": "user", ...}]`, a string `completion` becomes `[{"role": "assistant", ...}]`. Values that are already message lists are kept. |
| Preference | Prompt, chosen and rejected are converted to messages, see [Preference pairs](#preference-pairs). |
| Reinforcement | A string `prompt` becomes a single user message. |

The model's own chat template is used when it has one. A tokenizer without a template gets a plain fallback that renders `User: ...` and `Assistant: ...` turns and ends assistant turns with the existing EOS token, so no new tokens are added. A tokenizer with neither a template nor an EOS token fails with `Chat training requires a tokenizer with an EOS token.` The final output folder includes the tokenizer and its template, including for adapter-only runs.

A template only gives the data a consistent shape. It does not teach a base model to chat; for that, run supervised training first or start from an instruct model.

Summarization and Extractive QA don't use chat formatting.

### Message lists

You can supply `prompt` and `completion` as message lists instead of strings, for example to include a system message:

```json
{"prompt": [{"role": "system", "content": "Answer in one sentence."}, {"role": "user", "content": "What is LoRA?"}], "completion": [{"role": "assistant", "content": "LoRA trains small low-rank matrices beside frozen weights."}]}
```

!!! note "`messages` and `text` columns"
    TRL's SFT trainer also accepts a single `messages` column or a single `text` column, but ModelForge's alias step requires `prompt` and `completion` (or their aliases) first. A dataset with only `messages` or only `text` fails with `Dataset is missing required column(s): ...`. Split conversations into `prompt` and `completion` message lists instead.

## Examples by use case and objective

Every example below is a complete, valid JSONL file. Save it as `data.jsonl` and point ModelForge at it.

### Text generation

=== "Supervised"

    ```json
    {"prompt": "Once upon a time, in a village by the sea,", "completion": " there lived an old fisherman who had never caught a fish."}
    {"prompt": "The recipe begins with", "completion": " two cups of flour, a pinch of salt and a little warm water."}
    {"prompt": "Dear hiring manager,", "completion": " I am writing to apply for the junior developer position."}
    ```

    Alias form: `{"input_text": "...", "generated_text": "..."}`.

=== "Preference"

    ```json
    {"prompt": "Write a one-line slogan for a bakery.", "chosen": "Fresh from our oven to your table, every morning.", "rejected": "Bread."}
    {"prompt": "Continue the story: The door creaked open and", "chosen": " a thin line of candlelight spilled across the floor.", "rejected": " stuff happened."}
    {"prompt": "Describe the sea in one sentence.", "chosen": "The sea stretched grey and restless under a low sky.", "rejected": "It is wet."}
    ```

    Alias form: `{"input_text": "...", "preferred_continuation": "...", "rejected_continuation": "..."}`.

=== "Reinforcement"

    ```json
    {"prompt": "Complete with one word. The opposite of hot is", "answer": "cold"}
    {"prompt": "Complete with one number. 12 + 30 =", "answer": "42"}
    {"prompt": "Complete with one word. The capital of Japan is", "answer": "Tokyo"}
    ```

    Alias form: `{"input_text": "...", "ideal_continuation": "..."}`.

### Instruction following

=== "Supervised"

    ```json
    {"prompt": "Summarize this ticket: The user cannot log in after the latest update.", "completion": "Login fails after the update."}
    {"prompt": "Translate to French: Good morning.", "completion": "Bonjour."}
    {"prompt": "List three primary colors.", "completion": "Red, blue and yellow."}
    ```

    Alias form: `{"instruction": "...", "response": "..."}`.

=== "Preference"

    ```json
    {"prompt": "Explain LoRA in one sentence.", "chosen": "LoRA freezes the model and trains two small low-rank matrices beside selected layers.", "rejected": "It is a thing."}
    {"prompt": "How do I reverse a list in Python?", "chosen": "Use my_list.reverse() to reverse in place, or my_list[::-1] for a reversed copy.", "rejected": "You can't."}
    {"prompt": "Give me a polite way to decline a meeting.", "chosen": "Thank you for the invitation. Unfortunately I can't attend, but I'd be glad to review the notes.", "rejected": "No."}
    ```

    Alias form: `{"instruction": "...", "preferred_response": "...", "dispreferred_response": "..."}`.

=== "Reinforcement"

    ```json
    {"prompt": "What is 17 × 3? Reply with the number only.", "answer": "51"}
    {"prompt": "Is the sentiment of 'I loved it' positive or negative? Reply with one word.", "answer": "positive"}
    {"prompt": "What is the chemical symbol for gold? Reply with the symbol only.", "answer": "Au"}
    ```

    Alias form: `{"instruction": "...", "ideal_answer": "..."}`.

### Summarization

=== "Supervised"

    ```json
    {"document": "The city council met on Tuesday to discuss the new park. After two hours of debate, members voted 7 to 2 to fund construction, which will begin in the spring.", "summary": "The council voted 7-2 to fund a new park, with construction starting in spring."}
    {"document": "Researchers tested a new battery design that charges in ten minutes. In lab trials it kept 90 percent of its capacity after 1,000 cycles.", "summary": "A new battery charges in ten minutes and keeps 90% capacity after 1,000 cycles."}
    {"document": "The library will close for renovations from June to August. During that time, books can be returned at the community centre.", "summary": "The library closes June to August; returns go to the community centre."}
    ```

    Canonical form: `{"prompt": "...", "completion": "..."}`. Both fields must be strings. They are truncated to the plan's `max_length` tokens.

### Extractive QA

=== "Supervised"

    ```json
    {"context": "The library opened in 1999 on Main Street.", "question": "When did the library open?", "answers": {"text": ["1999"], "answer_start": [22]}}
    {"context": "Marie Curie was born in Warsaw in 1867.", "question": "Where was Marie Curie born?", "answers": {"text": ["Warsaw"], "answer_start": [24]}}
    {"context": "The meeting has been moved to Thursday afternoon.", "question": "Who is chairing the meeting?", "answers": {"text": [], "answer_start": []}}
    ```

    There are no aliases for Extractive QA.

## Preference pairs

DPO rows can take several shapes. All of them are converted to a shared prompt plus assistant continuations before training.

| Shape | Example | Result |
|---|---|---|
| Explicit prompt, string answers | `{"prompt": "Hi", "chosen": "Hello!", "rejected": "What."}` | Prompt becomes a user message, answers become assistant messages. |
| Implicit prompt, string answers | `{"chosen": "The capital of France is Paris.", "rejected": "The capital of France is Lyon."}` | The longest shared prefix (`The capital of France is`) becomes the prompt. |
| Full conversations | `{"chosen": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}], "rejected": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "What."}]}` | The shared turns become the prompt. They must end in a `user` or `tool` turn. |
| Prompt plus assistant-only lists | `{"prompt": [{"role": "user", "content": "Hi"}], "chosen": [{"role": "assistant", "content": "Hello!"}], "rejected": [{"role": "assistant", "content": "What."}]}` | Used as given. |

Rules, checked when training starts:

- `chosen` and `rejected` must both be strings or both be message lists.
- String pairs without a `prompt` need a shared prefix, otherwise: `Text preference pairs need a shared prompt. Add a prompt column.`
- Assistant-only message lists need a `prompt`.
- An explicit `prompt` alongside full conversations must match the shared turns, otherwise: `Explicit prompt does not match the preference conversations.`
- Each continuation must start with an assistant turn.

An explicit `prompt` column is the most predictable choice.

## Reinforcement answers

GRPO's built-in reward compares each generated completion with the row's `answer`. It scores 1.0 when they match exactly after trimming surrounding whitespace, and 0.0 otherwise. Case and punctuation count, so `"Tokyo."` does not match `"Tokyo"`.

- `answer` must be a string. Non-string answers always score 0.
- Write prompts that ask for the exact form of the answer ("Reply with the number only").
- Use tasks with one checkable answer: arithmetic, labels, short extractions. The reward doesn't measure helpfulness.
- Custom reward functions are not supported.

See [GRPO](../strategies/grpo.md).

## Extractive QA answers

`answers` follows the SQuAD layout: an object with parallel `text` and `answer_start` lists. `answer_start` counts **characters** in `context`, starting at 0.

```json
{"text": ["1999"], "answer_start": [22]}
```

- Only the first answer in each list is used for training.
- A single string and a single integer (`{"text": "1999", "answer_start": 22}`) are also accepted.
- Empty lists mark an unanswerable question. The model is trained to point at the tokenizer's CLS token.
- Long contexts are split into overlapping windows of `max_length` tokens (256 below 12 GB of VRAM, 384 otherwise) with a 128-token stride. A window that doesn't contain the whole answer is labelled as unanswerable.
- The model's tokenizer must have a CLS token, otherwise training fails with `Extractive QA requires a tokenizer with a CLS token.`

An off-by-one offset trains the model on the wrong span. Check offsets with `context[start:start + len(text)] == text`.

Use JSONL, JSON or Parquet for QA data. CSV can't store the nested `answers` object.

## Validating a dataset

**Validate** in the console, the notebook CLI and `POST /api/datasets/validate` run these steps. The terminal wizard runs the same checks during its dataset step and offers to pick another dataset if they fail.

1. Check that a local path exists.
2. Load the dataset (downloading Hub datasets in full).
3. Take the first split.
4. Apply aliases and the ambiguity rule.
5. Run the strategy's own format check.
6. Report the normalized columns and row count, or the first error.

```bash
curl -s -X POST http://127.0.0.1:8000/api/datasets/validate \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"path": "./data.jsonl", "remote": false, "use_case_slug": "instruction_following", "objective": "supervised"}'
```

```json
{"valid": true, "path": "./data.jsonl", "remote": false, "columns": ["prompt", "completion"],
 "format_description": "A dataset with a 'prompt' column ...", "error": null, "row_count": 3}
```

Validation checks column names, not values. Problems inside rows, such as a preference pair without a shared prompt, appear when the run starts and fail the run with the same message.

### Validation errors

| Message | Cause | Fix |
|---|---|---|
| `Dataset path not found: <path>` | The local path doesn't exist on the server's machine. | Use an absolute path, or one relative to where ModelForge was started. |
| `Dataset is missing required column(s): <names>.` | Neither an alias nor its canonical column is present. The message lists the **alias** names. | Rename columns to the canonical or alias names in [Required columns](#required-columns). |
| `Dataset contains both an alias and its canonical training column: ...` | See [The ambiguity rule](#the-ambiguity-rule). | Drop or rename one column. |
| `Dataset must contain the following columns: {'chosen', 'rejected'}...` | DPO check. | Add `chosen` and `rejected`. |
| `GRPO datasets must contain 'prompt' and 'answer' columns...` | GRPO check. | Add both columns. |
| `Extractive QA datasets require context, question, and answers columns. Missing: ...` | QA check. | Add the missing columns. |
| `No supported dataset files found in directory: <path>` | A folder with no supported file types. | See [Folders](#folders). |
| `Invalid JSON structure for dataset loading: ...` | A `.json` file that is neither JSONL, an array, nor an object with a list field. | Convert to JSONL. |
| `<Objective> is unavailable because strategy '<slug>' ...` | The objective has no registered strategy for this use case. | Pick another objective. |

## Streaming

None of ModelForge's interfaces loads datasets in streaming mode; files and Hub datasets are loaded in full. Streaming is only reachable when you call `DatasetFactory.load(..., streaming=True)` from your own code, and then:

- validation skips the column check and reports no columns or row count;
- single `.json` files must be JSONL, because the nested-field fallback is disabled;
- GRPO refuses the dataset: `GRPO does not support streaming datasets; load a finite dataset instead.`

## Data quality tips

- A few thousand clean, consistent examples usually beat a large noisy set.
- Keep formatting consistent across rows. The model learns your format as well as your content.
- Hold out some examples to try in the [Playground](../user-guide/playground.md) after training. ModelForge doesn't use an evaluation split.
- Check that most examples fit in the plan's `max_length` tokens. Longer ones are truncated.

## See also

- [Use cases and tasks](training-tasks.md)
- [How planning works](configuration-guide.md)
- [SFT](../strategies/sft.md), [DPO](../strategies/dpo.md), [GRPO](../strategies/grpo.md)
- [Common issues](../troubleshooting/common-issues.md)
