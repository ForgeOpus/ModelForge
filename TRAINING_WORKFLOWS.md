# Training workflows

The browser, terminal, and notebook workflows share dataset normalization and
training execution. Dataset aliases are renamed before validation and execution;
supplying both an alias and its canonical column is rejected as ambiguous.

## Supervised training

Instruction following uses `prompt` and `completion` (`instruction` and `response`
are accepted aliases). Text generation accepts a `text` column.

Summarization accepts `document` and `summary`, normalized to `prompt` and
`completion`. It uses a native encoder-decoder trainer rather than TRL's causal
language-model trainer.

## Preference training

Instruction following with the preference objective uses DPO. Supply `prompt`,
`chosen`, and `rejected`, using those exact column names, or the documented aliases
`instruction`, `preferred_response`, and `dispreferred_response`.

## Reinforcement training

Instruction following with the reinforcement objective uses GRPO. Supply `prompt`
and `answer` strings; `instruction` and `ideal_answer` are accepted aliases.
The built-in reward gives 1 for a completion matching the reference answer after
trimming surrounding whitespace, and 0 otherwise. It does not score general
helpfulness. Use tasks with objectively checkable reference answers. Streaming
datasets are not supported for this objective.

## Extractive question answering

Choose Extractive QA with the supervised objective. Supply `context`, `question`,
and `answers`. Each answers value contains `text` and `answer_start` lists, where
offsets count characters in the context. Training creates start/end token labels
and sliding windows for longer passages with a question-answering model head.

In the playground, send JSON such as:

```json
{"question": "Where is Paris?", "context": "Paris is in France."}
```

The playground returns a span from the context. Leave the playground or unload
the model to release its session before starting training.
