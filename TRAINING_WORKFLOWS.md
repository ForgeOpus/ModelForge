# Training workflows

The browser, terminal, and notebook workflows share dataset normalization and
training execution. Dataset aliases are renamed before validation and execution;
supplying both an alias and its canonical column is rejected as ambiguous.

## Supervised training

Instruction following uses `prompt` and `completion` (`instruction` and `response`
are accepted aliases). Text generation uses `prompt` and `completion` (`input_text`
and `generated_text` are accepted aliases).

Summarization accepts `document` and `summary`, normalized to `prompt` and
`completion`. It uses a native encoder-decoder trainer rather than TRL's causal
language-model trainer.

## Preference training

Instruction following and Text Generation with the preference objective use DPO.
Both require `chosen` and `rejected`; a shared `prompt` is optional when the pair
already contains its context. Instruction Following accepts `instruction`,
`preferred_response`, and `dispreferred_response` aliases. Text Generation accepts
`input_text`, `preferred_continuation`, and `rejected_continuation` aliases.

DPO aligns an existing model's responses. For instruction tuning, start from a
pretrained instruction/chat checkpoint, or perform supervised instruction tuning
before DPO. Tiny/random models published for library testing cannot provide
meaningful chat responses and are excluded from automatic recommendations.

Recommendations search within the capacity of all supported training methods,
then prefer the largest model in the popular candidate shortlist. The engine
selects full fine-tuning, LoRA, 8-bit QLoRA, or 4-bit QLoRA according to what fits
that model, preferring less quantization when possible. Instruction-following
searches require chat metadata or an instruction/chat model name. Adapter
replicas must fit the smallest selected GPU. System RAM is not treated as extra
VRAM; automatic CPU offloading for training is not configured. Capacity estimates
are conservative planning bounds rather than measured guarantees of fit.

Selecting another model in the new-run form recalculates its training method,
quantization, learning rate, adapter settings, and parallelism. Start remains
disabled until that model's plan is ready. The recommendations API also accepts
`model_id` to plan a specific model using its Hub parameter metadata. Launching
with a different model from a supplied shortlist recalculates the configuration
before recording the run.

## Chat formatting and saved tokenizers

Causal prompt/completion examples, DPO preference pairs, and GRPO prompts are
normalized to role/content messages before the trainer applies the tokenizer's
chat template. Full preference conversations are split into a shared prompt and
assistant continuations; conflicting explicit prompts are rejected. Normalization
also covers evaluation splits. Raw `text` language-modeling data and native
encoder-decoder tasks retain their own formats.

The model's native chat template is preserved. Tokenizers without one receive a
plain `User:`/`Assistant:` template using their existing EOS token. The final model
folder includes the tokenizer and template, including for adapter-only runs.
Both playground paths use that template and tokenize it once, avoiding duplicate
special tokens. A template gives data a consistent structure; it does not give
an untrained model language ability.

## TensorBoard logs

The TensorBoard callback writes to `tensorboard/run-<id>` within the output
directory. Resuming creates a new run directory; custom log roots also receive
run subdirectories. CLI/notebook launches use unique `run-<uuid>` directories.
The viewer watches the shared parent so each run has its own named curves and
can be selected or compared independently. Existing event files are retained.
The callback is configured
directly because Transformers 5 ignores the deprecated training-argument
`logging_dir`. Logging includes the first training step.

## Reinforcement training

Instruction Following and Text Generation with the reinforcement objective use GRPO.
Supply `prompt` and `answer` strings. Instruction Following accepts `instruction`
and `ideal_answer`; Text Generation accepts `input_text` and `ideal_continuation`.
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
