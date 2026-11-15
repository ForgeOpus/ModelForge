# Fine-Tuning Concepts

A pretrained model already knows language. Fine-tuning keeps training it on a small, focused dataset so it picks up a particular behaviour: following your instructions, writing in your format, or answering from a document. This page covers the ideas you need to read a ModelForge plan.

## Pretraining and fine-tuning

Base models on the Hub are pretrained on huge amounts of text to predict the next token. That gives them grammar, facts and reasoning, but not your task. Fine-tuning continues training from those weights with a learning rate small enough that the model adapts without forgetting what it knows.

Because the starting point is already capable, a few thousand good examples often matter more than millions of mediocre ones.

## What actually changes

Training adjusts the model's weights to make your examples more likely. There are two ways to do that.

| | Full fine-tuning | Parameter-efficient fine-tuning (PEFT) |
|---|---|---|
| What trains | Every weight | A small adapter beside frozen weights (LoRA) |
| Memory per parameter in the planner | 18 bytes | 2 bytes (LoRA bf16), 1 byte (QLoRA 8-bit), 0.5 bytes (QLoRA 4-bit), plus 0.1 GB for adapters |
| Quality | Highest for a given base model | Slightly lower, but lets you train much larger models in the same memory |
| What gets saved | A full copy of the model | Only the adapter, plus the tokenizer and `modelforge_config.json` |

Full fine-tuning stores a gradient and two Adam optimizer states for every weight, which is why it costs about nine times more memory per parameter than LoRA. QLoRA is LoRA on top of base weights quantized to 8 or 4 bits with bitsandbytes.

ModelForge decides which to use from your hardware. It computes a parameter ceiling for each tier (full fine-tuning, LoRA bf16, QLoRA 8-bit, QLoRA 4-bit), picks the largest popular Hub model that fits under any of them, and then trains it with the least-quantized tier that model fits in. The plan names the choice under **PEFT decision**. See [Full Fine-Tuning](../strategies/full-finetuning.md), [LoRA and QLoRA](../strategies/qlora.md) and [Hardware and Memory](../configuration/hardware-profiles.md).

## Objectives at a glance

The objective is how the model learns. Supervised training copies example answers, preference training learns which of two answers is better, and reinforcement training rewards answers that are correct.

| Objective | Strategy | You provide | The model learns |
|---|---|---|---|
| Supervised | `sft` | prompt and completion | To produce the completion for each prompt. |
| Preference | `dpo` | chosen and rejected answers (prompt optional) | To prefer chosen over rejected, without a separate reward model. |
| Reinforcement | `grpo` | prompt and a reference answer | To generate answers that match the reference. |

Summarization and extractive QA use the supervised objective with their own trainers (a sequence-to-sequence trainer and a QA trainer with start/end labels).

```json
{"prompt": "Summarise this ticket: …", "completion": "Customer cannot log in after the update."}
{"prompt": "Explain LoRA", "chosen": "LoRA trains two small matrices…", "rejected": "It is a thing."}
{"prompt": "What is 17 × 3? Reply with the number only.", "answer": "51"}
```

!!! warning "The built-in GRPO reward is exact match"
    A completion scores 1 when it equals the reference answer after trimming surrounding whitespace, and 0 otherwise. Use GRPO for tasks with one checkable answer, such as arithmetic, classification labels or short extractions. It won't reward general helpfulness, and custom reward functions aren't supported.

Which should you use?

- Start with **Supervised**. It's the most forgiving and the cheapest.
- Add **Preference** when outputs are correct but you prefer some over others. DPO aligns existing behaviour, so start from an instruction-tuned model or run SFT first.
- Use **Reinforcement** when you can't write the ideal answer, but you can check whether an answer is right.

Details: [SFT](../strategies/sft.md), [DPO](../strategies/dpo.md), [GRPO](../strategies/grpo.md).

## Use cases

A use case is the goal in plain terms. Choosing one fixes which kind of model ModelForge searches for, which head it loads, and which objectives it can train.

| Use case | Model type | Objectives |
|---|---|---|
| Instruction Following | Causal LM (`text-generation`) | Supervised, Preference, Reinforcement |
| Text Generation | Causal LM (`text-generation`) | Supervised, Preference, Reinforcement |
| Summarization | Encoder-decoder (`summarization`) | Supervised |
| Extractive QA | Encoder with QA head (`question-answering`) | Supervised |

See [Use Cases and Tasks](../configuration/training-tasks.md).

## When fine-tuning helps

| Good fit | Poor fit |
|---|---|
| A consistent format or tone across outputs | Teaching new facts that change often (use retrieval instead) |
| A narrow task the base model half-does | Tasks a good prompt already solves |
| Preferring one style of answer over another | Very small datasets of a few dozen rows |
| Extracting answers from passages | Making a small model as capable as a much larger one |

## The loop in ModelForge

1. Pick what you want the model to do: a use case and an objective.
2. Prepare a dataset with the columns that objective needs. See [Dataset Formats](../configuration/dataset-formats.md).
3. Generate a plan and start training.
4. Try the result in the [Playground](../user-guide/playground.md), then adjust the data and train again.

## See also

- [Quick Start](quickstart.md)
- [How Planning Works](../configuration/configuration-guide.md)
- [Strategy Overview](../strategies/overview.md)
