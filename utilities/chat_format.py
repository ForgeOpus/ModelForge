"""Shared conversational data contract for training and inference."""

# Base tokenizers have no native chat syntax. Use ordinary text and the existing
# EOS token so no untrained vocabulary entries or embedding resize are needed.
FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] | title }}: {{ message['content'] }}"
    "{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}"
    "{{ '\\n\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
)


def ensure_chat_template(tokenizer):
    if not tokenizer.chat_template:
        if tokenizer.eos_token is None:
            raise ValueError("Chat training requires a tokenizer with an EOS token.")
        tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    return tokenizer


def chat_inputs(tokenizer, messages, **kwargs):
    """Tokenize exactly once; templates already include their special tokens."""
    ensure_chat_template(tokenizer)
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", **kwargs,
    )


def normalize_preference(example):
    from trl import extract_prompt

    prompt = example.get("prompt")
    chosen, rejected = example["chosen"], example["rejected"]
    if isinstance(chosen, list) != isinstance(rejected, list):
        raise ValueError("Chosen and rejected must both be strings or both be conversations.")
    if isinstance(chosen, list):
        # Full chats may coexist with a redundant text or message-list prompt.
        if chosen and rejected and chosen[0]["role"] != "assistant":
            pair = extract_prompt({"chosen": chosen, "rejected": rejected})
            if not pair["prompt"] or pair["prompt"][-1]["role"] not in {"user", "tool"}:
                raise ValueError("Preference conversations must share a prompt ending in a user or tool turn.")
            if isinstance(prompt, list) and prompt != pair["prompt"]:
                raise ValueError("Explicit prompt does not match the preference conversations.")
            if isinstance(prompt, str) and prompt != pair["prompt"][-1]["content"]:
                raise ValueError("Explicit prompt does not match the preference conversations.")
            return pair
        if prompt is None:
            raise ValueError("Assistant-only preference pairs require a shared prompt.")
    else:
        if prompt is None:
            pair = extract_prompt({"chosen": chosen, "rejected": rejected})
            prompt, chosen, rejected = pair["prompt"], pair["chosen"], pair["rejected"]
            if not prompt:
                raise ValueError("Text preference pairs need a shared prompt. Add a prompt column.")
        chosen = [{"role": "assistant", "content": chosen}]
        rejected = [{"role": "assistant", "content": rejected}]
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("Preference pairs require a nonempty shared prompt.")
    for response in (chosen, rejected):
        if not response or response[0]["role"] != "assistant":
            raise ValueError("Preference continuations must start with an assistant turn.")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def normalize_supervised(example):
    prompt, completion = example["prompt"], example["completion"]
    return {
        "prompt": [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt,
        "completion": [{"role": "assistant", "content": completion}] if isinstance(completion, str) else completion,
    }


def normalize_prompts(example):
    prompt = example["prompt"]
    return {"prompt": [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt}


def map_evaluation(dataset, normalizer):
    if dataset is None:
        return None
    if isinstance(dataset, dict):
        return {name: split.map(normalizer) for name, split in dataset.items()}
    return dataset.map(normalizer)
