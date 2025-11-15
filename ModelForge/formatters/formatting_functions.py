"""
Task-specific formatting functions for Unsloth/TRL trainers.

These functions convert dataset examples into formatted text strings
that are ready for training. Each function must return a list of strings
as required by Unsloth.
"""

from typing import Callable, List


def create_text_generation_formatter(eos_token: str) -> Callable:
    """
    Create formatting function for text-generation tasks.

    Expected dataset fields after _format_dataset:
    - prompt: User input (raw text)
    - completion: Model output (raw text)

    Args:
        eos_token: The EOS token string from the tokenizer

    Returns:
        Formatting function that returns a list of formatted strings
    """
    def formatting_func(example: dict) -> List[str]:
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")

        # Format as USER/ASSISTANT conversation with EOS token
        text = f"USER: {prompt}\nASSISTANT: {completion}{eos_token}"

        return [text]  # Must return list as required by Unsloth

    return formatting_func


def create_summarization_formatter(eos_token: str) -> Callable:
    """
    Create formatting function for summarization tasks.

    Expected dataset fields after _format_dataset:
    - input: Document to summarize
    - output: Summary text

    Args:
        eos_token: The EOS token string from the tokenizer

    Returns:
        Formatting function that returns a list of formatted strings
    """
    def formatting_func(example: dict) -> List[str]:
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        # Format as document -> summary with EOS token
        text = f"Summarize the following document:\n{input_text}\n\nSummary:\n{output_text}{eos_token}"

        return [text]

    return formatting_func


def create_qa_formatter(eos_token: str) -> Callable:
    """
    Create formatting function for extractive question-answering tasks.

    Expected dataset fields (unchanged from original):
    - context: The context/passage
    - question: The question
    - answers: Dict with "text" field containing answer(s)

    Args:
        eos_token: The EOS token string from the tokenizer

    Returns:
        Formatting function that returns a list of formatted strings
    """
    def formatting_func(example: dict) -> List[str]:
        context = example.get("context", "")
        question = example.get("question", "")
        answers = example.get("answers", {})

        # Extract answer text (handle both dict and string formats)
        if isinstance(answers, dict):
            answer_text = answers.get("text", [""])[0] if "text" in answers else ""
        else:
            answer_text = str(answers)

        # Format as context + question -> answer with EOS token
        text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer_text}{eos_token}"

        return [text]

    return formatting_func


def get_formatting_func(task: str, eos_token: str) -> Callable:
    """
    Get the appropriate formatting function for a given task type.

    Args:
        task: Task type (e.g., "text-generation", "summarization", "extractive-question-answering")
        eos_token: The EOS token string from the tokenizer

    Returns:
        Formatting function for the specified task

    Raises:
        ValueError: If the task type is not supported
    """
    formatters = {
        "text-generation": create_text_generation_formatter,
        "summarization": create_summarization_formatter,
        "extractive-question-answering": create_qa_formatter,
    }

    formatter_creator = formatters.get(task)
    if formatter_creator is None:
        raise ValueError(
            f"No formatter available for task: {task}. "
            f"Supported tasks: {list(formatters.keys())}"
        )

    return formatter_creator(eos_token)
