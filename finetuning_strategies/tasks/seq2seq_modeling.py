from finetuning_strategies.tasks.task import FinetuningTask
from transformers import AutoModelForSeq2SeqLM


class Seq2SeqLM(FinetuningTask):
    """Sequence-to-sequence language modeling task (e.g. T5, BART).

    Used for tasks where the input and output are both sequences of variable length,
    such as summarization, translation, and question answering.
    """

    name = "Sequence-to-Sequence Modeling"
    slug = "seq2seq_modeling"
    description = (
        "Sequence-to-sequence (Seq2Seq) modeling trains encoder-decoder architectures "
        "to transform an input sequence into an output sequence. Common applications include "
        "summarization, machine translation, and abstractive question answering."
    )
    model_class = AutoModelForSeq2SeqLM
