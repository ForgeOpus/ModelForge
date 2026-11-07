import torch
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizerFast

from finetuning_strategies.strategies.qa import ExtractiveQA


def local_tokenizer(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "where", "is", "paris", "france", "."]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))


def test_span_preprocessing_and_tiny_local_model_forward(tmp_path):
    tokenizer = local_tokenizer(tmp_path)
    features = ExtractiveQA._prepare_features(
        tokenizer,
        {
            "question": ["where is paris"],
            "context": ["paris is france ."],
            "answers": [{"text": ["france"], "answer_start": [9]}],
        },
        max_length=32,
        doc_stride=8,
    )

    assert features["start_positions"][0] <= features["end_positions"][0]
    assert features["start_positions"][0] != features["input_ids"][0].index(tokenizer.cls_token_id)
    model = BertForQuestionAnswering(BertConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32,
    ))
    output = model(
        input_ids=torch.tensor(features["input_ids"]),
        attention_mask=torch.tensor(features["attention_mask"]),
        start_positions=torch.tensor(features["start_positions"]),
        end_positions=torch.tensor(features["end_positions"]),
    )
    assert output.loss is not None
