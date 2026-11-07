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


def test_native_qa_training_and_selection(tmp_path):
    from types import SimpleNamespace
    from datasets import Dataset
    from workflow.service import WorkflowService
    tokenizer = local_tokenizer(tmp_path)
    model = BertForQuestionAnswering(BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=8, num_hidden_layers=1, num_attention_heads=1, intermediate_size=16))
    provider = SimpleNamespace(load_model=lambda *args, **kwargs: model, load_tokenizer=lambda *args: tokenizer)
    strategy = ExtractiveQA()
    strategy.set_config({"max_length": 32, "doc_stride": 8, "num_train_epochs": 1, "per_device_train_batch_size": 1})
    dataset = Dataset.from_dict({"question": ["where is paris"], "context": ["paris is france ."], "answers": [{"text": ["france"], "answer_start": [9]}]})
    trainer = strategy.finetune("tiny", dataset, "question_answering", provider, "full_finetune", str(tmp_path / "output"))
    assert trainer.state.global_step == 1
    assert any(item["slug"] == "extractive_qa" for item in WorkflowService().list_use_cases())
