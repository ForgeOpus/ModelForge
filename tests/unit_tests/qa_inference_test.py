import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForQuestionAnswering, PreTrainedTokenizerFast

from utilities.qa_inference import decode_answer, parse_question


def test_span_decoding_excludes_question_and_invalid_order():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "where": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    encoding = fast("where", "hello world", return_tensors="pt", return_offsets_mapping=True)
    result = SimpleNamespace(start_logits=torch.tensor([[100., 5., 1.]]), end_logits=torch.tensor([[100., 0., 5.]]))
    assert decode_answer(result, encoding, "hello world") == "hello world"
    with pytest.raises(ValueError, match="question"):
        parse_question("a bare chat message")


def test_real_qa_playground_session(tmp_path, monkeypatch):
    from workflow.playground import PlaygroundSession
    from tokenizers.processors import TemplateProcessing
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "where": 4, "hello": 5, "world": 6}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", model_max_length=64)
    fast.save_pretrained(tmp_path)
    BertForQuestionAnswering(BertConfig(vocab_size=7, hidden_size=8, num_hidden_layers=1, num_attention_heads=1, intermediate_size=16, max_position_embeddings=64)).save_pretrained(tmp_path)
    (tmp_path / "modelforge_config.json").write_text(json.dumps({"task_slug": "question_answering"}))
    session = PlaygroundSession("qa", str(tmp_path))
    try:
        answer = "".join(session.generate([{"role": "user", "content": json.dumps({"question": "where", "context": "hello world"})}]))
        assert answer in {"hello", "world", "hello world"}
    finally:
        session.close()
    from web.playground import Playground, InferenceRequest
    import threading
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=128 * 1024**3))
    ledger = SimpleNamespace(get_run=lambda run_id: {"status": "completed", "output_dir": str(tmp_path), "model_id": str(tmp_path), "task_slug": "question_answering"})
    jobs = SimpleNamespace(resource_gate=threading.Lock(), has_active_jobs=lambda: False)
    result = Playground(jobs, ledger).infer(InferenceRequest(run_id=1, messages=[{"role": "user", "content": json.dumps({"question": "where", "context": "hello world"})}]))
    assert result["reply"] in {"hello", "world", "hello world"}
