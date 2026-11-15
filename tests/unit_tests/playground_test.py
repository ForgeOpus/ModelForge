from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from web.auth import TokenAuth
from web.app import create_app
from web.playground import InferenceRequest, Playground, PlaygroundError


class Ledger:
    def __init__(self, path, task="causal_language_modeling", status="completed"):
        self.run = {"id": 1, "status": status, "output_dir": str(path), "model_id": str(path), "task_slug": task}
    def get_run(self, run_id):
        return self.run if run_id == 1 else None
    def list_checkpoints(self, run_id):
        return []


class Jobs:
    def __init__(self, ledger, active=False):
        self.ledger = ledger
        self.workflow = object()
        self.resource_gate = threading.Lock()
        self.active = active
    def has_active_jobs(self):
        return self.active


def request(**overrides):
    return InferenceRequest(**({"run_id": 1, "messages": [{"role": "user", "content": "hello"}], "context_window": 1024, "max_new_tokens": 2} | overrides))


@pytest.fixture
def tiny_model(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3, "User": 4, "Assistant": 5, ":": 6}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    fast.save_pretrained(tmp_path)
    GPT2LMHeadModel(GPT2Config(vocab_size=7, n_positions=64, n_ctx=64, n_embd=8, n_layer=1, n_head=1, eos_token_id=2, pad_token_id=1)).save_pretrained(tmp_path)
    return tmp_path


def test_actual_cpu_generation_and_model_context_limit(tiny_model, monkeypatch):
    # Exercise the CPU placement branch explicitly; the CI/desktop host may
    # legitimately have a CUDA device available.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=128 * 1024**3))
    ledger = Ledger(tiny_model)
    service = Playground(Jobs(ledger), ledger)
    result = service.infer(request())
    assert result["device"] == "cpu"
    assert result["context_limit"] == 64
    assert isinstance(result["reply"], str)
    with pytest.raises(PlaygroundError, match="exceeding"):
        service.infer(request(messages=[{"role":"user","content":"hello " * 70}]))


def test_low_memory_refuses_before_loading(tiny_model, monkeypatch):
    import transformers
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=1024**3))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("model must not load"))
    ledger = Ledger(tiny_model)
    with pytest.raises(PlaygroundError, match="while training is active") as error:
        Playground(Jobs(ledger, active=True), ledger).infer(request())
    assert error.value.status == 409


def test_admission_and_artifact_checks(tmp_path):
    ledger = Ledger(tmp_path, status="running")
    jobs = Jobs(ledger)
    service = Playground(jobs, ledger)
    with pytest.raises(PlaygroundError, match="completed run"):
        service.infer(request())
    with pytest.raises(PlaygroundError, match="not registered"):
        service.infer(request(checkpoint_path=str(tmp_path)))
    jobs.resource_gate.acquire()
    try:
        with pytest.raises(PlaygroundError, match="busy"):
            service.infer(request())
    finally:
        jobs.resource_gate.release()


def test_endpoint_requires_auth_and_valid_messages(tmp_path):
    ledger = Ledger(tmp_path)
    client = TestClient(create_app(auth=TokenAuth("secret"), job_manager=Jobs(ledger)))
    body = {"run_id": 1, "messages": [{"role":"assistant", "content":"bad"}]}
    assert client.post("/api/playground/infer", json=body).status_code == 401
    assert client.post("/api/playground/infer", headers={"X-ModelForge-Token":"secret"}, json=body).status_code == 422


def test_session_playground_routes_use_the_session_manager(tmp_path):
    """The legacy inference service and websocket-session manager coexist."""
    ledger = Ledger(tmp_path)
    client = TestClient(create_app(auth=TokenAuth("secret"), job_manager=Jobs(ledger)))
    response = client.get("/api/playground/status", headers={"X-ModelForge-Token": "secret"})
    assert response.status_code == 200
    assert response.json()["available"] is True


def test_session_holds_resource_gate_until_closed(monkeypatch, tmp_path):
    from workflow.playground import PlaygroundManager

    class FakeSession:
        def __init__(self, session_id, model_path):
            self.id = session_id
            self.model_path = model_path
            self.base_model = "base"
        def close(self):
            pass

    monkeypatch.setattr("workflow.playground.PlaygroundSession", FakeSession)
    jobs = Jobs(Ledger(tmp_path))
    manager = PlaygroundManager(jobs)
    session = manager.open(str(tmp_path / "model"))
    assert jobs.resource_gate.locked()
    assert manager.close(session.id) is True
    assert not jobs.resource_gate.locked()


def test_adapter_cpu_generation(tiny_model, tmp_path, monkeypatch):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=128 * 1024**3))
    adapter_path = tmp_path / "adapter"
    base = AutoModelForCausalLM.from_pretrained(tiny_model)
    adapter = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, target_modules=["c_attn"]))
    adapter.save_pretrained(adapter_path)
    ledger = Ledger(adapter_path)
    ledger.run["model_id"] = str(tiny_model)
    result = Playground(Jobs(ledger, active=True), ledger).infer(request())
    assert result["training_active"] is True
    assert result["device"] == "cpu"


def test_token_classification(tiny_model, tmp_path, monkeypatch):
    from transformers import BertConfig, BertForTokenClassification, AutoTokenizer
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=128 * 1024**3))
    path = tmp_path / "classifier"
    BertForTokenClassification(BertConfig(vocab_size=7, max_position_embeddings=64, hidden_size=8, num_hidden_layers=1, num_attention_heads=1, intermediate_size=16, num_labels=2, id2label={0:"O", 1:"ENTITY"})).save_pretrained(path)
    AutoTokenizer.from_pretrained(tiny_model).save_pretrained(path)
    ledger = Ledger(path, task="token_classification")
    result = Playground(Jobs(ledger), ledger).infer(request())
    assert result["task"] == "token_classification"
    assert "hello:" in result["reply"]


def test_training_rejected_during_inference(tmp_path):
    from workflow.jobs import TrainingJobManager
    manager = TrainingJobManager(workflow=object(), ledger=object(), recover_interrupted=False)
    manager.resource_gate.acquire()
    try:
        with pytest.raises(RuntimeError, match="Playground inference"):
            manager.start_run(None)
    finally:
        manager.resource_gate.release()


def test_seq2seq_generation(tiny_model, tmp_path, monkeypatch):
    from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=128 * 1024**3))
    path = tmp_path / "seq2seq"
    config = T5Config(vocab_size=7, d_model=8, d_ff=16, num_layers=1, num_decoder_layers=1, num_heads=1, decoder_start_token_id=1, pad_token_id=1, eos_token_id=2)
    T5ForConditionalGeneration(config).save_pretrained(path)
    AutoTokenizer.from_pretrained(tiny_model).save_pretrained(path)
    ledger = Ledger(path, task="seq2seq_modeling")
    result = Playground(Jobs(ledger), ledger).infer(request())
    assert result["task"] == "seq2seq_modeling"
    assert isinstance(result["reply"], str)



def test_budget_protects_active_training():
    from web.playground import memory_budgets
    class FakeCuda:
        def is_available(self): return True
        def device_count(self): return 1
        def mem_get_info(self, index): return (20 * 1024**3, 24 * 1024**3)
    fake = SimpleNamespace(cuda=FakeCuda())
    assert 0 in memory_budgets(fake, 64 * 1024**3, 1024**3, False)
    assert 0 not in memory_budgets(fake, 64 * 1024**3, 1024**3, True)
    with pytest.raises(PlaygroundError, match="Not enough available RAM"):
        memory_budgets(fake, 1024**3, 1024**3, True)


def test_real_disk_offload_and_cleanup(tiny_model, tmp_path, monkeypatch):
    import accelerate
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=128 * 1024**3))
    monkeypatch.setenv("MODELFORGE_OFFLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(accelerate, "infer_auto_device_map", lambda *args, **kwargs: {
        "transformer.wte": "cpu", "transformer.wpe": "cpu", "transformer.drop": "cpu",
        "transformer.h.0": "disk", "transformer.ln_f": "cpu", "lm_head": "cpu"})
    ledger = Ledger(tiny_model)
    result = Playground(Jobs(ledger, active=True), ledger).infer(request())
    assert result["device"] == "CPU + disk"
    assert not list(tmp_path.glob("modelforge-offload-*"))


def test_disk_space_rejected_before_loading(tiny_model, tmp_path, monkeypatch):
    import accelerate
    import transformers
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=128 * 1024**3))
    monkeypatch.setenv("MODELFORGE_OFFLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(accelerate, "infer_auto_device_map", lambda *args, **kwargs: {"transformer": "cpu", "lm_head": "disk"})
    monkeypatch.setattr("web.playground.shutil.disk_usage", lambda path: SimpleNamespace(free=0))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("model must not load"))
    ledger = Ledger(tiny_model)
    with pytest.raises(PlaygroundError, match="disk space"):
        Playground(Jobs(ledger), ledger).infer(request())
    assert not list(tmp_path.glob("modelforge-offload-*"))


def test_adapter_with_real_disk_offload(tiny_model, tmp_path, monkeypatch):
    import accelerate
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=128 * 1024**3))
    monkeypatch.setenv("MODELFORGE_OFFLOAD_DIR", str(tmp_path))
    adapter_path = tmp_path / "adapter"
    base = AutoModelForCausalLM.from_pretrained(tiny_model)
    get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, target_modules=["c_attn"])).save_pretrained(adapter_path)
    original_plan = accelerate.infer_auto_device_map
    def plan(model, **kwargs):
        if model.__class__.__name__ == "GPT2LMHeadModel":
            return {"transformer.wte": "cpu", "transformer.wpe": "cpu", "transformer.drop": "cpu", "transformer.h.0": "disk", "transformer.ln_f": "cpu", "lm_head": "cpu"}
        return original_plan(model, **kwargs)
    monkeypatch.setattr(accelerate, "infer_auto_device_map", plan)
    ledger = Ledger(adapter_path)
    ledger.run["model_id"] = str(tiny_model)
    result = Playground(Jobs(ledger, active=True), ledger).infer(request())
    assert isinstance(result["reply"], str)
    assert not list(tmp_path.glob("modelforge-offload-*"))


def test_disk_offload_cleaned_up_after_generation_error(tiny_model, tmp_path, monkeypatch):
    import accelerate
    from transformers import GPT2LMHeadModel
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=128 * 1024**3))
    monkeypatch.setenv("MODELFORGE_OFFLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(accelerate, "infer_auto_device_map", lambda *args, **kwargs: {
        "transformer.wte": "cpu", "transformer.wpe": "cpu", "transformer.drop": "cpu",
        "transformer.h.0": "disk", "transformer.ln_f": "cpu", "lm_head": "cpu"})
    monkeypatch.setattr(GPT2LMHeadModel, "generate", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated failure")))
    ledger = Ledger(tiny_model)
    with pytest.raises(PlaygroundError, match="simulated failure"):
        Playground(Jobs(ledger), ledger).infer(request())
    assert not list(tmp_path.glob("modelforge-offload-*"))


def test_placement_description_for_mixed_devices():
    from web.playground import placement_description
    assert placement_description({"a": 0, "b": "cpu", "c": "disk"}) == ("GPU + CPU + disk", "cuda:0")
    assert placement_description({"a": "cpu", "b": "disk"}) == ("CPU + disk", "cpu")


def test_automatic_disk_placement(tiny_model, tmp_path, monkeypatch):
    monkeypatch.setattr("web.playground.memory_budgets", lambda *args: {"cpu": 4000})
    monkeypatch.setenv("MODELFORGE_OFFLOAD_DIR", str(tmp_path))
    ledger = Ledger(tiny_model)
    result = Playground(Jobs(ledger, active=True), ledger).infer(request())
    assert "disk" in result["device"]
    assert not list(tmp_path.glob("modelforge-offload-*"))



def test_adapter_loading_ignores_stale_full_model_in_checkpoint(tiny_model, tmp_path, monkeypatch, caplog):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=128 * 1024**3))
    path = tmp_path / 'mixed-checkpoint'
    # Simulate a reused checkpoint folder holding weights with incompatible
    # dimensions, plus a newer adapter trained against the declared base.
    GPT2LMHeadModel(GPT2Config(vocab_size=7, n_positions=64, n_embd=16, n_layer=2, n_head=1)).save_pretrained(path)
    base = AutoModelForCausalLM.from_pretrained(tiny_model)
    get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, target_modules=['c_attn'])).save_pretrained(path)
    original = AutoModelForCausalLM.from_pretrained
    sources = []
    def load(source, *args, **kwargs):
        sources.append(str(source))
        return original(source, *args, **kwargs)
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', load)
    ledger = Ledger(path)
    ledger.run['model_id'] = str(tiny_model)
    result = Playground(Jobs(ledger, active=True), ledger).infer(request())
    assert isinstance(result['reply'], str)
    assert sources == [str(tiny_model)]
    assert 'MISMATCH' not in caplog.text
    assert (path / 'model.safetensors').exists()  # never repair artifacts by deleting them
class _LiveThread:
    def is_alive(self):
        return True


class _StubSession:
    def __init__(self, session_id, model_path):
        self.id = session_id
        self.model_path = model_path
        self.base_model = "stub-base"
        self.device_note = "stub note"
        self.stopped = False

    def close(self):
        self.stopped = True

    def stop(self):
        self.stopped = True

    def generate(self, messages, **sampling):
        yield "hello"


def _route_client(tmp_path, *, active=False, threads=None):
    ledger = Ledger(tmp_path)
    jobs = Jobs(ledger, active=active)
    if threads is not None:
        jobs._threads = threads
    app = create_app(auth=TokenAuth("secret"), job_manager=jobs)
    return TestClient(app), app


def test_playground_routes_use_distinct_services(tmp_path, monkeypatch):
    monkeypatch.setattr("workflow.playground.PlaygroundSession", _StubSession)
    client, app = _route_client(tmp_path)
    headers = {"X-ModelForge-Token": "secret"}
    from workflow.playground import PlaygroundManager

    assert isinstance(app.state.session_manager, PlaygroundManager)
    assert isinstance(app.state.inference_playground, Playground)
    assert app.state.session_manager is not app.state.inference_playground
    assert not hasattr(app.state, "playground")

    for response in (
        client.get("/api/playground/status"),
        client.post("/api/playground/sessions", json={"model_path": str(tmp_path)}),
        client.delete("/api/playground/sessions/missing"),
        client.post("/api/playground/infer", json={"run_id": 1, "messages": [{"role": "user", "content": "hello"}]}),
    ):
        assert response.status_code == 401

    status = client.get("/api/playground/status", headers=headers)
    assert status.status_code == 200
    assert status.json() == {"available": True, "reason": None, "session_id": None, "model_path": None}

    opened = client.post("/api/playground/sessions", headers=headers, json={"model_path": str(tmp_path)})
    assert opened.status_code == 200
    body = opened.json()
    assert body["model_path"] == str(tmp_path)
    assert body["base_model"] == "stub-base"
    assert body["note"] == "stub note"
    session_id = body["session_id"]

    status = client.get("/api/playground/status", headers=headers)
    assert status.json()["session_id"] == session_id
    assert status.json()["model_path"] == str(tmp_path)

    closed = client.delete(f"/api/playground/sessions/{session_id}", headers=headers)
    assert closed.status_code == 200
    assert closed.json() == {"closed": True}

    inferred = client.post(
        "/api/playground/infer",
        headers=headers,
        json={"run_id": 99, "messages": [{"role": "user", "content": "hello"}]},
    )
    assert inferred.status_code == 404
    assert inferred.json()["detail"] == "Run not found."

    again = client.delete(f"/api/playground/sessions/{session_id}", headers=headers)
    assert again.json() == {"closed": False}
    assert client.get("/api/playground/status", headers=headers).json()["session_id"] is None


def test_training_active_refuses_session_without_breaking_infer(tmp_path):
    client, app = _route_client(tmp_path, active=True, threads={1: _LiveThread()})
    app.state.inference_playground.ledger.run["status"] = "running"
    headers = {"X-ModelForge-Token": "secret"}

    status = client.get("/api/playground/status", headers=headers)
    assert status.status_code == 200
    assert status.json()["available"] is False
    assert status.json()["session_id"] is None
    assert "training" in status.json()["reason"].lower()

    opened = client.post("/api/playground/sessions", headers=headers, json={"model_path": str(tmp_path)})
    assert opened.status_code == 409
    assert "training run" in opened.json()["detail"].lower()
    assert app.state.session_manager.current is None

    inferred = client.post(
        "/api/playground/infer",
        headers=headers,
        json={"run_id": 1, "messages": [{"role": "user", "content": "hello"}]},
    )
    assert inferred.status_code == 409
    assert "completed run" in inferred.json()["detail"]


def test_playground_websocket_auth_and_session_lookup(tmp_path, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr("workflow.playground.PlaygroundSession", _StubSession)
    client, _app = _route_client(tmp_path)
    headers = {"X-ModelForge-Token": "secret"}

    with pytest.raises(WebSocketDisconnect) as rejected:
        with client.websocket_connect("/api/playground/missing/chat?token=wrong") as socket:
            socket.receive_text()
    assert rejected.value.code == 1008

    with client.websocket_connect("/api/playground/missing/chat?token=secret") as socket:
        missing = socket.receive_json()
        assert missing["type"] == "error"
        assert "closed" in missing["message"]

    opened = client.post("/api/playground/sessions", headers=headers, json={"model_path": str(tmp_path)})
    session_id = opened.json()["session_id"]
    with client.websocket_connect(f"/api/playground/{session_id}/chat?token=secret") as socket:
        socket.send_json({"messages": [{"role": "user", "content": "hi"}], "max_new_tokens": 4})
        assert socket.receive_json() == {"type": "start"}
        assert socket.receive_json() == {"type": "token", "text": "hello"}
        assert socket.receive_json() == {"type": "done"}
