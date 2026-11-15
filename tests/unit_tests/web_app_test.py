from __future__ import annotations

from queue import Queue

from fastapi.testclient import TestClient

from utilities.database_manager import WebRunLedger
from web.app import create_app
from web.auth import TokenAuth


class FakeWorkflow:
    def system_info(self):
        return {"name": "ModelForge", "package": "modelforge-finetuning"}

    def detect_hardware(self):
        class Hardware:
            def model_dump(self):
                return {"gpus": []}

        return Hardware()

    def list_use_cases(self):
        return [{"slug": "instruction_following", "name": "Instruction Following"}]


class FakeJobs:
    def __init__(self, ledger):
        self.workflow = FakeWorkflow()
        self.ledger = ledger

    def start_run(self, request):
        return {"id": 1, "status": "queued", "model_id": request.model_id}

    def cancel_run(self, run_id):
        raise ValueError(f"No active job found for run {run_id}.")

    def resume_run(self, run_id, checkpoint_path):
        raise ValueError("Checkpoint is not registered")

    def get_event_queue(self, run_id):
        return Queue()


def make_client(tmp_path):
    ledger = WebRunLedger(tmp_path / "runs.db")
    auth = TokenAuth("test-token")
    app = create_app(auth=auth, job_manager=FakeJobs(ledger))
    return TestClient(app)


def test_api_requires_startup_token(tmp_path):
    client = make_client(tmp_path)

    response = client.get("/api/system")

    assert response.status_code == 401


def test_api_accepts_startup_token(tmp_path):
    client = make_client(tmp_path)

    response = client.get("/api/system", headers={"X-ModelForge-Token": "test-token"})

    assert response.status_code == 200
    assert response.json()["package"] == "modelforge-finetuning"


SPA_ROUTES = (
    "/",
    "/app",
    "/app/run",
    "/app/run/7",
    "/app/history",
    "/app/models",
    "/app/playground",
    "/app/system",
    "/app/docs",
    "/app/artifacts",
    "/app/settings",
    "/app/live",
    "/app/runs/7",
)


def _client_with_spa(tmp_path, monkeypatch):
    dist = tmp_path / "frontend_dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>ModelForge</title>", encoding="utf-8")
    (dist / "logo.svg").write_text("<svg id='logo'></svg>", encoding="utf-8")
    monkeypatch.setattr("web.app.frontend_dist_path", lambda: dist)
    return make_client(tmp_path)


def test_frontend_is_served_without_api_token(tmp_path, monkeypatch):
    client = _client_with_spa(tmp_path, monkeypatch)

    response = client.get("/app")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ModelForge" in response.text


def test_spa_fallback_serves_landing_and_workbench_routes(tmp_path, monkeypatch):
    client = _client_with_spa(tmp_path, monkeypatch)

    bodies = []
    for path in SPA_ROUTES:
        response = client.get(path)
        assert response.status_code == 200, path
        assert "text/html" in response.headers["content-type"]
        assert "ModelForge" in response.text
        bodies.append(response.text)
    assert len(set(bodies)) == 1

    static = client.get("/logo.svg")
    assert static.status_code == 200
    assert "<svg" in static.text

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"


def test_run_errors_are_http_responses(tmp_path):
    client = make_client(tmp_path)

    response = client.post("/api/runs/99/cancel", headers={"X-ModelForge-Token": "test-token"})

    assert response.status_code == 404
    assert "No active job" in response.text


def test_websocket_stop_is_processed_during_generation(tmp_path):
    import threading
    client = make_client(tmp_path)
    stopped = threading.Event()

    class Session:
        def generate(self, messages, **sampling):
            yield "first"
            if not stopped.wait(2):
                raise RuntimeError("Stop was never delivered during generation")

        def stop(self):
            stopped.set()

    client.app.state.session_manager.get = lambda session_id: Session()
    with client.websocket_connect("/api/playground/test/chat?token=test-token") as socket:
        socket.send_json({"messages": [{"role": "user", "content": "hello"}]})
        assert socket.receive_json()["type"] == "start"
        assert socket.receive_json()["type"] == "token"
        socket.send_json({"type": "stop"})
        assert socket.receive_json()["type"] == "done"
    assert stopped.is_set()
