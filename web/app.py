from __future__ import annotations

import asyncio
import json
import queue
import shutil
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utilities.database_manager import WebRunLedger
from web.auth import TokenAuth
from workflow.jobs import TrainingJobManager
from workflow.models import DatasetValidationRequest, RecommendationRequest, TrainingRunRequest
from workflow.playground import PlaygroundManager, PlaygroundUnavailable
from workflow.service import WorkflowService


class ResumeRequest(BaseModel):
    checkpoint_path: str


class ModelValidationRequest(BaseModel):
    model_id: str
    task_slug: str | None = None


class HubPushRequest(BaseModel):
    repo_id: str
    private: bool = False


class PlaygroundOpenRequest(BaseModel):
    model_path: str


def frontend_dist_path() -> Path:
    return Path(__file__).resolve().parent / "frontend_dist"


def create_app(
    auth: TokenAuth | None = None,
    job_manager: TrainingJobManager | None = None,
    tensorboard_host: str = "127.0.0.1",
    tensorboard_port: int = 6006,
) -> FastAPI:
    auth = auth or TokenAuth()
    if job_manager is None:
        workflow = WorkflowService()
        ledger = WebRunLedger()
        jobs = TrainingJobManager(workflow=workflow, ledger=ledger)
    else:
        jobs = job_manager
        workflow = jobs.workflow
        ledger = jobs.ledger

    app = FastAPI(
        title="ModelForge v4",
        description="Browser interface for modelforge-finetuning.",
        version="4.0.0",
    )
    playground = PlaygroundManager(jobs)

    app.state.auth = auth
    app.state.playground = playground
    app.state.workflow = workflow
    app.state.jobs = jobs
    app.state.ledger = ledger

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    require_auth = Depends(auth.dependency)

    def _with_tensorboard_url(run: dict | None) -> dict | None:
        """Attach the URL a user would open, without starting TensorBoard here.

        FinetuningStrategy.launch_tensorboard() calls webbrowser.open(), which
        would pop a browser on the training host rather than the viewer's
        machine. We only report where it would be served."""
        if run and run.get("tensorboard_log_dir"):
            run = {**run, "tensorboard_url": f"http://{tensorboard_host}:{tensorboard_port}/"}
        return run

    @app.get("/api/health")
    async def health():
        return {"status": "healthy", "version": "4.0.0"}

    @app.get("/api/system", dependencies=[require_auth])
    async def system():
        return workflow.system_info()

    @app.get("/api/hardware", dependencies=[require_auth])
    async def hardware():
        return workflow.detect_hardware().model_dump()

    @app.get("/api/use-cases", dependencies=[require_auth])
    async def use_cases():
        return {"use_cases": workflow.list_use_cases()}

    @app.post("/api/recommendations", dependencies=[require_auth])
    async def recommendations(request: RecommendationRequest):
        recommendation = workflow.recommend(request)
        return recommendation.model_dump()

    @app.post("/api/datasets/validate", dependencies=[require_auth])
    async def validate_dataset(request: DatasetValidationRequest):
        return workflow.validate_dataset(request).as_dict()

    @app.post("/api/runs", dependencies=[require_auth])
    async def create_run(request: TrainingRunRequest):
        try:
            return jobs.start_run(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs", dependencies=[require_auth])
    async def list_runs(limit: int = 100):
        return {"runs": [_with_tensorboard_url(run) for run in ledger.list_runs(limit=limit)]}

    @app.get("/api/runs/{run_id}", dependencies=[require_auth])
    async def get_run(run_id: int):
        run = ledger.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return _with_tensorboard_url(run)

    @app.post("/api/runs/{run_id}/cancel", dependencies=[require_auth])
    async def cancel_run(run_id: int):
        try:
            return jobs.cancel_run(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/events", dependencies=[require_auth])
    async def run_event_history(run_id: int):
        """Everything already emitted for this run.

        The websocket only carries events from the moment a client attaches, so
        without this a page reload mid-run shows an empty log. The Run screen
        paints this first, then layers the live stream on top."""
        getter = getattr(jobs, "get_event_history", None)
        return {"events": getter(run_id) if getter else []}

    @app.get("/api/runs/{run_id}/checkpoints", dependencies=[require_auth])
    async def list_checkpoints(run_id: int):
        return {"checkpoints": ledger.list_checkpoints(run_id)}

    @app.post("/api/runs/{run_id}/resume", dependencies=[require_auth])
    async def resume_run(run_id: int, request: ResumeRequest):
        try:
            return jobs.resume_run(run_id, request.checkpoint_path)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.websocket("/api/runs/{run_id}/events")
    async def run_events(websocket: WebSocket, run_id: int, token: str | None = None):
        try:
            auth.verify_token_value(token)
        except HTTPException:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        subscribe = getattr(jobs, "subscribe", None)
        event_queue = subscribe(run_id) if subscribe else jobs.get_event_queue(run_id)
        try:
            while True:
                try:
                    event = await asyncio.to_thread(event_queue.get, True, 1)
                    await websocket.send_text(json.dumps(event))
                except queue.Empty:
                    await websocket.send_text(json.dumps({"run_id": run_id, "type": "heartbeat"}))
        except WebSocketDisconnect:
            return
        finally:
            unsubscribe = getattr(jobs, "unsubscribe", None)
            if unsubscribe:
                unsubscribe(run_id, event_queue)

    # ------------------------------------------------------------------
    # Trained artifacts
    # ------------------------------------------------------------------

    @app.get("/api/models", dependencies=[require_auth])
    async def list_models():
        return {"models": ledger.list_models()}

    @app.get("/api/models/{model_id}", dependencies=[require_auth])
    async def get_model(model_id: int):
        model = ledger.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found.")
        return model

    @app.delete("/api/models/{model_id}", dependencies=[require_auth])
    async def delete_model(model_id: int, delete_files: bool = False):
        model = ledger.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found.")
        if delete_files:
            target = Path(model["model_path"])
            # Only ever remove a directory this ledger recorded, never a path
            # supplied by the caller.
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
        ledger.delete_model(model_id)
        return {"deleted": True, "files_removed": delete_files}

    @app.post("/api/models/validate", dependencies=[require_auth])
    async def validate_model(request: ModelValidationRequest):
        return workflow.validate_model_id(request.model_id, request.task_slug)

    @app.post("/api/models/{model_id}/push", dependencies=[require_auth])
    async def push_model(model_id: int, request: HubPushRequest):
        model = ledger.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found.")
        folder = Path(model["model_path"])
        if not folder.is_dir():
            raise HTTPException(
                status_code=409,
                detail=f"The files for this model are no longer at {folder}.",
            )
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo_id=request.repo_id, repo_type="model",
                            private=request.private, exist_ok=True)
            api.upload_folder(folder_path=str(folder), repo_id=request.repo_id, repo_type="model")
        except Exception as exc:  # noqa: BLE001 - the Hub error is the useful message
            raise HTTPException(status_code=502, detail=f"Upload failed: {exc}") from exc
        ledger.mark_model_pushed(model_id, request.repo_id)
        return {
            "pushed": True,
            "repo_id": request.repo_id,
            "url": f"https://huggingface.co/{request.repo_id}",
        }

    # ------------------------------------------------------------------
    # Playground
    # ------------------------------------------------------------------

    @app.get("/api/playground/status", dependencies=[require_auth])
    async def playground_status():
        available, reason = playground.availability()
        current = playground.current
        return {
            "available": available,
            "reason": reason,
            "session_id": current.id if current else None,
            "model_path": current.model_path if current else None,
        }

    @app.post("/api/playground/sessions", dependencies=[require_auth])
    async def open_playground(request: PlaygroundOpenRequest):
        try:
            session = await asyncio.to_thread(playground.open, request.model_path)
        except PlaygroundUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "session_id": session.id,
            "model_path": session.model_path,
            "base_model": session.base_model,
            "note": getattr(session, "device_note", None),
        }

    @app.delete("/api/playground/sessions/{session_id}", dependencies=[require_auth])
    async def close_playground(session_id: str):
        return {"closed": playground.close(session_id)}

    @app.websocket("/api/playground/{session_id}/chat")
    async def playground_chat(websocket: WebSocket, session_id: str, token: str | None = None):
        try:
            auth.verify_token_value(token)
        except HTTPException:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        session = playground.get(session_id)
        if session is None:
            await websocket.send_text(json.dumps({"type": "error", "message": "That playground session has closed."}))
            await websocket.close()
            return

        try:
            while True:
                raw = await websocket.receive_text()
                request = json.loads(raw)
                if request.get("type") == "stop":
                    session.stop()
                    continue

                messages = request.get("messages") or []
                sampling = {
                    "max_new_tokens": int(request.get("max_new_tokens", 256)),
                    "temperature": float(request.get("temperature", 0.7)),
                    "top_p": float(request.get("top_p", 0.9)),
                }
                await websocket.send_text(json.dumps({"type": "start"}))
                try:
                    chunks: queue.Queue = queue.Queue()
                    sentinel = object()

                    def produce():
                        try:
                            for piece in session.generate(messages, **sampling):
                                chunks.put(piece)
                        except Exception as exc:  # noqa: BLE001
                            chunks.put(RuntimeError(str(exc)))
                        finally:
                            chunks.put(sentinel)

                    threading.Thread(target=produce, daemon=True).start()
                    while True:
                        piece = await asyncio.to_thread(chunks.get)
                        if piece is sentinel:
                            break
                        if isinstance(piece, Exception):
                            raise piece
                        await websocket.send_text(json.dumps({"type": "token", "text": piece}))
                    await websocket.send_text(json.dumps({"type": "done"}))
                except Exception as exc:  # noqa: BLE001
                    await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except WebSocketDisconnect:
            return

    dist = frontend_dist_path()
    assets = dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    async def root():
        return FileResponse(dist / "index.html")

    @app.get("/app")
    async def app_index():
        return FileResponse(dist / "index.html")

    @app.get("/{path:path}")
    async def spa_fallback(path: str):
        target = dist / path
        if target.is_file():
            return FileResponse(target)
        return FileResponse(dist / "index.html")

    return app
