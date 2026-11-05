from __future__ import annotations

import asyncio
import json
import queue
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utilities.database_manager import WebRunLedger
from web.auth import TokenAuth
from web.playground import InferenceRequest, Playground, PlaygroundError
from workflow.jobs import TrainingJobManager
from workflow.models import DatasetValidationRequest, RecommendationRequest, TrainingRunRequest
from workflow.service import WorkflowService


class ResumeRequest(BaseModel):
    checkpoint_path: str


def frontend_dist_path() -> Path:
    return Path(__file__).resolve().parent / "frontend_dist"


def create_app(auth: TokenAuth | None = None, job_manager: TrainingJobManager | None = None) -> FastAPI:
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
    app.state.auth = auth
    app.state.workflow = workflow
    app.state.jobs = jobs
    app.state.ledger = ledger
    playground = Playground(jobs, ledger)
    app.state.playground = playground

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    require_auth = Depends(auth.dependency)

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
        return {"runs": ledger.list_runs(limit=limit)}

    @app.post("/api/playground/infer", dependencies=[require_auth])
    def playground_infer(request: InferenceRequest):
        try:
            return playground.infer(request)
        except PlaygroundError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.message) from exc

    @app.get("/api/runs/{run_id}", dependencies=[require_auth])
    async def get_run(run_id: int):
        run = ledger.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return run

    @app.post("/api/runs/{run_id}/cancel", dependencies=[require_auth])
    async def cancel_run(run_id: int):
        try:
            return jobs.cancel_run(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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
        event_queue = jobs.get_event_queue(run_id)
        try:
            while True:
                try:
                    event = await asyncio.to_thread(event_queue.get, True, 1)
                    await websocket.send_text(json.dumps(event))
                except queue.Empty:
                    await websocket.send_text(json.dumps({"run_id": run_id, "type": "heartbeat"}))
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
