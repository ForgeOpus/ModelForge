import os
import subprocess

from fastapi import APIRouter
from fastapi import Request
from starlette.responses import JSONResponse

from ..globals.globals_instance import global_manager

from pydantic import BaseModel, field_validator

router = APIRouter(
    prefix="/playground",
)

class PlaygroundRequest(BaseModel):
    model_path: str
    
    @field_validator("model_path")
    def validate_model_path(cls, model_path):
        if not model_path or not model_path.strip():
            raise ValueError("Model path cannot be empty.")
        return model_path.strip()

@router.get("/model_path")
async def get_model_path(request: Request) -> JSONResponse:
    return JSONResponse({
        "app_name": global_manager.app_name,
        "model_path": global_manager.model_path
    })


@router.post("/new")
async def new_playground(request: Request) -> None:
    form = await request.json()
    print(form)
    playground_request = PlaygroundRequest(model_path=form["model_path"])
    model_path = playground_request.model_path

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utilities"))
    chat_script = os.path.join(base_path, "chat_playground.py")
    if os.name == "nt":  # Windows
        # Note: shell=True is required for 'start' command (cmd.exe built-in)
        # Input is validated via PlaygroundRequest Pydantic model
        command = ["cmd.exe", "/c", "start", "python", chat_script, "--model_path", model_path]
        subprocess.Popen(command, shell=True)
    else:  # Unix/Linux/Mac
        # Use list format without string interpolation for security
        command = ["x-terminal-emulator", "-e", "python", chat_script, "--model_path", model_path]
        try:
            subprocess.Popen(command)
        except FileNotFoundError:
            # Fallback to gnome-terminal or xterm if x-terminal-emulator is not available
            try:
                subprocess.Popen(["gnome-terminal", "--", "python3", chat_script, "--model_path", model_path])
            except FileNotFoundError:
                subprocess.Popen(["xterm", "-e", "python3", chat_script, "--model_path", model_path])
