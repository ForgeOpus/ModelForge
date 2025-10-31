import os
import uvicorn

## Globals imports
from .globals.globals_instance import global_manager

## FastAPI imports
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .routers.finetuning_router import router as finetuning_router
from .routers.playground_router import router as playground_router
from .routers.hub_management_router import router as hub_management_router
from .routers.models_router import router as models_router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv


load_dotenv()

app = FastAPI()
## Static files
frontend_dir = os.path.join(os.path.dirname(__file__), "./Frontend/build")
app_name = "ModelForge"

# CORS origins - configurable via environment variable
cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:8000")
origins = [origin.strip() for origin in cors_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(prefix="/api", router=finetuning_router)
app.include_router(prefix="/api", router=playground_router)
app.include_router(prefix="/api", router=models_router)
app.include_router(prefix="/api", router=hub_management_router)

## Mount static files
app.mount(
    "/",
    StaticFiles(directory=frontend_dir, html=True),
    name="static"
)

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    index_file = os.path.join(frontend_dir,"index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    else:
        return JSONResponse({"detail": "index.html not found"}, status_code=404)

## Server endpoints
@app.get("/")
async def home(request: Request) -> JSONResponse:
    return JSONResponse({
        "app_name": app_name,
        "app_description": "No-code LLM finetuning for CUDA environments",
        "features": [
            "Intuitive no-code interface",
            "PEFT and LoRA-based finetuning",
            "4-bit/8-bit quantization",
            "GPU-accelerated performance"
        ]
    })

def main():
    ## Server Global Configurations
    print(f"global dirs: {global_manager.app_name}")
    uvicorn.run(app, host='127.0.0.1', port=8000)

if __name__ == "__main__":
    main()