from fastapi import APIRouter, HTTPException
from fastapi import Request
from starlette.responses import JSONResponse
from ..globals.globals_instance import global_manager

router = APIRouter(
    prefix="/models",
)

@router.get("/", response_class=JSONResponse)
async def list_models(request: Request) -> JSONResponse:
    try:
        models = global_manager.db_manager.get_all_models()
        return JSONResponse({"models": models})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error fetching models.")

@router.get("/{model_id}", response_class=JSONResponse)
async def get_model(model_id: int, request: Request) -> JSONResponse:
    try:
        model = global_manager.db_manager.get_model_by_id(model_id)  # Assumes this method exists
        if not model:
            raise HTTPException(status_code=404, detail="Model not found.")
        return JSONResponse(model)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error fetching model.")
