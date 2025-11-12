"""
Refactored models router.
Handles model listing and retrieval operations.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict

from ..services.model_service import ModelService
from ..dependencies import get_model_service
from ..logging_config import logger


router = APIRouter(prefix="/models")


@router.get("/", response_model=List[Dict])
async def get_all_models(
    model_service: ModelService = Depends(get_model_service),
):
    """
    Get all fine-tuned models.

    Args:
        model_service: Model service instance

    Returns:
        List of all models
    """
    logger.info("Fetching all models")
    try:
        models = model_service.get_all_models()
        return models
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}")
async def get_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """
    Get a specific model by ID.

    Args:
        model_id: Model identifier
        model_service: Model service instance

    Returns:
        Model details

    Raises:
        HTTPException: If model not found
    """
    logger.info(f"Fetching model: {model_id}")
    try:
        model = model_service.get_model_by_id(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task}")
async def get_models_by_task(
    task: str,
    model_service: ModelService = Depends(get_model_service),
):
    """
    Get all models for a specific task.

    Args:
        task: Task type
        model_service: Model service instance

    Returns:
        List of models for the task
    """
    logger.info(f"Fetching models for task: {task}")
    try:
        models = model_service.get_models_by_task(task)
        return models
    except Exception as e:
        logger.error(f"Error fetching models by task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """
    Delete a model.

    Args:
        model_id: Model identifier
        model_service: Model service instance

    Returns:
        Deletion confirmation

    Raises:
        HTTPException: If model not found or deletion fails
    """
    logger.info(f"Deleting model: {model_id}")
    try:
        success = model_service.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")

        return {"success": True, "message": "Model deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
