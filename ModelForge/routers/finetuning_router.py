"""
Refactored fine-tuning router.
Slim router that delegates to services for business logic.
"""
import os
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from starlette.responses import JSONResponse

from ..schemas.training_schemas import (
    TrainingConfig,
    TaskSelection,
    ModelSelection,
    ModelValidation,
    TrainingStatus,
    TrainingResult,
)
from ..services.training_service import TrainingService
from ..services.model_service import ModelService
from ..services.hardware_service import HardwareService
from ..utilities.settings_managers.FileManager import FileManager
from ..dependencies import (
    get_training_service,
    get_model_service,
    get_hardware_service,
    get_file_manager,
)
from ..exceptions import (
    ModelAccessError,
    DatasetValidationError,
    TrainingError,
    ConfigurationError,
)
from ..logging_config import logger


router = APIRouter(prefix="/finetune")


@router.post("/validate_task")
async def validate_task(data: TaskSelection):
    """
    Validate task selection.

    Args:
        data: Task selection data

    Returns:
        Validation result
    """
    logger.info(f"Validating task: {data.task}")
    return {"valid": True, "task": data.task}


@router.post("/validate_model")
async def validate_model(
    data: ModelSelection,
    model_service: ModelService = Depends(get_model_service),
):
    """
    Validate model selection.

    Args:
        data: Model selection data
        model_service: Model service instance

    Returns:
        Validation result
    """
    logger.info(f"Validating model: {data.selected_model}")
    return {"valid": True, "model": data.selected_model}


@router.post("/validate_custom_model")
async def validate_custom_model(
    data: ModelValidation,
    model_service: ModelService = Depends(get_model_service),
):
    """
    Validate custom model repository.

    Args:
        data: Model validation data
        model_service: Model service instance

    Returns:
        Validation result

    Raises:
        HTTPException: If model is not accessible
    """
    logger.info(f"Validating custom model: {data.repo_name}")

    try:
        result = model_service.validate_model_access(
            repo_name=data.repo_name,
            model_class="AutoModelForCausalLM",
        )
        return result

    except ModelAccessError as e:
        logger.error(f"Model access error: {e}")
        raise HTTPException(status_code=403, detail=str(e))

    except Exception as e:
        logger.error(f"Model validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load_settings")
async def load_settings(
    json_file: UploadFile = File(...),
    settings: str = Form(...),
    file_manager: FileManager = Depends(get_file_manager),
):
    """
    Upload dataset file.

    Args:
        json_file: Dataset file (JSON/JSONL)
        settings: JSON string with settings
        file_manager: File manager instance

    Returns:
        Upload result with file path

    Raises:
        HTTPException: If file upload fails
    """
    logger.info(f"Uploading dataset: {json_file.filename}")

    try:
        # Validate file type
        if not json_file.filename.endswith(('.json', '.jsonl')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JSON and JSONL files are allowed."
            )

        # Read file content
        file_content = await json_file.read()

        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{json_file.filename}"

        # Save file
        default_dirs = file_manager.return_default_dirs()
        file_path = os.path.join(default_dirs["datasets"], filename)
        file_manager.save_file(file_path, file_content)

        logger.info(f"Dataset uploaded successfully: {file_path}")

        return {
            "success": True,
            "file_path": file_path,
            "filename": filename,
            "message": "Dataset uploaded successfully",
        }

    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start_training")
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
    training_service: TrainingService = Depends(get_training_service),
    hardware_service: HardwareService = Depends(get_hardware_service),
):
    """
    Start model training.

    Args:
        config: Training configuration
        background_tasks: FastAPI background tasks
        training_service: Training service instance
        hardware_service: Hardware service instance

    Returns:
        Training start confirmation

    Raises:
        HTTPException: If validation fails or training cannot start
    """
    logger.info(f"Starting training: {config.task} with {config.strategy}")

    try:
        # Validate dataset
        dataset_info = training_service.validate_and_prepare_dataset(
            dataset_path=config.dataset,
            task=config.task,
            strategy=config.strategy,
        )

        logger.info(f"Dataset validated: {dataset_info['num_examples']} examples")

        # Validate batch size for hardware
        compute_profile = hardware_service.get_compute_profile()
        if not hardware_service.validate_batch_size(
            config.per_device_train_batch_size,
            compute_profile
        ):
            logger.warning(
                f"Batch size {config.per_device_train_batch_size} may be too large "
                f"for {compute_profile} hardware"
            )

        # Convert config to dict
        config_dict = config.model_dump()

        # Start training in background
        background_tasks.add_task(training_service.train_model, config_dict)

        return {
            "success": True,
            "message": "Training started successfully",
            "dataset_info": dataset_info,
        }

    except DatasetValidationError as e:
        logger.error(f"Dataset validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status(
    training_service: TrainingService = Depends(get_training_service),
):
    """
    Get training status.

    Args:
        training_service: Training service instance

    Returns:
        Training status
    """
    status = training_service.get_training_status()
    return status


@router.post("/reset_status")
async def reset_status(
    training_service: TrainingService = Depends(get_training_service),
):
    """
    Reset training status to idle.

    Args:
        training_service: Training service instance

    Returns:
        Success confirmation
    """
    training_service.reset_training_status()
    logger.info("Training status reset")
    return {"success": True, "message": "Status reset"}


@router.get("/hardware_specs")
async def get_hardware_specs(
    hardware_service: HardwareService = Depends(get_hardware_service),
):
    """
    Get hardware specifications.

    Args:
        hardware_service: Hardware service instance

    Returns:
        Hardware specifications
    """
    specs = hardware_service.get_hardware_specs()
    return specs


@router.get("/recommended_models/{task}")
async def get_recommended_models(
    task: str,
    hardware_service: HardwareService = Depends(get_hardware_service),
):
    """
    Get recommended models for a task.

    Args:
        task: Task type
        hardware_service: Hardware service instance

    Returns:
        Model recommendations

    Raises:
        HTTPException: If task is invalid
    """
    try:
        recommendations = hardware_service.get_recommended_models(task)
        return recommendations

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
