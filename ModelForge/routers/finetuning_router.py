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
    get_session_data,
    update_session_data,
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


@router.post("/detect")
async def detect_hardware(
    data: TaskSelection,
    hardware_service: HardwareService = Depends(get_hardware_service),
):
    """
    Detect hardware and get model recommendations for a task.

    This endpoint combines hardware detection with model recommendations,
    providing a complete profile for the frontend.

    Args:
        data: Task selection data
        hardware_service: Hardware service instance

    Returns:
        Combined hardware specs and model recommendations

    Raises:
        HTTPException: If detection fails
    """
    logger.info(f"Detecting hardware for task: {data.task}")

    try:
        # Store task in session for later use
        update_session_data("task", data.task)

        # Get hardware specifications
        hardware_specs = hardware_service.get_hardware_specs()

        # Get compute profile
        compute_profile = hardware_service.get_compute_profile()

        # Get model recommendations for the task
        recommendations = hardware_service.get_recommended_models(data.task)

        # Extract the recommended model and possible options
        model_recommendation = recommendations.get("recommended_model", "")
        possible_options = recommendations.get("possible_models", [])

        # Build response matching frontend expectations
        response = {
            "status_code": 200,
            "profile": compute_profile,
            "task": data.task,
            "gpu_name": hardware_specs.get("gpu_name", "Unknown"),
            "gpu_total_memory_gb": hardware_specs.get("gpu_memory_gb", 0),
            "ram_total_gb": hardware_specs.get("ram_gb", 0),
            "available_diskspace_gb": hardware_specs.get("disk_space_gb", 0),
            "cpu_cores": hardware_specs.get("cpu_cores", 0),
            "model_recommendation": model_recommendation,
            "possible_options": possible_options,
        }

        logger.info(f"Hardware detection complete: {compute_profile} profile")
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error detecting hardware: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set_model")
async def set_model(data: ModelSelection):
    """
    Set the selected model for training.

    This endpoint stores the user's model selection in the session cache
    for use in subsequent training configuration steps.

    Args:
        data: Model selection data

    Returns:
        Success confirmation with selected model
    """
    logger.info(f"Setting selected model: {data.selected_model}")

    try:
        # Store model selection in session
        update_session_data("selected_model", data.selected_model)

        return {
            "success": True,
            "selected_model": data.selected_model,
            "message": "Model selection saved successfully",
        }

    except Exception as e:
        logger.error(f"Error setting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set_custom_model")
async def set_custom_model(
    data: ModelValidation,
    model_service: ModelService = Depends(get_model_service),
):
    """
    Set a custom model for training.

    Validates the custom model repository and stores it in the session cache.

    Args:
        data: Model validation data with repo_name
        model_service: Model service instance

    Returns:
        Validation result and success confirmation

    Raises:
        HTTPException: If model validation fails
    """
    logger.info(f"Setting custom model: {data.repo_name}")

    try:
        # Validate the custom model
        result = model_service.validate_model_access(
            repo_name=data.repo_name,
            model_class="AutoModelForCausalLM",
        )

        if result.get("valid", False):
            # Store custom model in session
            update_session_data("selected_model", data.repo_name)
            update_session_data("is_custom_model", True)

            return {
                "success": True,
                "selected_model": data.repo_name,
                "message": "Custom model validated and saved successfully",
                **result,
            }
        else:
            raise HTTPException(status_code=400, detail="Model validation failed")

    except ModelAccessError as e:
        logger.error(f"Model access error: {e}")
        raise HTTPException(status_code=403, detail=str(e))

    except Exception as e:
        logger.error(f"Error setting custom model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session")
async def get_session():
    """
    Get current session data.

    Returns session information including selected task, model, and other
    selections made during the workflow.

    Returns:
        Session data dictionary
    """
    logger.info("Retrieving session data")

    return {
        "success": True,
        "task": get_session_data("task"),
        "selected_model": get_session_data("selected_model"),
        "is_custom_model": get_session_data("is_custom_model") or False,
    }


@router.get("/load_settings")
async def get_default_settings(
    hardware_service: HardwareService = Depends(get_hardware_service),
):
    """
    Get default training settings based on hardware profile.

    Returns hardware-appropriate default training configuration including
    batch size, learning rate, sequence length, and other training parameters.

    Args:
        hardware_service: Hardware service instance

    Returns:
        Default training settings dictionary

    Raises:
        HTTPException: If settings generation fails
    """
    logger.info("Getting default training settings")

    try:
        # Get hardware profile
        compute_profile = hardware_service.get_compute_profile()

        # Get session data
        selected_model = get_session_data("selected_model")
        selected_task = get_session_data("task")

        # Define hardware-specific defaults
        settings_by_profile = {
            "low_end": {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "num_train_epochs": 3,
                "learning_rate": 2e-4,
                "max_seq_length": 512,
                "warmup_ratio": 0.1,
                "logging_steps": 10,
                "save_strategy": "epoch",
                "optim": "adamw_torch",
                "gradient_checkpointing": True,
                "fp16": True,
            },
            "mid_range": {
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "num_train_epochs": 3,
                "learning_rate": 2e-4,
                "max_seq_length": 1024,
                "warmup_ratio": 0.1,
                "logging_steps": 10,
                "save_strategy": "epoch",
                "optim": "adamw_torch",
                "gradient_checkpointing": True,
                "fp16": True,
            },
            "high_end": {
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "num_train_epochs": 3,
                "learning_rate": 2e-4,
                "max_seq_length": 2048,
                "warmup_ratio": 0.1,
                "logging_steps": 10,
                "save_strategy": "epoch",
                "optim": "adamw_torch",
                "gradient_checkpointing": True,
                "fp16": True,
            },
        }

        # Get settings for current profile (default to mid_range if not found)
        settings = settings_by_profile.get(compute_profile, settings_by_profile["mid_range"])

        # Add context information
        response = {
            "success": True,
            "compute_profile": compute_profile,
            "selected_model": selected_model,
            "selected_task": selected_task,
            "default_values": settings,
        }

        logger.info(f"Returning default settings for {compute_profile} profile")
        return response

    except Exception as e:
        logger.error(f"Error getting default settings: {e}")
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
