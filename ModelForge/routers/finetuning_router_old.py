from datetime import datetime
import json
import os
import shutil
import traceback
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi import Request
from starlette.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

from ..utilities.finetuning.CausalLLMTuner import CausalLLMFinetuner
from ..utilities.finetuning.QuestionAnsweringTuner import QuestionAnsweringTuner
from ..utilities.finetuning.Seq2SeqLMTuner import Seq2SeqFinetuner
from ..utilities.hardware_detection.model_validator import ModelValidator

from ..globals.globals_instance import global_manager

router = APIRouter(
    prefix="/finetune",
)

# Valid task types
VALID_TASKS = ["text-generation", "summarization", "extractive-question-answering"]
VALID_TASKS_STR = "'text-generation', 'summarization', or 'extractive-question-answering'"

## Pydantic Data Validator Classes
class TaskFormData(BaseModel):
    task: str
    @field_validator("task")
    def validate_task(cls, task):
        if task not in VALID_TASKS:
            raise ValueError(f"Invalid task. Must be one of {VALID_TASKS_STR}.")
        return task

class SelectedModelFormData(BaseModel):
    selected_model: str
    @field_validator("selected_model")
    def validate_selected_model(cls, selected_model):
        if not selected_model:
            raise ValueError("Selected model cannot be empty.")
        return selected_model

class CustomModelValidationData(BaseModel):
    repo_name: str
    @field_validator("repo_name")
    def validate_repo_name(cls, repo_name):
        if not repo_name or not repo_name.strip():
            raise ValueError("Repository name cannot be empty.")
        return repo_name.strip()

class SettingsFormData(BaseModel):
    task: str
    model_name: str
    num_train_epochs: int
    compute_specs: str
    lora_r: int
    lora_alpha: int
    lora_dropout: int
    use_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_quant_type: str
    use_nested_quant: bool
    bnb_4bit_quant_type: str
    fp16: bool
    bf16: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_grad_norm: float
    learning_rate: float
    weight_decay: float
    optim: str
    lr_scheduler_type: str
    max_steps: int
    warmup_ratio: float
    group_by_length: bool
    packing: bool
    max_seq_length: int
    dataset: str

    @field_validator("dataset")
    def validate_dataset_prescence(cls, dataset):
        if not dataset:
            raise ValueError("Dataset cannot be empty.")
        return dataset
    @field_validator("task")
    def validate_task(cls, task):
        if task not in VALID_TASKS:
            raise ValueError(f"Invalid task. Must be one of {VALID_TASKS_STR}.")
        return task
    @field_validator("model_name")
    def validate_model_name(cls, model_name):
        if not model_name:
            raise ValueError("Model name cannot be empty.")
        return model_name
    @field_validator("num_train_epochs")
    def validate_num_train_epochs(cls, num_train_epochs):
        if num_train_epochs <= 0:
            raise ValueError("Number of training epochs must be greater than 0.")
        if num_train_epochs >= 50:
            raise ValueError("Number of training epochs must be less than 50.")
        return num_train_epochs
    @field_validator("compute_specs")
    def validate_compute_specs(cls, compute_specs):
        if compute_specs not in ["low_end", "mid_range", "high_end"]:
            raise ValueError("Invalid compute specs. Must be one of 'low_end', 'mid_range', or 'high_end'.")
        return compute_specs
    @field_validator("lora_r")
    def validate_lora_r(cls, lora_r):
        if lora_r not in [8, 16, 32, 64]:
            raise ValueError("LoRA rank must be 8, 16, 32, or 64.")
        return lora_r
    @field_validator("lora_alpha")
    def validate_lora_alpha(cls, lora_alpha):
        if lora_alpha >= 0.5:
            raise ValueError("LoRA learning rate is too high. Gradients will explode.")
        return lora_alpha
    @field_validator("lora_dropout")
    def validate_lora_dropout(cls, lora_dropout):
        if lora_dropout < 0.0 or lora_dropout > 1.0:
            raise ValueError("LoRA dropout probability must be between 0.0 and 1.0.")
        return lora_dropout
    @field_validator("use_4bit")
    def validate_use_4bit(cls, use_4bit):
        if not isinstance(use_4bit, bool):
            raise ValueError("use_4bit must be a boolean value.")
        return use_4bit
    @field_validator("bnb_4bit_compute_dtype")
    def validate_bnb_4bit_compute_dtype(cls, bnb_4bit_compute_dtype):
        if bnb_4bit_compute_dtype not in ["float16", "bfloat16"]:
            raise ValueError("Invalid compute dtype. Must be 'float16' or 'bfloat16'.")
        return bnb_4bit_compute_dtype
    @field_validator("bnb_4bit_use_quant_type")
    def validate_bnb_4bit_use_quant_type(cls, bnb_4bit_use_quant_type):
        if bnb_4bit_use_quant_type not in ["fp4", "int8"]:
            raise ValueError("Invalid quantization type. Must be 'fp4' or 'int8'.")
        return bnb_4bit_use_quant_type
    @field_validator("use_nested_quant")
    def validate_use_nested_quant(cls, use_nested_quant):
        if not isinstance(use_nested_quant, bool):
            raise ValueError("use_nested_quant must be true or false.")
        return use_nested_quant
    @field_validator("bnb_4bit_quant_type")
    def validate_bnb_4bit_quant_type(cls, bnb_4bit_quant_type):
        if bnb_4bit_quant_type not in ["fp4", "int8"]:
            raise ValueError("Invalid quantization type. Must be 'fp4' or 'int8'.")
        return bnb_4bit_quant_type
    @field_validator("fp16")
    def validate_fp16(cls, fp16):
        if not isinstance(fp16, bool):
            raise ValueError("fp16 must be true or false.")
        return fp16
    @field_validator("bf16")
    def validate_bf16(cls, bf16):
        if not isinstance(bf16, bool):
            raise ValueError("bf16 must be true or false.")
        return bf16
    @field_validator("per_device_train_batch_size")
    def validate_per_device_train_batch_size(cls, per_device_train_batch_size):
        if per_device_train_batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        return per_device_train_batch_size
    @field_validator("per_device_eval_batch_size")
    def validate_per_device_eval_batch_size(cls, per_device_eval_batch_size):
        if per_device_eval_batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        return per_device_eval_batch_size
    @field_validator("gradient_accumulation_steps")
    def validate_gradient_accumulation_steps(cls, gradient_accumulation_steps):
        if gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be greater than 0.")
        return gradient_accumulation_steps
    @field_validator("gradient_checkpointing")
    def validate_gradient_checkpointing(cls, gradient_checkpointing):
        if not isinstance(gradient_checkpointing, bool):
            raise ValueError("Gradient checkpointing must be true or false.")
        return gradient_checkpointing
    @field_validator("max_grad_norm")
    def validate_max_grad_norm(cls, max_grad_norm):
        if max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be greater than 0.")
        return max_grad_norm
    @field_validator("learning_rate")
    def validate_learning_rate(cls, learning_rate):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")
        elif learning_rate > 0.3:
            raise ValueError("Learning rate must be less than 0.3. Higher learning rates cause exploding gradients.")
        return learning_rate
    @field_validator("weight_decay")
    def validate_weight_decay(cls, weight_decay):
        if weight_decay < 0:
            raise ValueError("Weight decay must be greater than or equal to 0.")
        return weight_decay
    @field_validator("optim")
    def validate_optim(cls, optim):
        if optim not in ["paged_adamw_32bit", "paged_adamw_8bit", "adamw_torch", "adamw_hf"]:
            raise ValueError("Invalid optimizer. Must be one of 'paged_adamw_32bit', 'paged_adamw_8bit', 'adamw_torch', or 'adamw_hf'.")
        return optim
    @field_validator("lr_scheduler_type")
    def validate_lr_scheduler_type(cls, lr_scheduler_type):
        if lr_scheduler_type not in ["linear", "cosine", "constant", "constant_with_warmup"]:
            raise ValueError("Invalid learning rate scheduler type. Must be one of 'linear', 'cosine', 'constant', or 'constant_with_warmup'.")
        return lr_scheduler_type
    @field_validator("max_steps")
    def validate_max_steps(cls, max_steps):
        if max_steps < 0:
            raise ValueError("Max steps must be greater than or equal to 0.")
        elif max_steps > 100:
            raise ValueError("Max steps must be less than 100.")
        return max_steps
    @field_validator("warmup_ratio")
    def validate_warmup_ratio(cls, warmup_ratio):
        if warmup_ratio < 0.0 or warmup_ratio > 1.0:
            raise ValueError("Warmup ratio must be between 0.0 and 1.0.")
        return warmup_ratio
    @field_validator("group_by_length")
    def validate_group_by_length(cls, group_by_length):
        if not isinstance(group_by_length, bool):
            raise ValueError("Group by length must be true or false.")
        return group_by_length
    @field_validator("packing")
    def validate_packing(cls, packing):
        if not isinstance(packing, bool):
            raise ValueError("Packing must be true or false.")
        return packing
    @field_validator("max_seq_length")
    def validate_max_seq_length(cls, max_seq_length):
        if max_seq_length < -1:
            raise ValueError("Max sequence length must be greater than 0 or it should be -1 if you want to use the default Max Sequence Length.")
        return max_seq_length
    @field_validator("dataset")
    def validate_dataset(cls, dataset):
        if not dataset:
            raise ValueError("Dataset cannot be empty.")
        return dataset
    
    @model_validator(mode='after')
    def validate_batch_size_with_compute_specs(self):
        """Validate batch size based on compute specs"""
        if self.per_device_train_batch_size > 3 and self.compute_specs != "high_end":
            raise ValueError("Batch size must be 3 or less. Your device cannot support a higher batch size.")
        elif self.per_device_train_batch_size > 8 and self.compute_specs == "high_end":
            raise ValueError("Batch size must be 8 or less. Higher batch sizes cause out of memory error.")
        return self


@router.get("/detect")
async def detect_hardware_page(request: Request) -> JSONResponse:
    global_manager.clear_settings_cache()  # Clear the cache to ensure fresh detection
    return JSONResponse({
        "app_name": global_manager.app_name,
        "message": "Ready to detect hardware"
    })

@router.post("/detect", response_class=JSONResponse)
async def detect_hardware(request: Request) -> JSONResponse:
    try:
        form = await request.json()
        print(form)
        task = TaskFormData(task=form["task"])
        task = task.task
        global_manager.settings_builder.task = task
        # Reset custom model settings when running new hardware detection
        global_manager.settings_builder.is_custom_model = False
        model_requirements, hardware_profile, model_recommendation, possible_options = global_manager.hardware_detector.run(task)
        global_manager.settings_builder.compute_profile = model_requirements["profile"]
        global_manager.settings_cache.update({
            "model_requirements": model_requirements,
            "hardware_profile": hardware_profile,
            "model_recommendation": model_recommendation,
            "selected_model": None,
            "is_custom_model": False
        })
        return JSONResponse(
            {
                "status_code": 200,
                "profile": model_requirements["profile"],
                "task": task,
                "gpu_name": hardware_profile.get("gpu_name"),
                "gpu_total_memory_gb": hardware_profile.get("gpu_total_memory_gb"),
                "ram_total_gb": hardware_profile.get("ram_total_gb"),
                "available_diskspace_gb": hardware_profile.get("available_diskspace_gb"),
                "cpu_cores": hardware_profile.get("cpu_cores"),
                "model_recommendation": model_recommendation,
                "possible_options": possible_options,

            }
        )
    except RuntimeError as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except ValueError as e:
        print(e)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task. Must be one of {VALID_TASKS_STR}."
        )
    except Exception as e:
        print("General exception triggered")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Error detecting hardware. Please try again later."
        )

@router.post("/set_model")
async def set_model(request: Request) -> None:
    try:
        form = await request.json()
        selected_model = SelectedModelFormData(selected_model=form["selected_model"])
        global_manager.settings_cache["selected_model"] = selected_model
        global_manager.settings_cache["is_custom_model"] = False
        global_manager.settings_builder.set_recommended_model(selected_model.selected_model)
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail="Error setting model. Please try again later."
        )

@router.post("/validate_custom_model", response_class=JSONResponse)
async def validate_custom_model(request: Request) -> JSONResponse:
    """
    Validate a custom model from HuggingFace Hub.
    
    Note: Currently validates repository existence but not task compatibility.
    Consider adding architecture-task compatibility checks (e.g., ensure
    summarization models aren't used for text generation tasks) for better
    user experience and to prevent fine-tuning failures.
    """
    try:
        form = await request.json()
        validation_data = CustomModelValidationData(repo_name=form["repo_name"])
        
        # Create model validator and validate the repository
        model_validator = ModelValidator()
        validation_result = model_validator.validate_repo_name(validation_data.repo_name)
        
        # Add default warnings for custom models
        if validation_result["valid"]:
            validation_result["warnings"] = model_validator.get_default_warnings()
        
        return JSONResponse({
            "status_code": 200,
            "validation": validation_result
        })
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        print(f"Custom model validation error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error validating custom model. Please try again later."
        )

@router.post("/set_custom_model", response_class=JSONResponse)
async def set_custom_model(request: Request) -> JSONResponse:
    try:
        form = await request.json()
        validation_data = CustomModelValidationData(repo_name=form["repo_name"])
        
        # Validate the custom model first
        model_validator = ModelValidator()
        validation_result = model_validator.validate_repo_name(validation_data.repo_name)
        
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid custom model: {validation_result.get('error', 'Unknown error')}"
            )
        
        # Set the custom model in global settings
        global_manager.settings_cache["selected_model"] = validation_data
        global_manager.settings_cache["is_custom_model"] = True
        global_manager.settings_builder.set_custom_model(validation_data.repo_name)
        
        return JSONResponse({
            "status_code": 200,
            "message": "Custom model set successfully",
            "model_name": validation_data.repo_name,
            "warnings": model_validator.get_default_warnings()
        })
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error setting custom model: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error setting custom model. Please try again later."
        )

@router.get("/load_settings")
async def load_settings_page(request: Request) -> JSONResponse:
    if not global_manager.settings_cache:
        raise HTTPException(
            status_code=400,
            detail="No hardware detection data available. Please run hardware detection first."
        )
    return JSONResponse({
        "default_values": global_manager.settings_builder.get_settings()
    })


def gen_uuid(filename) -> str:
    extension = filename.split(".")[-1]
    return str(uuid.uuid4()).replace("-", "") + "." + extension


@router.post("/load_settings")
async def load_settings(json_file: UploadFile = File(...), settings: str = Form(...)) -> None:
    print("Loading settings...")
    # Validate file type
    if json_file.content_type != "application/json" and json_file.content_type != "application/x-jsonlines" and not json_file.filename.endswith(('.json', '.jsonl')):
        raise HTTPException(400, "Only JSON and JSONL files accepted")

    json_filename = gen_uuid(json_file.filename)
    file_content = await json_file.read()
    file_path = os.path.join(global_manager.file_manager.return_default_dirs()["datasets"], json_filename)
    file_path = global_manager.file_manager.save_file(file_path=file_path, content=file_content)
    global_manager.settings_builder.dataset = file_path

    settings_data = json.loads(settings)
    settings_data["dataset"] = file_path
    global_manager.settings_builder.set_settings(settings_data)


def finetuning_task(llm_tuner) -> None:
    output_dir = None
    try:
        llm_tuner.load_dataset(global_manager.settings_builder.dataset)
        output_dir = llm_tuner.output_dir  # Store for cleanup on failure
        path = llm_tuner.finetune()
        
        # Use the path returned from finetune (should be absolute)
        model_path = os.path.abspath(path) if not os.path.isabs(path) else path

        model_data = {
            "model_name": global_manager.settings_builder.fine_tuned_name.split('/')[-1] if global_manager.settings_builder.fine_tuned_name else os.path.basename(model_path),
            "base_model": global_manager.settings_builder.model_name,
            "task": global_manager.settings_builder.task,
            "description": f"Fine-tuned {global_manager.settings_builder.model_name} for {global_manager.settings_builder.task}" + 
                          (" (Custom Model)" if global_manager.settings_builder.is_custom_model else " (Recommended Model)"),
            "creation_date": datetime.now().isoformat(),
            "model_path": model_path,
            "is_custom_base_model": global_manager.settings_builder.is_custom_model
        }
        global_manager.db_manager.add_model(model_data)
    
    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        # Cleanup failed fine-tuning artifacts
        if output_dir and os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                print(f"Cleaned up failed fine-tuning artifacts at: {output_dir}")
            except Exception as cleanup_error:
                print(f"Warning: Could not cleanup output directory: {cleanup_error}")
        raise

    finally:
        global_manager.settings_cache.clear()
        global_manager.finetuning_status["status"] = "idle"
        global_manager.finetuning_status["message"] = ""
        global_manager.finetuning_status["progress"] = 0
        global_manager.settings_builder.reset()
        del llm_tuner


@router.get("/status")
async def finetuning_status_page(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": global_manager.finetuning_status["status"],
        "progress": global_manager.finetuning_status["progress"],
        "message": global_manager.finetuning_status["message"]
    })

@router.get("/start")
async def start_finetuning_page(request: Request, background_task: BackgroundTasks) -> JSONResponse:

    print(global_manager.settings_builder.get_settings())
    
    # Log whether using custom model
    if global_manager.settings_builder.is_custom_model:
        print(f"Starting finetuning with CUSTOM MODEL: {global_manager.settings_builder.model_name}")
    else:
        print(f"Starting finetuning with RECOMMENDED MODEL: {global_manager.settings_builder.model_name}")

    if not global_manager.settings_cache:
        raise HTTPException(
            status_code=400,
            detail="No hardware detection data available. Please run hardware detection first."
        )
    if global_manager.finetuning_status["status"] != "idle":
        print(global_manager.finetuning_status)
        raise HTTPException(
            status_code=400,
            detail="A finetuning is already in progress. Please wait until it completes."
        )
    
    # Validate available disk space (require at least 10GB free)
    try:
        stat = shutil.disk_usage(global_manager.model_path)
        available_gb = stat.free / (1024 ** 3)  # Convert to GB
        if available_gb < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient disk space. Available: {available_gb:.2f}GB. Required: at least 10GB."
            )
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")
    
    global_manager.finetuning_status["status"] = "initializing"
    global_manager.finetuning_status["message"] = "Starting finetuning process..."
    if global_manager.settings_builder.task == "text-generation":
        llm_tuner = CausalLLMFinetuner(
            model_name=global_manager.settings_builder.model_name,
            compute_specs=global_manager.settings_builder.compute_profile,
            pipeline_task="text-generation"
        )
    elif global_manager.settings_builder.task == "summarization":
        llm_tuner = Seq2SeqFinetuner(
            model_name=global_manager.settings_builder.model_name,
            compute_specs=global_manager.settings_builder.compute_profile,
            pipeline_task="summarization"
        )
    elif global_manager.settings_builder.task == "extractive-question-answering":
        llm_tuner = QuestionAnsweringTuner(
            model_name=global_manager.settings_builder.model_name,
            compute_specs=global_manager.settings_builder.compute_profile,
            pipeline_task="question-answering"
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task. Must be one of {VALID_TASKS_STR}."
        )

    llm_tuner.set_settings(**global_manager.settings_builder.get_settings())

    background_task.add_task(finetuning_task, llm_tuner)
    global_manager.finetuning_status["status"] = "running"
    global_manager.finetuning_status["message"] = "Finetuning process started."
    return JSONResponse({
        "app_name":global_manager.app_name,
        "message": "Finetuning process started.",
    })
