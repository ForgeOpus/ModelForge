"""
Training configuration schemas.
Pydantic models for training request validation.
"""
from pydantic import BaseModel, field_validator, Field
from typing import Optional


VALID_TASKS = ["text-generation", "summarization", "extractive-question-answering"]
VALID_STRATEGIES = ["sft", "rlhf", "dpo", "qlora"]
VALID_PROVIDERS = ["huggingface", "unsloth"]


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    # Model and task settings
    task: str
    model_name: str
    provider: str = "huggingface"
    strategy: str = "sft"
    dataset: str
    compute_specs: str

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Quantization settings
    use_4bit: bool = True
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # Training precision
    fp16: bool = False
    bf16: bool = False

    # Training hyperparameters
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    max_steps: int = -1
    warmup_ratio: float = 0.03
    group_by_length: bool = True
    packing: bool = False

    # Sequence settings
    max_seq_length: Optional[int] = None

    # Evaluation settings
    eval_split: float = 0.2
    eval_steps: int = 100

    @field_validator("task")
    @classmethod
    def validate_task(cls, v):
        if v not in VALID_TASKS:
            raise ValueError(
                f"Invalid task: {v}. Must be one of {VALID_TASKS}"
            )
        return v

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v):
        if v not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy: {v}. Must be one of {VALID_STRATEGIES}"
            )
        return v

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v):
        if v not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider: {v}. Must be one of {VALID_PROVIDERS}"
            )
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v):
        if not v or not v.strip():
            raise ValueError("Dataset path cannot be empty")
        return v.strip()

    @field_validator("num_train_epochs")
    @classmethod
    def validate_epochs(cls, v):
        if v < 1:
            raise ValueError("Number of epochs must be at least 1")
        return v

    @field_validator("per_device_train_batch_size")
    @classmethod
    def validate_batch_size(cls, v):
        if v < 1:
            raise ValueError("Batch size must be at least 1")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("Learning rate must be between 0 and 1")
        return v

    @field_validator("lora_r")
    @classmethod
    def validate_lora_r(cls, v):
        if v < 1 or v > 256:
            raise ValueError("LoRA rank must be between 1 and 256")
        return v

    @field_validator("eval_split")
    @classmethod
    def validate_eval_split(cls, v):
        if v < 0 or v >= 1:
            raise ValueError("Evaluation split must be between 0 and 1")
        return v


class TaskSelection(BaseModel):
    """Task selection schema."""
    task: str

    @field_validator("task")
    @classmethod
    def validate_task(cls, v):
        if v not in VALID_TASKS:
            raise ValueError(
                f"Invalid task: {v}. Must be one of {VALID_TASKS}"
            )
        return v


class ModelSelection(BaseModel):
    """Model selection schema."""
    selected_model: str

    @field_validator("selected_model")
    @classmethod
    def validate_model(cls, v):
        if not v or not v.strip():
            raise ValueError("Selected model cannot be empty")
        return v.strip()


class ModelValidation(BaseModel):
    """Model validation schema."""
    repo_name: str

    @field_validator("repo_name")
    @classmethod
    def validate_repo_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Repository name cannot be empty")
        return v.strip()


class TrainingStatus(BaseModel):
    """Training status response."""
    status: str  # idle, running, completed, error
    progress: int = 0  # 0-100
    message: str = ""
    error: Optional[str] = None


class TrainingResult(BaseModel):
    """Training result response."""
    success: bool
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    message: str
    error: Optional[str] = None
