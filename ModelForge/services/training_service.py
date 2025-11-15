"""
Training service for orchestrating model fine-tuning.
Coordinates providers, strategies, and training execution.
"""
import os
import json
import uuid
from typing import Dict, Any, Optional
from datasets import load_dataset
from transformers import TrainerCallback

from ..providers.provider_factory import ProviderFactory
from ..strategies.strategy_factory import StrategyFactory
from ..utilities.finetuning.quantization import QuantizationFactory
from ..evaluation.dataset_validator import DatasetValidator
from ..evaluation.metrics import MetricsCalculator
from ..database.database_manager import DatabaseManager
from ..utilities.settings_managers.FileManager import FileManager
from ..exceptions import TrainingError, DatasetValidationError
from ..logging_config import logger


class ProgressCallback(TrainerCallback):
    """Callback to update training progress."""

    def __init__(self, status_dict: Dict):
        super().__init__()
        self.status_dict = status_dict

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Update progress during training."""
        if state.max_steps <= 0:
            return

        progress = min(95, int((state.global_step / state.max_steps) * 100))
        self.status_dict["progress"] = progress
        self.status_dict["message"] = f"Training step {state.global_step}/{state.max_steps}"

    def on_train_end(self, args, state, control, **kwargs):
        """Mark training as complete."""
        self.status_dict["progress"] = 100
        self.status_dict["message"] = "Training completed!"


class TrainingService:
    """Service for managing model training."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        file_manager: FileManager,
    ):
        """
        Initialize training service.

        Args:
            db_manager: Database manager instance
            file_manager: File manager instance
        """
        self.db_manager = db_manager
        self.file_manager = file_manager
        self.default_dirs = file_manager.return_default_dirs()

        # Training status (should be stored in Redis for production)
        self.training_status = {
            "status": "idle",
            "progress": 0,
            "message": "",
        }

        logger.info("Training service initialized")

    def get_training_status(self) -> Dict:
        """Get current training status."""
        return self.training_status.copy()

    def reset_training_status(self):
        """Reset training status to idle."""
        self.training_status = {
            "status": "idle",
            "progress": 0,
            "message": "",
        }

    def validate_and_prepare_dataset(
        self,
        dataset_path: str,
        task: str,
        strategy: str,
    ) -> Dict:
        """
        Validate dataset and get information.

        Args:
            dataset_path: Path to dataset file
            task: Task type
            strategy: Training strategy

        Returns:
            Dictionary with dataset info

        Raises:
            DatasetValidationError: If validation fails
        """
        logger.info(f"Validating dataset: {dataset_path}")

        # Validate dataset
        DatasetValidator.validate_dataset(
            dataset_path=dataset_path,
            task=task,
            strategy=strategy,
            min_examples=10,
        )

        # Load and get info
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        return {
            "num_examples": len(dataset),
            "fields": dataset.column_names,
        }

    def train_model(
        self,
        config: Dict[str, Any],
        background: bool = False,
    ) -> Dict:
        """
        Train a model with the given configuration.

        Args:
            config: Training configuration dictionary
            background: Whether to run in background (for async execution)

        Returns:
            Dictionary with training result

        Raises:
            TrainingError: If training fails
        """
        logger.info(f"Starting training with config: {config.get('task')}, {config.get('strategy')}")

        try:
            # Update status
            self.training_status["status"] = "running"
            self.training_status["progress"] = 0
            self.training_status["message"] = "Initializing training..."

            # Create provider
            provider_name = config.get("provider", "huggingface")

            # CRITICAL: Configure single-process mode for Unsloth BEFORE any initialization
            # This must happen before creating provider, strategy, or loading model
            # to ensure AcceleratorState initializes in non-distributed mode
            if provider_name == "unsloth":
                logger.info("Configuring single-process mode for Unsloth compatibility")

                # Remove ALL distributed environment variables to prevent auto-detection
                distributed_vars = [
                    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR", "MASTER_PORT",
                    "PMI_RANK", "PMI_SIZE",
                    "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE",
                    "MV2_COMM_WORLD_RANK", "MV2_COMM_WORLD_SIZE"
                ]
                for var in distributed_vars:
                    if var in os.environ:
                        logger.debug(f"Removing distributed environment variable: {var}")
                        del os.environ[var]

                # Explicitly set single-process indicators
                os.environ["WORLD_SIZE"] = "1"
                os.environ["LOCAL_RANK"] = "-1"
                os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"  # Bypass device_map check in Accelerate

                # Reset AcceleratorState if it was already initialized (defensive)
                try:
                    from accelerate.state import AcceleratorState, PartialState
                    if AcceleratorState._shared_state or PartialState._shared_state:
                        logger.warning("Resetting Accelerate state for Unsloth single-process mode")
                        AcceleratorState._reset_state(reset_partial_state=True)
                except Exception as e:
                    logger.debug(f"Could not reset Accelerate state: {e}")

                logger.info("Single-process mode configured successfully")

            provider = ProviderFactory.create_provider(provider_name)

            # Create strategy
            strategy_name = config.get("strategy", "sft")
            strategy = StrategyFactory.create_strategy(strategy_name)

            # Create quantization config
            quant_config = QuantizationFactory.create_config(
                use_4bit=config.get("use_4bit", True),
                use_8bit=config.get("use_8bit", False),
                compute_dtype=config.get("bnb_4bit_compute_dtype", "float16"),
                quant_type=config.get("bnb_4bit_quant_type", "nf4"),
                use_double_quant=config.get("use_nested_quant", False),
            )

            # Load model
            self.training_status["message"] = "Loading model..."
            model_class_map = {
                "text-generation": "AutoModelForCausalLM",
                "summarization": "AutoModelForSeq2SeqLM",
                "extractive-question-answering": "AutoModelForQuestionAnswering",
            }
            model_class = model_class_map[config["task"]]

            # Handle Unsloth special case (returns model and tokenizer together)
            if provider_name == "unsloth":
                model, tokenizer = provider.load_model(
                    model_id=config["model_name"],
                    model_class=model_class,
                    quantization_config=quant_config,
                    max_seq_length=config.get("max_seq_length", 2048),
                )
                tokenizer.eos_token = tokenizer.eos_token or tokenizer.sep_token
                # Store eos_token in config for use by training strategies
                config["eos_token"] = tokenizer.eos_token
            else:
                model = provider.load_model(
                    model_id=config["model_name"],
                    model_class=model_class,
                    quantization_config=quant_config,
                )
                tokenizer = provider.load_tokenizer(config["model_name"])
                tokenizer.eos_token = tokenizer.eos_token or tokenizer.sep_token
                # Store eos_token in config for use by training strategies
                config["eos_token"] = tokenizer.eos_token

            # Auto-detect and correct precision settings to prevent Unsloth errors
            config = self._auto_detect_precision_settings(model, config)

            # Load and prepare dataset
            self.training_status["message"] = "Loading dataset..."
            dataset = load_dataset(
                "json",
                data_files=config["dataset"],
                split="train"
            )

            # Format dataset based on task
            dataset = self._format_dataset(dataset, config["task"], config.get("compute_specs", "low_end"))

            # Split into train/eval
            eval_split = config.get("eval_split", 0.2)
            if eval_split > 0:
                split_dataset = dataset.train_test_split(test_size=eval_split, seed=42)
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
            else:
                train_dataset = dataset
                eval_dataset = None

            # Prepare dataset with strategy
            train_dataset = strategy.prepare_dataset(train_dataset, tokenizer, config)
            if eval_dataset:
                eval_dataset = strategy.prepare_dataset(eval_dataset, tokenizer, config)

            # Prepare model with strategy
            self.training_status["message"] = "Preparing model for training..."

            # Handle Unsloth special case for model preparation
            if provider_name == "unsloth":
                model = provider.prepare_for_training(
                    model=model,
                    lora_r=config.get("lora_r", 16),
                    lora_alpha=config.get("lora_alpha", 32),
                    lora_dropout=config.get("lora_dropout", 0.1),
                )
            else:
                model = strategy.prepare_model(model, config)

            # Generate output paths
            model_id = str(uuid.uuid4())
            safe_model_name = config["model_name"].replace("/", "-").replace("\\", "-")
            model_output_path = os.path.join(
                self.default_dirs["models"],
                f"{safe_model_name}_{model_id}"
            )
            checkpoint_dir = os.path.join(
                self.default_dirs["model_checkpoints"],
                f"{safe_model_name}_{model_id}"
            )

            # Update config with paths
            config["output_dir"] = checkpoint_dir
            config["logging_dir"] = "./training_logs"

            # Calculate max_steps for proper progress tracking
            if config.get("max_steps", -1) <= 0:
                # Calculate based on dataset size and batch settings
                num_examples = len(train_dataset)
                batch_size = config.get("per_device_train_batch_size", 1)
                gradient_accumulation = config.get("gradient_accumulation_steps", 4)
                num_epochs = config.get("num_train_epochs", 1)
                
                effective_batch_size = batch_size * gradient_accumulation
                steps_per_epoch = max(1, num_examples // effective_batch_size)
                total_steps = steps_per_epoch * num_epochs
                
                config["max_steps"] = total_steps
                logger.info(f"Calculated max_steps: {total_steps} (epochs={num_epochs}, examples={num_examples}, effective_batch={effective_batch_size})")

            tokenizer.eos_token = tokenizer.eos_token or tokenizer.sep_token
            # Store eos_token in config for use by training strategies
            config["eos_token"] = tokenizer.eos_token

            # Get metrics function
            metrics_fn = MetricsCalculator.get_metrics_fn_for_task(
                config["task"],
                tokenizer
            )

            # Create trainer with progress callback and precision failsafe
            self.training_status["message"] = "Creating trainer..."
            trainer = self._create_trainer_with_failsafe(
                strategy=strategy,
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                config=config,
                callbacks=[ProgressCallback(self.training_status)],
            )

            # Verify single-process mode for Unsloth (debug logging)
            if provider_name == "unsloth":
                try:
                    from accelerate.state import PartialState
                    state = PartialState()
                    logger.info(
                        f"Accelerate state verified - "
                        f"Distributed type: {state.distributed_type}, "
                        f"Num processes: {state.num_processes}, "
                        f"Use distributed: {state.use_distributed}"
                    )
                except Exception as e:
                    logger.debug(f"Could not verify Accelerate state: {e}")

            # Train
            self.training_status["message"] = "Training in progress..."
            trainer.train()

            # Save model
            self.training_status["message"] = "Saving model..."
            trainer.model.save_pretrained(model_output_path)

            # Save tokenizer
            tokenizer.save_pretrained(model_output_path)

            # Create modelforge config file
            self._create_model_config(
                model_output_path,
                config["task"],
                model_class,
            )

            # Add to database
            self.db_manager.add_model(
                model_id=model_id,
                name=f"{safe_model_name}_finetuned",
                base_model=config["model_name"],
                task=config["task"],
                path=model_output_path,
                strategy=strategy_name,
                provider=provider_name,
                compute_profile=config.get("compute_specs"),
                config=json.dumps(config),
            )

            # Update status
            self.training_status["status"] = "completed"
            self.training_status["progress"] = 100
            self.training_status["message"] = "Training completed successfully!"

            logger.info(f"Training completed successfully: {model_id}")

            return {
                "success": True,
                "model_id": model_id,
                "model_path": model_output_path,
                "message": "Training completed successfully",
            }

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self.training_status["status"] = "error"
            self.training_status["message"] = str(e)

            return {
                "success": False,
                "model_id": None,
                "model_path": None,
                "message": "Training failed",
                "error": str(e),
            }

    def _format_dataset(self, dataset, task: str, compute_specs: str):
        """
        Format dataset based on task type.

        This method only renames columns to match expected field names.
        Text formatting (prefixes, EOS tokens) is handled by task-specific
        formatting_func in the training strategies.
        """
        if task == "text-generation":
            # Rename columns to expected names (formatting_func will add prefixes/EOS)
            dataset = dataset.rename_column("input", "prompt")
            dataset = dataset.rename_column("output", "completion")

        elif task == "summarization":
            # Rename columns to expected names
            dataset = dataset.rename_column("document", "input")
            dataset = dataset.rename_column("summary", "output")

        elif task == "extractive-question-answering":
            # QA datasets are already in correct format (context, question, answers)
            # No renaming needed
            pass

        return dataset

    def _auto_detect_precision_settings(self, model: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-detect model precision and correct fp16/bf16 config settings.

        Detects model dtype and auto-corrects precision settings to prevent
        trainer errors when config doesn't match model dtype.

        Args:
            model: Loaded model instance (before PEFT wrapping)
            config: Training configuration dictionary

        Returns:
            Updated config dictionary with corrected precision settings

        Note:
            Mutates config in-place and returns it for convenience.
            Should be called after model loading, before PEFT preparation.
        """
        try:
            # Import torch locally to avoid adding dependency to module level
            import torch

            # Try to get model dtype - handle PEFT-wrapped models
            model_dtype = None

            # Try direct access first
            if hasattr(model, 'dtype'):
                model_dtype = model.dtype
            # Try base_model for PEFT models (shouldn't be wrapped yet, but defensive)
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'dtype'):
                model_dtype = model.base_model.dtype
            # Try config.torch_dtype as fallback
            elif hasattr(model, 'config') and hasattr(model.config, 'torch_dtype'):
                model_dtype = model.config.torch_dtype

            if model_dtype is None:
                logger.warning(
                    "Could not determine model dtype for precision auto-detection. "
                    "Using config values as-is."
                )
                return config

            # Map torch dtypes to config flags
            # Note: Precision flags must match model dtype even for quantized models
            dtype_to_config = {
                torch.float16: ("fp16", True, "bf16", False),
                torch.bfloat16: ("bf16", True, "fp16", False),
                torch.float32: ("fp16", False, "bf16", False),
            }

            if model_dtype not in dtype_to_config:
                logger.warning(
                    f"Unknown model dtype: {model_dtype}. "
                    f"Expected float16, bfloat16, or float32. "
                    f"Using config values as-is."
                )
                return config

            # Get expected config values for this dtype
            enable_flag, enable_value, disable_flag, disable_value = dtype_to_config[model_dtype]

            # Get current config values
            current_enable = config.get(enable_flag, False)
            current_disable = config.get(disable_flag, False)

            # Check if correction is needed
            needs_correction = (current_enable != enable_value) or (current_disable != disable_value)

            if needs_correction:
                logger.warning(
                    f"Model dtype mismatch detected! "
                    f"Model dtype: {model_dtype}, "
                    f"Config: {enable_flag}={current_enable}, {disable_flag}={current_disable}. "
                    f"Auto-correcting to: {enable_flag}={enable_value}, {disable_flag}={disable_value}"
                )

                # Apply corrections
                config[enable_flag] = enable_value
                config[disable_flag] = disable_value

                logger.info(
                    f"Precision settings corrected: {enable_flag}={enable_value}, "
                    f"{disable_flag}={disable_value}"
                )
            else:
                logger.info(
                    f"Precision settings validated: Model dtype {model_dtype} matches "
                    f"config ({enable_flag}={enable_value}, {disable_flag}={disable_value})"
                )

            return config

        except ImportError as e:
            logger.error(f"Failed to import torch for precision detection: {e}")
            return config
        except Exception as e:
            # Defensive: never crash training due to auto-detection
            logger.warning(
                f"Error during precision auto-detection: {e}. "
                f"Using config values as-is.",
                exc_info=True
            )
            return config

    def _create_trainer_with_failsafe(
        self,
        strategy: Any,
        model: Any,
        train_dataset: Any,
        eval_dataset: Any,
        tokenizer: Any,
        config: Dict,
        callbacks: list = None,
    ) -> Any:
        """
        Create trainer with automatic precision mismatch recovery.

        This is a failsafe wrapper that catches Unsloth precision errors
        and automatically corrects the config to match the model's actual dtype.

        Args:
            strategy: Training strategy instance
            model: Prepared model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer instance
            config: Training configuration
            callbacks: Training callbacks

        Returns:
            Trainer instance

        Note:
            This is a last-resort failsafe. The primary auto-detection should
            prevent these errors, but this ensures we NEVER fail with precision
            mismatches regardless of what went wrong upstream.
        """
        try:
            # Attempt to create trainer
            return strategy.create_trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                config=config,
                callbacks=callbacks,
            )

        except TypeError as e:
            # Check if this is an Unsloth precision mismatch error
            error_msg = str(e)

            if "Unsloth:" in error_msg and ("bfloat16" in error_msg or "float16" in error_msg):
                logger.warning(
                    f"Unsloth precision mismatch detected: {error_msg}. "
                    f"Auto-correcting config and retrying..."
                )

                # Parse error message to determine correct precision
                # Error format: "Unsloth: Model is in X precision but you want to use Y precision"
                model_is_bfloat16 = "Model is in bfloat16" in error_msg
                model_is_float16 = "Model is in float16" in error_msg or "Model is in fp16" in error_msg

                if model_is_bfloat16:
                    logger.info("Correcting config to bf16=True, fp16=False")
                    config["bf16"] = True
                    config["fp16"] = False
                elif model_is_float16:
                    logger.info("Correcting config to fp16=True, bf16=False")
                    config["fp16"] = True
                    config["bf16"] = False
                else:
                    # Couldn't parse - try to detect from model directly
                    logger.warning("Could not parse error message, attempting direct model dtype detection")
                    import torch

                    model_dtype = None
                    if hasattr(model, 'dtype'):
                        model_dtype = model.dtype
                    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'dtype'):
                        model_dtype = model.base_model.dtype
                    elif hasattr(model, 'config') and hasattr(model.config, 'torch_dtype'):
                        model_dtype = model.config.torch_dtype

                    if model_dtype == torch.bfloat16:
                        logger.info("Detected model dtype: bfloat16, setting bf16=True, fp16=False")
                        config["bf16"] = True
                        config["fp16"] = False
                    elif model_dtype == torch.float16:
                        logger.info("Detected model dtype: float16, setting fp16=True, bf16=False")
                        config["fp16"] = True
                        config["bf16"] = False
                    else:
                        # Last resort: default to bf16 (safer for modern GPUs)
                        logger.warning(
                            f"Could not determine model dtype (got {model_dtype}). "
                            f"Defaulting to bf16=True, fp16=False"
                        )
                        config["bf16"] = True
                        config["fp16"] = False

                # Retry with corrected config
                logger.info("Retrying trainer creation with corrected precision settings...")
                return strategy.create_trainer(
                    model=model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=tokenizer,
                    config=config,
                    callbacks=callbacks,
                )
            else:
                # Not an Unsloth precision error, re-raise
                raise

    def _create_model_config(self, config_dir: str, pipeline_task: str, model_class: str):
        """Create modelforge config file for playground compatibility."""
        try:
            config = {
                "model_class": model_class.replace("AutoModel", "AutoPeftModel"),
                "pipeline_task": pipeline_task,
            }

            config_path = os.path.join(config_dir, "modelforge_config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

            logger.info(f"Config file created: {config_path}")

        except Exception as e:
            logger.error(f"Error creating config file: {e}")
