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
            else:
                model = provider.load_model(
                    model_id=config["model_name"],
                    model_class=model_class,
                    quantization_config=quant_config,
                )
                tokenizer = provider.load_tokenizer(config["model_name"])

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

            # Get metrics function
            metrics_fn = MetricsCalculator.get_metrics_fn_for_task(
                config["task"],
                tokenizer
            )

            # Create trainer with progress callback
            self.training_status["message"] = "Creating trainer..."
            trainer = strategy.create_trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                config=config,
                callbacks=[ProgressCallback(self.training_status)],
            )

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
        """Format dataset based on task type."""
        if task == "text-generation":
            # Rename columns
            dataset = dataset.rename_column("input", "prompt")
            dataset = dataset.rename_column("output", "completion")
            # Apply formatting
            def format_fn(example):
                return {
                    "prompt": "USER: " + example.get("prompt", ""),
                    "completion": "ASSISTANT: " + example.get("completion", "") + "<|endoftext|>",
                }
            dataset = dataset.map(format_fn)

        elif task == "summarization":
            # Rename columns
            dataset = dataset.rename_column("document", "input")
            dataset = dataset.rename_column("summary", "output")

        elif task == "extractive-question-answering":
            # QA datasets need special tokenization
            # This is handled by the strategy
            pass

        return dataset

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
