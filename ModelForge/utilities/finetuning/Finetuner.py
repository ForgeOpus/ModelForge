from abc import ABC, abstractmethod
import webbrowser
import tensorboard
from typing import Dict, List, Optional, Union
import json
from transformers import TrainerCallback


class ProgressCallback(TrainerCallback):
    """
    Callback to update the global finetuning status progress during training.
    """
    def __init__(self):
        super().__init__()
        from ...globals.globals_instance import global_manager
        self.global_manager = global_manager
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when logging happens. Updates progress based on training steps.
        """
        # Determine total steps - state.max_steps is automatically calculated by trainer
        # even when using epochs instead of max_steps parameter
        if state.max_steps <= 0:
            # Can't calculate progress without knowing total steps
            return
            
        # Calculate progress as percentage of total steps
        # Cap at 95% during training, will reach 100% on completion
        progress = min(95, int((state.global_step / state.max_steps) * 100))
        self.global_manager.finetuning_status["progress"] = progress
        self.global_manager.finetuning_status["message"] = f"Training step {state.global_step}/{state.max_steps}"
        
    def on_train_end(self, args, state, control, **kwargs):
        """
        Called at the end of training. Sets progress to 100%.
        """
        self.global_manager.finetuning_status["progress"] = 100
        self.global_manager.finetuning_status["message"] = "Training completed!"


class Finetuner(ABC):
    """
    Abstract base class for finetuning models.
    """
    def __init__(self, model_name: str, compute_specs: str, pipeline_task) -> None:
        """
        Initialize the Finetuner with model name and logging directory.

        :param model_name: The name of the model to be finetuned.
        :param compute_specs: Directory for logging.
        """

        self.pipeline_task = pipeline_task
        self.compute_specs = compute_specs

        # BitsAndBytesConfig settings
        self.load_in_4bit = None
        self.use_4bit = None
        self.bnb_4bit_quant_type = None
        self.bnb_4bit_compute_dtype = None
        self.bnb_4bit_use_quant_type = None

        self.load_in_8bit = None
        self.use_8bit = None
        self.bnb_8bit_quant_type = None
        self.bnb_8bit_compute_dtype = None
        self.bnb_8bit_use_quant_type = None

        self.use_nested_quant = None

        # PEFT settings
        self.lora_r = None
        self.lora_alpha = None
        self.lora_dropout = None

        # TrainingArguments settings
        self.optim = None
        self.weight_decay = None
        self.learning_rate = None
        self.gradient_checkpointing = None
        self.max_grad_norm = None
        self.gradient_accumulation_steps = None
        self.per_device_eval_batch_size = None
        self.per_device_train_batch_size = None
        self.lr_scheduler_type = None
        self.bf16 = None
        self.fp16 = None
        self.max_steps = None
        self.warmup_ratio = None
        self.group_by_length = None
        self.packing = None
        self.logging_steps = 25
        self.save_steps = 0
        self.num_train_epochs = None
        self.max_seq_length = None

        # Extras
        self.device_map = None
        self.output_dir = None
        self.fine_tuned_name = None
        self.dataset = None
        self.model_name = model_name
        self.logging_dir = "./training_logs"


    @staticmethod
    @abstractmethod
    def format_example(example: dict, specs: str, **kwargs) -> Dict:
        """
        Format the example for training with the chat templates of the expected models.
        :param example: The example to be formatted.
        :param specs: The computational environment specs (low_end, mid_range, or high_end).
        :return: A dictionary of the formatted example with the correct keys.
        """
        pass

    @abstractmethod
    def load_dataset(self, dataset_path: str) -> None:
        """
        Load the dataset from the specified path.
        :param dataset_path: Path to the dataset file.
        :return: None
        """
        pass

    @staticmethod
    def gen_uuid() -> str:
        """
        Generate a unique identifier for the finetuned model.
        :return: A string representing the unique identifier.
        """
        import uuid
        return str(uuid.uuid4())

    def set_settings(self, **kwargs) -> None:
        """
        Set the settings for the finetuner based on the provided keyword arguments.
        :param kwargs: The keyword arguments to set the settings.
        :return: None
        """
        from ...globals.globals_instance import global_manager

        # Basic settings
        uid = self.gen_uuid()
        safe_model_name = self.model_name.replace('/', '-').replace('\\', '-')
        
        # Use FileManager default directories for consistent structure
        default_dirs = global_manager.file_manager.return_default_dirs()
        self.fine_tuned_name = f"{default_dirs['models']}/{safe_model_name}_{uid}"
        self.output_dir = f"{default_dirs['model_checkpoints']}/{safe_model_name}_{uid}"
        self.num_train_epochs = kwargs.get('num_train_epochs')

        self.max_seq_length = kwargs.get('max_seq_length')

        # LoRA settings
        self.lora_r = kwargs.get('lora_r')
        self.lora_alpha = kwargs.get('lora_alpha')
        self.lora_dropout = kwargs.get('lora_dropout')

        # Quantization settings
        self.use_4bit = kwargs.get('use_4bit')
        self.load_in_4bit = kwargs.get('load_in_4bit')
        self.bnb_4bit_compute_dtype = kwargs.get('bnb_4bit_compute_dtype')
        self.bnb_4bit_quant_type = kwargs.get('bnb_4bit_quant_type')
        self.bnb_4bit_use_quant_type = kwargs.get('bnb_4bit_use_quant_type')

        self.use_8bit = kwargs.get('use_8bit')
        self.load_in_8bit = kwargs.get('load_in_8bit')
        self.bnb_8bit_compute_dtype = kwargs.get('bnb_8bit_compute_dtype')
        self.bnb_8bit_quant_type = kwargs.get('bnb_8bit_quant_type')
        self.bnb_8bit_use_quant_type = kwargs.get('bnb_8bit_use_quant_type')

        self.use_nested_quant = kwargs.get('use_nested_quant')

        # Training precision
        self.fp16 = kwargs.get('fp16')
        self.bf16 = kwargs.get('bf16')

        # Batch and steps configuration
        self.per_device_train_batch_size = kwargs.get('per_device_train_batch_size')
        self.per_device_eval_batch_size = kwargs.get('per_device_eval_batch_size')
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps')
        self.save_steps = kwargs.get('save_steps') if kwargs.get('save_steps') else self.save_steps
        self.logging_steps = kwargs.get('logging_steps') if kwargs.get('logging_steps') else self.logging_steps

        # Optimization settings
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing')
        self.max_grad_norm = kwargs.get('max_grad_norm')
        self.learning_rate = kwargs.get('learning_rate')
        self.weight_decay = kwargs.get('weight_decay')
        self.optim = kwargs.get('optim')
        self.lr_scheduler_type = kwargs.get('lr_scheduler_type')
        self.max_steps = kwargs.get('max_steps')
        self.warmup_ratio = kwargs.get('warmup_ratio')

        # Advanced configuration
        self.group_by_length = kwargs.get('group_by_length')
        self.packing = kwargs.get('packing')
        self.device_map = kwargs.get('device_map')

    def invalid_access(self):
        print("*"*100)
        print("You do not have access to this model.")
        print(f"Please visit https://huggingface.co/{self.model_name} to request access.")
        print("*" * 100)

    @abstractmethod
    def finetune(self) -> bool | str:
        """
        Finetune the model with the provided data.
        :return: model_path if model is successfully fine-tuned, False otherwise.
        """
        pass

    @staticmethod
    def build_config_file(config_dir: str, pipeline_task:str, model_class: str) -> bool:
        """
        Builds the chat playground configuration file for the fine-tuned model.
        :param config_dir: Path to the model adapter settings directory. This is where the configurations file will also be saved.
        :param pipeline_task: The pipeline task flag for the model, as defined by huggingface. eg "text-generation".
        :param model_class: The model class name for the model, as defined by huggingface. eg "AutoModelForCausalLM".
        :return: True if the configuration file is successfully created, False otherwise.
        """
        print(config_dir)
        try:
            with open(config_dir + "/modelforge_config.json", "w") as f:
                configs = {
                    "model_class": model_class,
                    "pipeline_task": pipeline_task,
                }
                configs = json.dumps(configs, indent=4)
                f.write(configs)
            print(f"Configuration file saved to {config_dir}/modelforge_config.json")
            return True
        except Exception as e:
            print(f"Error saving configuration file: {e}")
            return False

    def initiate_tensorboard(self) -> None:
        """
        Initialize TensorBoard for logging.
        :return: None
        """
        try:
            tb = tensorboard.program.TensorBoard()
            tb.configure(argv=[None, '--logdir', self.logging_dir])
            url = tb.launch()
            webbrowser.open(url)
        except Exception as e:
            print("Error starting TensorBoard:", e)

    def report_finish(self, message=None, error=False) -> None:
        """
        Report the finish of the fine-tuning process.
        :param message: Error message if finetuning fails.
        :param error: True if an error occurred, False otherwise.
        :return: None
        """
        print("*" * 100)
        if not error:
            print("Model fine-tuned successfully!")
            print(f"Model save to {self.fine_tuned_name}")
            print(f"Try out your new model in our chat playground!")
            self.initiate_tensorboard()
        else:
            print("Model fine-tuning failed!")
            print(f"Error occurred: {message}")
        print("*" * 100)
