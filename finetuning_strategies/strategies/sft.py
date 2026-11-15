'''
Defines the Supervised Fine-Tuning (SFT) strategy for model fine-tuning.
This strategy involves training a model on a labeled dataset using supervised
learning techniques. The SFT class inherits from the Finetuner base class and
implements the finetune method to perform the fine-tuning process. The provider
attribute indicates the specific fine-tuning provider or framework being used for
this strategy.
'''
from finetuning_strategies.tasks.task import FinetuningTask
from finetuning_strategies.strategies.strategy import FinetuningStrategy
from typing import List, Literal, Any, Dict, Optional, Callable, Union
from finetuning_strategies.model_factory import TaskModels
from datasets import IterableDataset, Dataset
from .strategy import FinetuningStrategy
from providers.provider import Provider
from trl import SFTTrainer, SFTConfig
from peft import PeftConfig
from pydantic import BaseModel
from utilities.hardware_detector import HardwareDetector
from utilities.precision import training_precision
from recommendation_engine.validation_models.recommendation import RecommendedModel
import importlib.util


# Liger kernels are an optional CUDA-only extra (`pip install modelforge-finetuning[gpu]`).
# They are never installed on CPU or Windows, so probe before enabling them.
LIGER_KERNEL_AVAILABLE = importlib.util.find_spec("liger_kernel") is not None


def _liger_kernel_available() -> bool:
    """Liger needs CUDA *and* the optional liger-kernel package.

    TRL raises rather than falling back when the optional package is absent.
    Only enable it when the package is installed."""
    from importlib.util import find_spec

    try:
        return find_spec("liger_kernel") is not None
    except (ImportError, ValueError):
        return False


LIGER_KERNEL_AVAILABLE = _liger_kernel_available()


class SFTRecommendedConfig(BaseModel):
    """Recommended SFTTrainer configuration for a detected hardware profile.

    Everything here is forwarded to build_sft_config() verbatim, which is what
    lets an interface hand a modified copy back and have it take effect.

    The schedule fields below default to build_sft_config()'s own values, so a
    run started without touching them behaves exactly as it did before they
    were surfaced -- they were always in force, just unreachable."""
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    packing: bool
    packing_strategy: str
    max_length: int
    use_liger_kernel: bool
    activation_offloading: bool
    bf16: bool
    fp16: bool
    model_init_kwargs: dict
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-5
    per_device_eval_batch_size: int = 8
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42


class LoraRecommendedConfig(BaseModel):
    """Recommended LoRA adapter configuration."""
    r: int
    lora_alpha: int
    target_modules: list[str] | str


class SFT(FinetuningStrategy):

    optimizer = None
    config: Optional[SFTConfig] = None

    name = "Supervised Fine-Tuning"
    slug = "sft"
    description = """
    Supervised Fine-Tuning (SFT) is a strategy for fine-tuning pre-trained language models on specific tasks using labeled datasets. 
    In this approach, the model is trained to predict the correct output based on the input data, leveraging the knowledge it has acquired 
    during pre-training. SFT teaches the model to follow instructions, for output formats and to perform specific tasks by providing it with 
    examples of the desired behavior. 
    This strategy uses a highly-curated high-quality dataset that is often smaller in size compared to the datasets used for pre-training.
    """
    supported_providers = ["huggingface"]
    supported_tasks = ["causal_language_modeling", "seq2seq_modeling"]
    extra_model_copies = 0
    batch_multiplier = 1.0

    @staticmethod
    def validate_data_format(iterable_dataset: IterableDataset) -> bool:
        columns = list(iterable_dataset.features.keys())
        if "messages" in columns:
            return True
        if "text" in columns:
            return True
        if "prompt" in columns and "completion" in columns:
            return True
        raise ValueError(
            "Dataset must contain one of the following column sets: "
            "['messages'] for conversational format, "
            "['text'] for language modeling format, or "
            "['prompt', 'completion'] for prompt-completion format."
        )

    @staticmethod
    def apply_data_format(
        prompt: str,
        completion: str,
        type_: Literal["conversational", "language_modeling", "prompt", "conversational_prompt"] = "prompt",
    ):
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")
        if not isinstance(completion, str):
            raise ValueError("Completion must be a string.")

        if type_ == "conversational":
            return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]}
        elif type_ == "language_modeling":
            return {"text": prompt}
        elif type_ == "prompt":
            return {"prompt": prompt, "completion": completion}
        elif type_ == "conversational_prompt":
            return {
                "prompt": [{"role": "user", "content": prompt}],
                "completion": [{"role": "assistant", "content": completion}],
            }
        else:
            raise ValueError(
                f"Chat template format '{type_}' is invalid. "
                "Please choose one of 'conversational', 'language_modeling', 'prompt', or 'conversational_prompt'."
            )

    def build_sft_config(
        self,
        output_dir: str = "sft_output",
        # SFT-specific params
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        chat_template_path: Optional[str] = None,
        dataset_text_field: str = "text",
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        dataset_num_proc: Optional[int] = None,
        eos_token: Optional[str] = None,
        max_length: Optional[int] = 1024,
        truncation_mode: str = "keep_start",
        shuffle_dataset: bool = False,
        packing: bool = False,
        packing_strategy: str = "bfd",
        padding_free: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        eval_packing: Optional[bool] = None,
        completion_only_loss: Optional[bool] = None,
        assistant_only_loss: bool = False,
        loss_type: str = "nll",
        activation_offloading: bool = False,
        # Key TrainingArguments params
        num_train_epochs: float = 3.0,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.0,
        warmup_ratio: float = 0.0,
        warmup_steps: int = 0,
        lr_scheduler_type: str = "linear",
        optim: str = "adamw_torch",
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: Optional[int] = None,
        evaluation_strategy: str = "no",
        load_best_model_at_end: bool = False,
        gradient_checkpointing: bool = True,
        bf16: bool = True,
        fp16: bool = False,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        report_to: Union[str, List[str]] = "none",
        logging_dir: Optional[str] = None,
        seed: int = 42,
        dataloader_num_workers: int = 0,
        remove_unused_columns: bool = True,
        max_grad_norm: float = 1.0,
        **training_kwargs: Any,
    ) -> None:
        self.config = SFTConfig(
            output_dir=output_dir,
            model_init_kwargs=model_init_kwargs,
            chat_template_path=chat_template_path,
            dataset_text_field=dataset_text_field,
            dataset_kwargs=dataset_kwargs,
            dataset_num_proc=dataset_num_proc,
            eos_token=eos_token,
            max_length=max_length,
            truncation_mode=truncation_mode,
            shuffle_dataset=shuffle_dataset,
            packing=packing,
            packing_strategy=packing_strategy,
            padding_free=padding_free,
            pad_to_multiple_of=pad_to_multiple_of,
            eval_packing=eval_packing,
            completion_only_loss=completion_only_loss,
            assistant_only_loss=assistant_only_loss,
            loss_type=loss_type,
            activation_offloading=activation_offloading,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            lr_scheduler_type=lr_scheduler_type,
            optim=optim,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            load_best_model_at_end=load_best_model_at_end,
            gradient_checkpointing=gradient_checkpointing,
            bf16=bf16,
            fp16=fp16,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            report_to=report_to,
            logging_dir=logging_dir,
            seed=seed,
            dataloader_num_workers=dataloader_num_workers,
            remove_unused_columns=remove_unused_columns,
            max_grad_norm=max_grad_norm,
            **training_kwargs,
        )

    def set_config(self, config: Dict[str, Any]) -> None:
        if self.config is None:
            self.build_sft_config(**config)
        else:
            for key, value in config.items():
                setattr(self.config, key, value)

    @classmethod
    def build_recommended_config(
        cls,
        peft_tier: str,
        hardware: HardwareDetector.HardwareSpecs,
        chosen_model: RecommendedModel,
    ) -> tuple[SFTRecommendedConfig, LoraRecommendedConfig | None]:
        """Build a recommended SFT configuration keyed off peft_tier, hardware, and chosen model size.

        LoRA target modules use Llama-style names as the initial convention (known limitation:
        other architectures use different module names).
        """
        raw_vram_gb = sum(
            g["total_memory_gb"]
            for g in hardware.gpus
            if isinstance(g["total_memory_gb"], (int, float))
        )

        # Determine max_length and activation_offloading from available VRAM
        if raw_vram_gb < 12:
            max_length = 512
            activation_offloading = True
        elif raw_vram_gb < 32:
            max_length = 1024
            activation_offloading = False
        else:
            max_length = 2048
            activation_offloading = False

        # Determine gradient_accumulation_steps from VRAM (smaller VRAM needs larger accumulation
        # to compensate for batch_size=1)
        if raw_vram_gb < 12:
            gradient_accumulation_steps = 16
        elif raw_vram_gb < 32:
            gradient_accumulation_steps = 8
        else:
            gradient_accumulation_steps = 4

        # Determine quantization and whether bf16 is available for the base weights
        if peft_tier == "full_finetune":
            model_init_kwargs: dict = {}
            bf16 = True
            fp16 = False
        elif peft_tier == "lora_bf16":
            model_init_kwargs = {}
            bf16 = True
            fp16 = False
        elif peft_tier == "qlora_8bit":
            model_init_kwargs = {"quantization_config": {"load_in_8bit": True}}
            bf16 = True
            fp16 = False
        elif peft_tier == "qlora_4bit":
            model_init_kwargs = {"quantization_config": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True}}
            bf16 = True
            fp16 = False
        else:
            raise ValueError(
                f"Unknown peft_tier '{peft_tier}'. "
                f"Expected one of: full_finetune, lora_bf16, qlora_8bit, qlora_4bit."
            )

        # CPU-only: disable bf16 and quantisation (bitsandbytes requires CUDA)
        if raw_vram_gb == 0:
            bf16 = False
            fp16 = False
            model_init_kwargs = {}

        bf16, fp16 = training_precision(hardware)
        if raw_vram_gb == 0:
            activation_offloading = False
        strategy_config = SFTRecommendedConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            packing=True,
            packing_strategy="bfd",
            max_length=max_length,
            use_liger_kernel=(raw_vram_gb > 0 and _liger_kernel_available()),
            activation_offloading=activation_offloading,
            bf16=bf16,
            fp16=fp16,
            model_init_kwargs=model_init_kwargs,
        )

        if peft_tier == "full_finetune":
            return strategy_config, None

        # LoRA rank and target modules keyed off model parameter count
        p = chosen_model.num_parameters
        if p < 1_000_000_000:
            rank = 8
            target_modules = ["q_proj", "v_proj"]
        elif p < 7_000_000_000:
            rank = 16
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif p < 13_000_000_000:
            rank = 32
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            rank = 64
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]

        peft_config = LoraRecommendedConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules="all-linear",
        )
        return strategy_config, peft_config

    def finetune(
        self,
        model_name: str,
        dataset: Union[Dataset, IterableDataset],
        task_slug: str,
        provider: Provider,
        peft_approach: Literal["full_finetune", "lora_bf16", "qlora_8bit", "qlora_4bit"] = "full_finetune",
        output_dir: Optional[str] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict]] = None,
        peft_config: Optional[PeftConfig] = None,
        formatting_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[list] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> SFTTrainer:
        self.validate_data_format(dataset)

        sft_config = self.config if self.config is not None else SFTConfig(output_dir="sft_output")
        output_dir = output_dir or sft_config.output_dir

        model_init_kwargs = sft_config.model_init_kwargs or {}
        task_class: type[FinetuningTask] = TaskModels.get_task_class(task_slug)
        model = provider.load_model(model_name, task_class.model_class, **model_init_kwargs)
        tokenizer = provider.load_tokenizer(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if task_slug == "seq2seq_modeling":
            return self._finetune_seq2seq(
                model, tokenizer, dataset, sft_config, output_dir, peft_config,
                eval_dataset, callbacks, resume_from_checkpoint,
            )

        from utilities.chat_format import ensure_chat_template, normalize_supervised, map_evaluation
        if formatting_func is None and "text" not in dataset.column_names:
            ensure_chat_template(tokenizer)
            if "prompt" in dataset.column_names and "completion" in dataset.column_names:
                dataset = dataset.map(normalize_supervised)
                eval_dataset = map_evaluation(eval_dataset, normalize_supervised)

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        self.launch_tensorboard(sft_config, trainer=trainer)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        if peft_approach != "full_finetune":
            trainer.model.save_pretrained(output_dir)
        else:
            trainer.save_model(output_dir)

        tokenizer.save_pretrained(output_dir)
        return trainer

    def _finetune_seq2seq(
        self,
        model,
        tokenizer,
        dataset,
        sft_config,
        output_dir: str,
        peft_config,
        eval_dataset,
        callbacks,
        resume_from_checkpoint,
    ):
        """Run the documented prompt/completion contract with Seq2SeqTrainer.

        TRL's SFTTrainer is designed for causal LMs; passing an encoder-decoder
        model to it produces invalid labels.  Keep the SFT public strategy but
        route the seq2seq task through Transformers' native trainer.
        """
        from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
        from peft import get_peft_model

        required = {"prompt", "completion"}
        missing = required - set(dataset.column_names)
        if missing:
            raise ValueError(
                "Seq2seq training requires prompt and completion columns. "
                f"Missing: {', '.join(sorted(missing))}."
            )
        if eval_dataset is not None:
            missing = required - set(eval_dataset.column_names)
            if missing:
                raise ValueError(
                    "Seq2seq evaluation requires prompt and completion columns. "
                    f"Missing: {', '.join(sorted(missing))}."
                )
        max_length = sft_config.max_length or 1024

        def tokenize(batch):
            inputs = tokenizer(batch["prompt"], truncation=True, max_length=max_length)
            labels = tokenizer(text_target=batch["completion"], truncation=True, max_length=max_length)
            inputs["labels"] = labels["input_ids"]
            return inputs

        dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Only pass arguments common to the installed Transformers trainer;
        # SFT-only options such as packing must not leak into it.
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=sft_config.num_train_epochs,
            per_device_train_batch_size=sft_config.per_device_train_batch_size,
            per_device_eval_batch_size=sft_config.per_device_eval_batch_size,
            gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
            learning_rate=sft_config.learning_rate,
            weight_decay=sft_config.weight_decay,
            warmup_ratio=sft_config.warmup_ratio,
            warmup_steps=sft_config.warmup_steps,
            lr_scheduler_type=sft_config.lr_scheduler_type,
            optim=sft_config.optim,
            logging_steps=sft_config.logging_steps,
            save_steps=sft_config.save_steps,
            gradient_checkpointing=sft_config.gradient_checkpointing,
            bf16=sft_config.bf16,
            fp16=sft_config.fp16,
            report_to=sft_config.report_to,
            logging_dir=sft_config.logging_dir,
            seed=sft_config.seed,
            dataloader_num_workers=sft_config.dataloader_num_workers,
            remove_unused_columns=True,
            max_grad_norm=sft_config.max_grad_norm,
            fsdp=getattr(sft_config, "fsdp", ""),
            fsdp_config=getattr(sft_config, "fsdp_config", None),
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            callbacks=callbacks,
        )
        self.launch_tensorboard(args, trainer=trainer)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        return trainer
