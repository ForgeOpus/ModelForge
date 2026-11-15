from finetuning_strategies.tasks.task import FinetuningTask
from finetuning_strategies.strategies.strategy import FinetuningStrategy
from finetuning_strategies.strategies.sft import LoraRecommendedConfig, LIGER_KERNEL_AVAILABLE
from typing import List, Literal, Any, Dict, Optional, Callable, Union
from finetuning_strategies.model_factory import TaskModels
from datasets import IterableDataset, Dataset
from providers.provider import Provider
from trl import DPOTrainer, DPOConfig
from peft import PeftConfig
from pydantic import BaseModel
from utilities.hardware_detector import HardwareDetector
from utilities.precision import training_precision
from recommendation_engine.validation_models.recommendation import RecommendedModel


class DPORecommendedConfig(BaseModel):
    """Recommended DPOTrainer configuration for a detected hardware profile."""
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_length: int
    use_liger_kernel: bool
    activation_offloading: bool
    bf16: bool
    fp16: bool
    model_init_kwargs: dict
    beta: float
    loss_type: str
    learning_rate: Optional[float] = None
    # Defaults mirror build_dpo_config(), so surfacing them changes nothing
    # for a run that leaves them alone.
    num_train_epochs: float = 3.0
    per_device_eval_batch_size: int = 8
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42


class DPO(FinetuningStrategy):

    optimizer = None
    config: Optional[DPOConfig] = None

    name = "Direct Preference Optimization"
    slug = "dpo"
    description = """Fine-tune language models using direct preference feedback for more efficient optimization.
    In this strategy, the model is trained to directly optimize for user preferences, allowing for faster
    convergence and improved performance on tasks where preference data is available. This strategy avoids the
    need of a dedicated reward model and replaces it with a pairwise loss that compares model outputs based on preference labels.
    This provides a lower memory overhead compared to reinforcement learning approaches, while still effectively leveraging preference data for fine-tuning.
    """

    supported_providers = ["huggingface"]
    supported_tasks = ["causal_language_modeling"]
    # PEFT: ref model is the frozen base implicitly managed by DPOTrainer — no extra copy.
    # Full fine-tune: a separate ref_model is explicitly loaded, so 1 extra copy is required.
    # Set to 1 to be conservative (slight VRAM over-estimate for PEFT tiers).
    extra_model_copies = 1
    # DPO processes chosen + rejected in each forward pass, doubling activation memory vs SFT.
    batch_multiplier = 2.0

    @staticmethod
    def validate_data_format(iterable_dataset: IterableDataset) -> None:
        required_columns = {"chosen", "rejected"}
        column_names = set(iterable_dataset.column_names)
        if not required_columns.issubset(column_names):
            raise ValueError(
                f"Dataset must contain the following columns: {required_columns}. "
                f"Found columns: {set(iterable_dataset.column_names)}"
            )

    @staticmethod
    def normalize_conversational_pairs(dataset: Union[Dataset, IterableDataset]):
        """Normalize every preference row before TRL applies the model template."""
        from utilities.chat_format import normalize_preference

        return dataset.map(normalize_preference)

    @staticmethod
    def apply_data_format(
        prompt: Optional[str],
        preferred_output: str,
        non_preferred_output: str,
        type_: Literal["standard", "conversational"]
    ) -> Dict[str, Any]:
        if type_ == "standard":
            if prompt:
                return {"prompt": prompt, "chosen": preferred_output, "rejected": non_preferred_output}
            else:
                return {"chosen": preferred_output, "rejected": non_preferred_output}
        elif type_ == "conversational":
            if prompt:
                return {
                    "prompt": [{"role": "user", "content": prompt}],
                    "chosen": [{"role": "assistant", "content": preferred_output}],
                    "rejected": [{"role": "assistant", "content": non_preferred_output}],
                }
            else:
                return {
                    "chosen": [{"role": "assistant", "content": preferred_output}],
                    "rejected": [{"role": "assistant", "content": non_preferred_output}],
                }
        else:
            raise ValueError(
                f"Unsupported formatting type: {type_}.\nPlease choose from 'standard' or 'conversational'."
            )

    def build_dpo_config(
        self,
        output_dir: str = "dpo_output",
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        max_length: Optional[int] = 1024,
        num_train_epochs: float = 3.0,
        per_device_train_batch_size: int = 1,
        per_device_eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 5e-7,
        weight_decay: float = 0.0,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "cosine",
        optim: str = "adamw_torch",
        logging_steps: int = 10,
        save_steps: int = 500,
        gradient_checkpointing: bool = True,
        use_liger_kernel: bool = False,
        activation_offloading: bool = False,
        bf16: bool = True,
        fp16: bool = False,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        report_to: Union[str, List[str]] = "none",
        logging_dir: Optional[str] = None,
        seed: int = 42,
        **training_kwargs: Any,
    ) -> None:
        self.config = DPOConfig(
            output_dir=output_dir,
            beta=beta,
            loss_type=loss_type,
            model_init_kwargs=model_init_kwargs,
            max_length=max_length,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            optim=optim,
            logging_steps=logging_steps,
            save_steps=save_steps,
            gradient_checkpointing=gradient_checkpointing,
            use_liger_kernel=use_liger_kernel,
            activation_offloading=activation_offloading,
            bf16=bf16,
            fp16=fp16,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            report_to=report_to,
            logging_dir=logging_dir,
            seed=seed,
            **training_kwargs,
        )

    def set_config(self, config: Dict[str, Any]) -> None:
        if self.config is None:
            self.build_dpo_config(**config)
        else:
            for key, value in config.items():
                setattr(self.config, key, value)

    @classmethod
    def build_recommended_config(
        cls,
        peft_tier: str,
        hardware: HardwareDetector.HardwareSpecs,
        chosen_model: RecommendedModel,
    ) -> tuple[DPORecommendedConfig, LoraRecommendedConfig | None]:
        """Build a recommended DPO configuration keyed off peft_tier, hardware, and chosen model size."""
        raw_vram_gb = sum(
            g["total_memory_gb"]
            for g in hardware.gpus
            if isinstance(g["total_memory_gb"], (int, float))
        )

        if raw_vram_gb < 12:
            max_length = 512
            activation_offloading = True
        elif raw_vram_gb < 32:
            max_length = 1024
            activation_offloading = False
        else:
            max_length = 2048
            activation_offloading = False

        # DPO's doubled activation budget (batch_multiplier=2.0) means tighter VRAM tolerance —
        # use larger gradient accumulation steps to compensate for mandatory batch_size=1.
        if raw_vram_gb < 12:
            gradient_accumulation_steps = 32
        elif raw_vram_gb < 32:
            gradient_accumulation_steps = 16
        else:
            gradient_accumulation_steps = 8

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

        if raw_vram_gb == 0:
            bf16 = False
            fp16 = False
            model_init_kwargs = {}

        # DPO requires conservative LRs — preference alignment is sensitive to overshooting.
        # Full FT is even more conservative since all weights are updated simultaneously.
        p = chosen_model.num_parameters
        if peft_tier == "full_finetune":
            learning_rate = 1e-7
        elif p < 1_000_000_000:
            learning_rate = 5e-6
        elif p < 7_000_000_000:
            learning_rate = 5e-7
        elif p < 13_000_000_000:
            learning_rate = 2e-7
        else:
            learning_rate = 1e-7

        bf16, fp16 = training_precision(hardware)
        if raw_vram_gb == 0:
            activation_offloading = False
        strategy_config = DPORecommendedConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            max_length=max_length,
            use_liger_kernel=(raw_vram_gb > 0 and LIGER_KERNEL_AVAILABLE),
            activation_offloading=activation_offloading,
            bf16=bf16,
            fp16=fp16,
            model_init_kwargs=model_init_kwargs,
            beta=0.1,
            loss_type="sigmoid",
            learning_rate=learning_rate,
        )

        if peft_tier == "full_finetune":
            return strategy_config, None

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
        peft_approach: Literal["full_finetune", "lora_bf16", "qlora_8bit", "qlora_4bit"],
        output_dir: str,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict]] = None,
        peft_config: Optional[PeftConfig] = None,
        callbacks: Optional[list] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> DPOTrainer:
        self.validate_data_format(dataset)
        dataset = self.normalize_conversational_pairs(dataset)
        from utilities.chat_format import ensure_chat_template, map_evaluation, normalize_preference
        eval_dataset = map_evaluation(eval_dataset, normalize_preference)

        dpo_config = self.config if self.config is not None else DPOConfig(output_dir=output_dir)

        model_init_kwargs = dpo_config.model_init_kwargs or {}
        task_class: type[FinetuningTask] = TaskModels.get_task_class(task_slug)
        model = provider.load_model(model_name, task_class.model_class, **model_init_kwargs)
        tokenizer = ensure_chat_template(provider.load_tokenizer(model_name))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # For full fine-tuning, a frozen reference model must be loaded separately.
        # For PEFT, DPOTrainer manages the reference implicitly from the frozen base.
        ref_model = None
        if peft_approach == "full_finetune":
            ref_model = provider.load_model(model_name, task_class.model_class)
            for param in ref_model.parameters():
                param.requires_grad = False

        # The provider has already consumed these loading options. Keep the
        # saved strategy config intact, but don't ask TRL to apply them again
        # to an instantiated model.
        from copy import copy
        dpo_config = copy(dpo_config)
        dpo_config.model_init_kwargs = None
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
            peft_config=peft_config,
        )

        self.launch_tensorboard(dpo_config, trainer=trainer)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        if peft_approach != "full_finetune":
            trainer.model.save_pretrained(output_dir)
        else:
            trainer.save_model(output_dir)

        tokenizer.save_pretrained(output_dir)
        return trainer
