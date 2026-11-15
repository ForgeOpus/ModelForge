"""Group Relative Policy Optimization with a safe, local reward contract.

ModelForge intentionally does not accept executable reward functions from a
dataset or request.  GRPO runs one declarative reward only: an answer earns
``1.0`` when the generated completion exactly matches that row's ``answer``
column (apart from surrounding whitespace), and otherwise earns ``0.0``.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

from datasets import Dataset, IterableDataset
from peft import PeftConfig
from pydantic import BaseModel
from trl import GRPOConfig, GRPOTrainer

from finetuning_strategies.model_factory import TaskModels
from finetuning_strategies.strategies.strategy import FinetuningStrategy
from providers.provider import Provider
from recommendation_engine.validation_models.recommendation import RecommendedModel
from utilities.hardware_detector import HardwareDetector
from utilities.precision import training_precision


def _completion_text(completion: Any) -> str:
    """Extract text from either TRL's standard or conversational completion."""
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"].strip()
    return ""


def exact_match_reward(*, completions: list[Any], answer: list[Any], **_: Any) -> list[float]:
    """Return deterministic binary rewards for the declared ``answer`` column.

    This deliberately has no dynamic import, expression evaluation, network
    access, or user-provided callback path.  It is the initial safe reward
    contract for reinforcement runs.
    """
    if len(completions) != len(answer):
        raise ValueError("GRPO reward received mismatched completion and answer counts.")
    return [
        1.0 if isinstance(expected, str) and _completion_text(actual) == expected.strip() else 0.0
        for actual, expected in zip(completions, answer, strict=True)
    ]


class GRPORecommendedConfig(BaseModel):
    """Subset of :class:`trl.GRPOConfig` selected by ModelForge."""

    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_completion_length: int
    num_generations: int
    beta: float
    use_liger_kernel: bool
    bf16: bool
    fp16: bool
    model_init_kwargs: dict
    learning_rate: float
    num_train_epochs: float = 3.0
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42


class GRPOLoraRecommendedConfig(BaseModel):
    r: int
    lora_alpha: int
    target_modules: str | list[str]


class GRPO(FinetuningStrategy):
    """Reinforcement fine-tuning for causal LMs with exact-answer rewards."""

    config: Optional[GRPOConfig] = None

    name = "Group Relative Policy Optimization"
    slug = "grpo"
    description = (
        "Optimize a causal language model from sampled answers using a deterministic "
        "exact-match reward against each dataset row's answer column."
    )
    supported_providers = ["huggingface"]
    supported_tasks = ["causal_language_modeling"]
    # GRPO uses multiple sampled completions rather than a second reference model
    # (beta=0), so generation memory rather than model copies is its main cost.
    extra_model_copies = 0
    batch_multiplier = 4.0

    @staticmethod
    def validate_data_format(dataset: Dataset | IterableDataset) -> None:
        # TRL currently rejects IterableDataset before the first optimization
        # step; make the limitation actionable at dataset validation time.
        if isinstance(dataset, IterableDataset):
            raise ValueError("GRPO does not support streaming datasets; load a finite dataset instead.")
        columns = set(dataset.column_names)
        required = {"prompt", "answer"}
        if not required.issubset(columns):
            raise ValueError(
                "GRPO datasets must contain 'prompt' and 'answer' columns. "
                "'answer' is used by ModelForge's built-in exact-match reward. "
                f"Found columns: {sorted(columns)}"
            )

    @staticmethod
    def apply_data_format(prompt: str, answer: str) -> dict[str, str]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer must be a non-empty string.")
        return {"prompt": prompt, "answer": answer}

    def build_grpo_config(
        self,
        output_dir: str = "grpo_output",
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        max_completion_length: int = 256,
        num_generations: int = 4,
        beta: float = 0.0,
        num_train_epochs: float = 3.0,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        learning_rate: float = 1e-6,
        weight_decay: float = 0.0,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "cosine",
        optim: str = "adamw_torch",
        logging_steps: int = 10,
        save_steps: int = 500,
        gradient_checkpointing: bool = True,
        use_liger_kernel: bool = False,
        bf16: bool = True,
        fp16: bool = False,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        report_to: Union[str, list[str]] = "none",
        logging_dir: Optional[str] = None,
        seed: int = 42,
        **training_kwargs: Any,
    ) -> None:
        self.config = GRPOConfig(
            output_dir=output_dir,
            model_init_kwargs=model_init_kwargs,
            max_completion_length=max_completion_length,
            num_generations=num_generations,
            beta=beta,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
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
            self.build_grpo_config(**config)
        else:
            for key, value in config.items():
                setattr(self.config, key, value)

    @classmethod
    def build_recommended_config(
        cls,
        peft_tier: str,
        hardware: HardwareDetector.HardwareSpecs,
        chosen_model: RecommendedModel,
    ) -> tuple[GRPORecommendedConfig, GRPOLoraRecommendedConfig | None]:
        raw_vram_gb = sum(
            gpu["total_memory_gb"]
            for gpu in hardware.gpus
            if isinstance(gpu["total_memory_gb"], (int, float))
        )
        if raw_vram_gb < 12:
            max_completion_length, gradient_accumulation_steps = 128, 32
        elif raw_vram_gb < 32:
            max_completion_length, gradient_accumulation_steps = 256, 16
        else:
            max_completion_length, gradient_accumulation_steps = 512, 8

        if peft_tier == "full_finetune":
            model_init_kwargs: dict[str, Any] = {}
        elif peft_tier == "lora_bf16":
            model_init_kwargs = {}
        elif peft_tier == "qlora_8bit":
            model_init_kwargs = {"quantization_config": {"load_in_8bit": True}}
        elif peft_tier == "qlora_4bit":
            model_init_kwargs = {"quantization_config": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True}}
        else:
            raise ValueError(f"Unknown peft_tier '{peft_tier}'.")

        bf16, fp16 = training_precision(hardware)
        if raw_vram_gb == 0:
            model_init_kwargs = {}

        config = GRPORecommendedConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            max_completion_length=max_completion_length,
            # GRPO needs at least two samples per prompt.  Every recommended
            # accumulation count is a multiple of four, satisfying TRL's group
            # divisibility requirement on common multi-GPU world sizes.
            num_generations=4,
            beta=0.0,
            use_liger_kernel=False,
            bf16=bf16,
            fp16=fp16,
            model_init_kwargs=model_init_kwargs,
            learning_rate=1e-6 if peft_tier == "full_finetune" else 5e-6,
        )
        if peft_tier == "full_finetune":
            return config, None

        rank = 8 if chosen_model.num_parameters < 1_000_000_000 else 16
        return config, GRPOLoraRecommendedConfig(
            r=rank,
            lora_alpha=rank * 2,
            # PEFT resolves this architecture-independently, unlike a hard
            # coded Llama-only module list.
            target_modules="all-linear",
        )

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
    ) -> GRPOTrainer:
        if task_slug not in self.supported_tasks:
            raise ValueError(f"GRPO only supports {self.supported_tasks}; got '{task_slug}'.")
        self.validate_data_format(dataset)

        config = self.config if self.config is not None else GRPOConfig(output_dir=output_dir)
        task_class = TaskModels.get_task_class(task_slug)
        model = provider.load_model(
            model_name, task_class.model_class, **(config.model_init_kwargs or {})
        )
        from utilities.chat_format import ensure_chat_template, normalize_prompts, map_evaluation
        tokenizer = ensure_chat_template(provider.load_tokenizer(model_name))
        dataset = dataset.map(normalize_prompts)
        eval_dataset = map_evaluation(eval_dataset, normalize_prompts)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=exact_match_reward,
            args=config,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
            peft_config=peft_config,
        )
        self.launch_tensorboard(config, trainer=trainer)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if peft_approach == "full_finetune":
            trainer.save_model(output_dir)
        else:
            trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        return trainer
