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
    supported_providers = ["huggingface", "unsloth"]

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

    def finetune(
        self,
        model_name: str,
        dataset: Union[Dataset, IterableDataset],
        task_slug: str,
        provider: Provider,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict]] = None,
        peft_config: Optional[PeftConfig] = None,
        formatting_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[list] = None,
    ) -> SFTTrainer:
        self.validate_data_format(dataset)

        sft_config = self.config if self.config is not None else SFTConfig(output_dir="sft_output")

        model_init_kwargs = sft_config.model_init_kwargs or {}
        task_class: type[FinetuningTask] = TaskModels.get_task_class(task_slug)
        model = provider.load_model(model_name, task_class.model_class, **model_init_kwargs)
        tokenizer = provider.load_tokenizer(model_name)

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

        trainer.train()
        return trainer
