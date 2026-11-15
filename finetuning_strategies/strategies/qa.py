"""Native extractive-question-answering fine-tuning."""
from __future__ import annotations

from typing import Any, Dict, Optional

from datasets import Dataset, IterableDataset
from pydantic import BaseModel
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from finetuning_strategies.model_factory import TaskModels
from finetuning_strategies.strategies.strategy import FinetuningStrategy
from providers.provider import Provider
from utilities.hardware_detector import HardwareDetector
from utilities.precision import training_precision
from recommendation_engine.validation_models.recommendation import RecommendedModel
from finetuning_strategies.strategies.sft import LoraRecommendedConfig


class QARecommendedConfig(BaseModel):
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_length: int
    doc_stride: int
    learning_rate: float
    num_train_epochs: float = 3.0
    logging_steps: int = 10
    save_steps: int = 500
    bf16: bool = False
    fp16: bool = False
    model_init_kwargs: dict


class ExtractiveQA(FinetuningStrategy):
    name = "Extractive Question Answering"
    slug = "qa"
    description = "Fine-tunes a span-prediction model on context, question, and answer offsets."
    supported_providers = ["huggingface"]
    supported_tasks = ["question_answering"]
    extra_model_copies = 0
    batch_multiplier = 1.0

    def __init__(self):
        self.config: dict[str, Any] = {}

    @staticmethod
    def validate_data_format(dataset: Dataset | IterableDataset) -> bool:
        columns = set(dataset.column_names)
        required = {"context", "question", "answers"}
        missing = required - columns
        if missing:
            raise ValueError(
                "Extractive QA datasets require context, question, and answers columns. "
                f"Missing: {', '.join(sorted(missing))}."
            )
        return True

    def set_config(self, config: Dict[str, Any]) -> None:
        self.config.update({key: value for key, value in config.items() if value is not None})

    @classmethod
    def build_recommended_config(cls, peft_tier: str, hardware: HardwareDetector.HardwareSpecs,
                                 chosen_model: RecommendedModel):
        raw_vram = sum(g["total_memory_gb"] for g in hardware.gpus if isinstance(g["total_memory_gb"], (int, float)))
        max_length = 256 if raw_vram < 12 else 384
        bf16, fp16 = training_precision(hardware)
        if peft_tier == "qlora_4bit":
            model_init_kwargs = {"quantization_config": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True}}
        elif peft_tier == "qlora_8bit":
            model_init_kwargs = {"quantization_config": {"load_in_8bit": True}}
        else:
            model_init_kwargs = {}
        if raw_vram == 0:
            model_init_kwargs = {}
        return QARecommendedConfig(
            per_device_train_batch_size=4 if raw_vram >= 12 else 1,
            gradient_accumulation_steps=1 if raw_vram >= 12 else 8,
            gradient_checkpointing=raw_vram < 12,
            max_length=max_length,
            doc_stride=128,
            learning_rate=3e-5,
            bf16=bf16,
            fp16=fp16,
            model_init_kwargs=model_init_kwargs,
        ), (None if peft_tier == "full_finetune" else LoraRecommendedConfig(
            r=16 if chosen_model.num_parameters < 7_000_000_000 else 32,
            lora_alpha=32 if chosen_model.num_parameters < 7_000_000_000 else 64,
            target_modules="all-linear",
        ))

    @staticmethod
    def _prepare_features(tokenizer, batch, max_length: int, doc_stride: int):
        questions = [question.lstrip() for question in batch["question"]]
        tokenized = tokenizer(
            questions,
            batch["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        starts, ends = [], []
        for feature_index, offsets in enumerate(offset_mapping):
            sample_index = sample_mapping[feature_index]
            answers = batch["answers"][sample_index]
            answer_starts = answers.get("answer_start", [])
            answer_texts = answers.get("text", [])
            if isinstance(answer_starts, int):
                answer_starts = [answer_starts]
            if isinstance(answer_texts, str):
                answer_texts = [answer_texts]
            sequence_ids = tokenized.sequence_ids(feature_index)
            cls_index = tokenized["input_ids"][feature_index].index(tokenizer.cls_token_id)
            if not answer_starts or not answer_texts:
                starts.append(cls_index)
                ends.append(cls_index)
                continue
            start_char = answer_starts[0]
            end_char = start_char + len(answer_texts[0])
            context_indices = [i for i, sequence_id in enumerate(sequence_ids) if sequence_id == 1]
            context_start, context_end = context_indices[0], context_indices[-1]
            if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
                starts.append(cls_index)
                ends.append(cls_index)
                continue
            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= start_char:
                token_start += 1
            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= end_char:
                token_end -= 1
            starts.append(token_start - 1)
            ends.append(token_end + 1)
        tokenized["start_positions"] = starts
        tokenized["end_positions"] = ends
        return tokenized

    def finetune(self, model_name: str, dataset: Dataset | IterableDataset, task_slug: str,
                 provider: Provider, peft_approach: str, output_dir: str, peft_config=None,
                 callbacks=None, resume_from_checkpoint: str | None = None) -> Trainer:
        self.validate_data_format(dataset)
        task_class = TaskModels.get_task_class(task_slug)
        model = provider.load_model(
            model_name, task_class.model_class,
            **self.config.get("model_init_kwargs", {}),
        )
        tokenizer = provider.load_tokenizer(model_name)
        if tokenizer.cls_token_id is None:
            raise ValueError("Extractive QA requires a tokenizer with a CLS token.")
        max_length = self.config.get("max_length", 384)
        doc_stride = self.config.get("doc_stride", 128)
        prepare = lambda batch: self._prepare_features(tokenizer, batch, max_length, doc_stride)
        dataset = dataset.map(prepare, batched=True, remove_columns=dataset.column_names)
        if peft_config is not None:
            from peft import get_peft_model, prepare_model_for_kbit_training
            if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
            gradient_checkpointing=self.config.get("gradient_checkpointing", False),
            learning_rate=self.config.get("learning_rate", 3e-5),
            num_train_epochs=self.config.get("num_train_epochs", 3.0),
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 500),
            bf16=self.config.get("bf16", False),
            fp16=self.config.get("fp16", False),
            report_to=self.config.get("report_to", "none"),
            logging_dir=self.config.get("logging_dir"),
            fsdp=self.config.get("fsdp", ""),
            fsdp_config=self.config.get("fsdp_config"),
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset,
                          processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                          callbacks=callbacks)
        self.launch_tensorboard(args, trainer=trainer)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(output_dir)
        return trainer
