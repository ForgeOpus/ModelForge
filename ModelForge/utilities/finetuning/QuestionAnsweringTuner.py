import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from typing import Dict
from .Finetuner import Finetuner, ProgressCallback
import os
import traceback
import huggingface_hub.errors as hf_errors

class QuestionAnsweringTuner(Finetuner):
    def __init__(self, model_name: str, compute_specs="low_end", pipeline_task="question-answering") -> None:
        super().__init__(model_name, compute_specs, pipeline_task)
        self.task = TaskType.QUESTION_ANS
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def format_example(self, example: dict, specs: str, **kwargs) -> Dict:
        question = example["question"].strip()
        context = example["context"]
        answer = example["answers"]
        inputs = self.tokenizer(
            question,
            context,
            max_length= 512 if kwargs.get("context_len") is None else kwargs.get("context_len"),
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(0)

        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1

        start_position = 0
        end_position = 0

        if not (offset_mapping[context_start][0] > end_char or
                offset_mapping[context_end][1] < start_char):

            idx = context_start
            while idx <= context_end and offset_mapping[idx][0] <= start_char:
                idx += 1
            start_position = idx - 1

            idx = context_end
            while idx >= context_start and offset_mapping[idx][1] >= end_char:
                idx -= 1
            end_position = idx + 1

        inputs["start_positions"] = start_position
        inputs["end_positions"] = end_position
        return inputs

    def load_dataset(self, dataset_path: str) -> None:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        self.dataset = dataset.map(self.format_example, remove_columns=dataset.column_names, fn_kwargs={"specs":"low"})
        print(self.dataset)

    def set_settings(self, **kwargs) -> None:
        """
        Set model training settings from keyword arguments.
        Groups settings by category for better organization.
        """
        super().set_settings(**kwargs)

    def finetune(self) -> bool | str:
        print("Starting Question Answering fine-tuning process...")
        try:
            compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
            bits_n_bytes_config = None
            if self.use_4bit:
                bits_n_bytes_config = BitsAndBytesConfig(
                    load_in_4bit=self.use_4bit,
                    bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=self.use_nested_quant,
                )
            elif self.use_8bit:
                bits_n_bytes_config = BitsAndBytesConfig(
                    load_in_8bit=self.use_8bit,
                )

            if self.use_4bit or self.use_8bit:
                model = AutoModelForQuestionAnswering.from_pretrained(
                    self.model_name,
                    quantization_config=bits_n_bytes_config,
                    device_map=self.device_map,
                    use_cache=False,
                    num_processes=1
                )
            else:
                model = AutoModelForQuestionAnswering.from_pretrained(
                    self.model_name,
                    device_map=self.device_map,
                    use_cache=False,
                    num_processes=1
                )

            peft_config = LoraConfig(
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                r=self.lora_r,
                bias="none",
                task_type=self.task,
                target_modules='all-linear',
            )

            print("Setting training args")

            training_arguments = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.num_train_epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                optim=self.optim,
                save_steps=self.save_steps,
                logging_steps=self.logging_steps,
                learning_rate=self.learning_rate,
                warmup_ratio=self.warmup_ratio,
                weight_decay=self.weight_decay,
                fp16=self.fp16,
                bf16=self.bf16,
                max_grad_norm=self.max_grad_norm,
                group_by_length=self.group_by_length,
                lr_scheduler_type=self.lr_scheduler_type,
                report_to="tensorboard",
                logging_dir=self.logging_dir,
            )

            model = get_peft_model(model, peft_config)

            print("building trainer")

            trainer = Trainer(
                model=model,
                train_dataset=self.dataset,
                args=training_arguments,
                callbacks=[ProgressCallback()],
            )
            trainer.train()
            trainer.model.save_pretrained(self.fine_tuned_name)
            modelforge_config_file = os.path.abspath(self.fine_tuned_name)
            config_file_result = self.build_config_file(modelforge_config_file, self.pipeline_task,
                                                        "AutoPeftModelForQuestionAnswering")
            if not config_file_result:
                raise Warning(
                    "Error building config file.\nRetry finetuning. This might cause problems in the model playground.")
            super().report_finish()
            return self.fine_tuned_name

        except hf_errors.GatedRepoError as e:
            super().invalid_access()
            super().report_finish(error=True, message="You do not have access to this model.")
            return False
        except hf_errors.HfHubHTTPError as e:
            print("An unknown huggingface error occurred.")
            print("Error traceback:")
            print(traceback.format_exc())
            super().report_finish(error=True, message="Unknown HuggingFace network error")
            return False
        except Exception as e:
            print(f"An error occurred during training:")
            print(traceback.format_exc())
            super().report_finish(error=True, message=e)
            return False
