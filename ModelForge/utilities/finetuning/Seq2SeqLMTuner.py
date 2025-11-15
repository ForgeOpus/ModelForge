from typing import Dict
import torch
from datasets import load_dataset

# Import unsloth first to prevent EOS token corruption
# This must come before TRL imports to ensure proper tokenizer initialization
try:
    import unsloth
except ImportError:
    pass

from trl import SFTTrainer, SFTConfig
from .Finetuner import Finetuner, ProgressCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import os
from huggingface_hub import errors as hf_errors
import traceback

class Seq2SeqFinetuner(Finetuner):
    def __init__(self, model_name: str, compute_specs="low_end", pipeline_task="summarization") -> None:
        super().__init__(model_name, compute_specs, pipeline_task)
        self.task = TaskType.SEQ_2_SEQ_LM

    @staticmethod
    def format_example(example: dict, specs: str, keys=None) -> Dict | None:
        # Concatenate the context, question, and answer into a single text field.
        if keys is None:
            keys = ["article", "summary"]
            
        # Format is the same regardless of specs, so we can simplify
        return {
            "text": f'''
                ["role": "system", "content": "You are a text summarization assistant."],
                ["role": "user", "content": {example[keys[0]]}],
                ["role": "assistant", "content": {example[keys[1]]}]
            '''
        }

    def load_dataset(self, dataset_path: str) -> None:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        keys = dataset.column_names
        dataset = dataset.map(lambda example: self.format_example(example, self.compute_specs, keys))
        dataset = dataset.remove_columns(keys)
        print(dataset[0])
        self.dataset = dataset

    def set_settings(self, **kwargs) -> None:
        """
        Set model training settings from keyword arguments.
        Groups settings by category for better organization.
        """
        super().set_settings(**kwargs)


    def finetune(self) -> bool | str:
        print("Starting Seq2Seq fine-tuning process...")
        try:
            bits_n_bytes_config = None
            if self.use_4bit:
                bits_n_bytes_config = BitsAndBytesConfig(
                    load_in_4bit=self.use_4bit,
                    bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=self.use_nested_quant,
                )
            elif self.use_8bit:
                bits_n_bytes_config = BitsAndBytesConfig(
                    load_in_8bit=self.use_8bit,
                )
            if self.use_4bit or self.use_8bit:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    quantization_config=bits_n_bytes_config,
                    device_map=self.device_map,
                    use_cache=False,
                    num_processes=1
                )
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    device_map=self.device_map,
                    use_cache=False,
                    num_processes=1
                )
            print(model.dtype)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            tokenizer.max_seq_length = tokenizer.model_max_length

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                target_modules='all-linear',
            )

            training_args = SFTConfig(
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
                max_steps=self.max_steps,
                group_by_length=self.group_by_length,
                lr_scheduler_type=self.lr_scheduler_type,
                report_to="tensorboard",
                logging_dir=self.logging_dir,
                max_length=None
            )

            model = get_peft_model(model, peft_config)

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=self.dataset,
                callbacks=[ProgressCallback()],
                dataset_num_proc=1,  # Stabilize preprocessing (single process)
            )

            trainer.train()
            trainer.model.save_pretrained(self.fine_tuned_name)
            print(f"Model saved to: {self.fine_tuned_name}")
            modelforge_config_file = os.path.abspath(self.fine_tuned_name)
            config_file_result = self.build_config_file(modelforge_config_file, self.pipeline_task,
                                                        "AutoPeftModelForSeq2SeqLM")
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
            print(f"Error during fine-tuning: {e}")
            return False