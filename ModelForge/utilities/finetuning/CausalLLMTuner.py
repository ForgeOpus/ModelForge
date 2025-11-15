import torch
from datasets import load_dataset

# Import unsloth first to prevent EOS token corruption
# This must come before TRL imports to ensure proper tokenizer initialization
try:
    import unsloth
except ImportError:
    pass

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Optional
from .Finetuner import Finetuner, ProgressCallback
import os
from huggingface_hub import errors as hf_errors
import traceback


class CausalLLMFinetuner(Finetuner):
    def __init__(self, model_name: str, compute_specs="low_end", pipeline_task="text-generation") -> None:
        super().__init__(model_name, compute_specs, pipeline_task)
        self.task = TaskType.CAUSAL_LM

    @staticmethod
    def format_example(example: dict, specs: str, **kwargs) -> Dict[str, Optional[str]]:
        """
        Format the example for training with the chat templates of the expected models.
        :param example: The example to be formatted.
        :param specs: The computational environment specs (low_end, mid_range, or high_end).
        :return: A dictionary with 'prompt' and 'completion' keys.
        """
        return {
            "prompt" : "USER:" + example.get("prompt", ""),
            "completion": "ASSISTANT: " + example.get("completion", "") + "<|endoftext|>"
        }

    def load_dataset(self, dataset_path:str) -> None:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        dataset = dataset.rename_column("input", "prompt")
        dataset = dataset.rename_column("output", "completion")
        dataset = dataset.map(lambda x: self.format_example(x, self.compute_specs))
        print(dataset[0])
        self.dataset = dataset

    def set_settings(self, **kwargs) -> None:
        """
        Set model training settings from keyword arguments.
        Groups settings by category for better organization.
        """
        super().set_settings(**kwargs)


    def finetune(self) -> bool | str:
        print("Starting Causal LM fine-tuning process...")
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
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bits_n_bytes_config,
                    device_map=self.device_map,
                    use_cache=False,
                    num_processes=1
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device_map,
                    use_cache=False,
                    num_processes=1
                )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            peft_config = LoraConfig(
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                r=self.lora_r,
                bias="none",
                task_type=self.task,
                target_modules='all-linear',
            )

            print("Setting training args")

            training_arguments = SFTConfig(
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
                max_length=None,
            )

            model = get_peft_model(model, peft_config)

            print("building trainer")

            trainer = SFTTrainer(
                model=model,
                train_dataset=self.dataset,
                args=training_arguments,
                callbacks=[ProgressCallback()],
                dataset_num_proc=1,  # Stabilize preprocessing (single process)
            )
            trainer.train()
            trainer.model.save_pretrained(self.fine_tuned_name)
            modelforge_config_file = os.path.abspath(self.fine_tuned_name)
            config_file_result = self.build_config_file(modelforge_config_file, self.pipeline_task,
                                                        "AutoPeftModelForCausalLM")
            if not config_file_result:
                raise Warning("Error building config file.\nRetry finetuning. This might cause problems in the model playground.")
            super().report_finish()
            return self.fine_tuned_name
        except Exception as e:
            print(f"An error occurred during training: {e}")
            super().report_finish(error=True, message=e)
            return False
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