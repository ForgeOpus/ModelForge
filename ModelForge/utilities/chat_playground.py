import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, TextStreamer, pipeline
from peft import PeftConfig
import argparse
import os
import json
from pydantic import BaseModel, field_validator
import traceback
import peft

class ModelForgeConfig(BaseModel):
    model_class: str
    pipeline_task: str

    @field_validator("model_class")
    def validate_model_class(cls, v):
        if v is None:
            raise ValueError("model_class must be defined")
        if not isinstance(v, str):
            raise ValueError("model_class must be a string")
    @field_validator("pipeline_task")
    def validate_pipeline_task(cls, v):
        if v is None:
            raise ValueError("pipeline_task must be defined")
        if not isinstance(v, str):
            raise ValueError("pipeline_task must be a string")

class PlaygroundModel:

    MIN_CONTEXT_LENGTH = 32

    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("CUDA is not available. Exiting...")
            exit(1)

        print("Loading model...")
        try:
            file_path = os.path.join(model_path, "modelforge_config.json")
            with open(file_path, "r") as f:
                self.modelforge_config = json.loads(f.read())

            config = PeftConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            tokenizer.pad_token = tokenizer.eos_token
            module = getattr(peft, self.modelforge_config["model_class"])
            peft_model = module.from_pretrained(model_path, config=config, is_trainable=False)
            self.generator = pipeline(
                self.modelforge_config["pipeline_task"],
                streamer=streamer,
                model=peft_model,
                tokenizer=tokenizer
            )
        except AttributeError:
            print(f"Model class {self.modelforge_config['model_class']} not found in peft module.")
            exit(1)
        except KeyError:
            print(f"Pipeline task {self.modelforge_config['pipeline_task']} not found in definitions for the pipeline object of transformers.")
            exit(1)
        except Exception as e:
            print(traceback.format_exc())
            exit(1)

    def generate_response(self, prompt: str, context=None, temperature=0.2, top_p=0.92, top_k=50,
                          repetition_penalty=1.3):
        try:
            tokenizer = self.generator.tokenizer
            model_config = self.generator.model.config
            max_length = getattr(model_config, "max_position_embeddings", 2048)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            input_len = input_ids.shape[1]
            max_new_tokens = max(self.MIN_CONTEXT_LENGTH, max_length - input_len)

            if input_len >= max_length:
                prompt = tokenizer.decode(input_ids[0, -max_length + self.MIN_CONTEXT_LENGTH:])
                print("Prompt truncated to fit model context window.")

            if context is None:
                response = self.generator(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )[0]['generated_text']
            else:
                response = self.generator(
                    question=prompt,
                    context=context,
                )["answer"]
            return response
        except Exception as e:
            print(f"Error during generation: {e}")
            return ""

    def chat(self):
        print("Chat started. Type '/bye' to exit or /view_settings to view current model configurations.")
        try:
            if self.modelforge_config["pipeline_task"] == "question-answering":
                while True:
                    context = input("Enter Context: ").strip()
                    query = input("Enter Query: ").strip()
                    if query.lower() == "/bye":
                        break
                    elif query.lower() == "/view_settings":
                        print(f"ModelForge settings:\n{self.modelforge_config}")
                        print()
                        print(f"Model configurations:\n{self.generator.model.config}")
                        print()
                        print(f"Model tokenizer configurations:\n{self.generator.tokenizer}")
                        print()
                        continue
                    elif context == "":
                        print("Context cannot be empty.")
                        continue
                    response = self.generate_response(prompt=query, context=context)
                    print(f"Assistant:", end=" ", flush=True)
                    print(response)
            else:
                while True:
                    user_input = input("You: ").strip()
                    if user_input.lower() == "/bye":
                        break
                    if user_input.lower() == "/view_settings":
                        print(f"ModelForge settings:\n{self.modelforge_config}")
                        print()
                        print(f"Model configurations:\n{self.generator.model.config}")
                        print()
                        print(f"Model tokenizer configurations:\n{self.generator.tokenizer}")
                        print()
                        continue
                    print(f"Assistant: ", end=" ", flush=True)
                    response = self.generate_response(user_input)
                    print(response)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"An error occurred: {e}")
            print(traceback.format_exc())
        finally:
            self.clean_up()

    def clean_up(self):
        if hasattr(self, 'generator'):
            del self.generator
        torch.cuda.empty_cache()
        print("Resources cleaned")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with QLoRA fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved modelforge model directory")
    args = parser.parse_args()

    bot = PlaygroundModel(model_path=args.model_path)
    bot.chat()