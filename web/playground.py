"""Ephemeral inference for models saved by ModelForge.

No conversations or loaded models are retained between requests. Active training
uses the CPU path so inference cannot consume the trainer's VRAM.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class InferenceRequest(BaseModel):
    run_id: int = Field(gt=0)
    checkpoint_path: str | None = None
    context_window: Literal[1024, 4096, 8016, 32768] = 4096
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_messages(self):
        if any(message.role != ("user" if index % 2 == 0 else "assistant") for index, message in enumerate(self.messages)) or self.messages[-1].role != "user":
            raise ValueError("Messages must alternate user and assistant, ending with a user message.")
        if sum(len(message.content) for message in self.messages) > 500_000:
            raise ValueError("Conversation is too large.")
        return self


class PlaygroundError(Exception):
    def __init__(self, message: str, status: int = 400):
        self.message = message
        self.status = status
        super().__init__(message)


def memory_budgets(torch, available: int, runtime_bytes: int, active: bool) -> dict:
    """Leave room for execution, the OS, and potential growth of a trainer."""
    reserve = max(8 * 1024**3 if active else 2 * 1024**3,
                  available // 2 if active else available // 4)
    cpu_budget = available - reserve - runtime_bytes
    if cpu_budget < 256 * 1024**2:
        raise PlaygroundError("Not enough available RAM for safe inference" +
                              (" while training is active." if active else "."), 409)
    budgets = {"cpu": cpu_budget}
    if not active and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(index)
            except (RuntimeError, AssertionError):
                continue
            budget = free - max(1024**3, total // 10) - runtime_bytes
            if budget >= 256 * 1024**2:
                budgets[index] = budget
    return budgets


def placement_description(device_map: dict) -> tuple[str, str]:
    """Return a user-facing placement and the device used for input tensors."""
    values = list(device_map.values())
    gpu = next((v for v in values if isinstance(v, int) or str(v).startswith("cuda:")), None)
    execution = (f"cuda:{gpu}" if isinstance(gpu, int) else str(gpu)) if gpu is not None else "cpu"
    parts = []
    if gpu is not None:
        parts.append("GPU")
    if any(str(v) == "cpu" for v in values):
        parts.append("CPU")
    if any(str(v) == "disk" for v in values):
        parts.append("disk")
    label = execution if len(parts) == 1 and parts[0] in {"GPU", "CPU"} else " + ".join(parts)
    return label, execution


class Playground:
    def __init__(self, jobs, ledger):
        self.jobs = jobs
        self.ledger = ledger

    def _artifact(self, request: InferenceRequest):
        run = self.ledger.get_run(request.run_id)
        if not run:
            raise PlaygroundError("Run not found.", 404)
        if request.checkpoint_path:
            if not any(item["path"] == request.checkpoint_path for item in self.ledger.list_checkpoints(request.run_id)):
                raise PlaygroundError("Checkpoint is not registered for this run.", 404)
            path = Path(request.checkpoint_path)
        else:
            if run["status"] != "completed":
                raise PlaygroundError("Choose a completed run or a registered checkpoint.", 409)
            path = Path(run["output_dir"])
        if not path.is_dir():
            raise PlaygroundError("The saved model directory is unavailable.", 404)
        task = run.get("task_slug")
        if task not in {"causal_language_modeling", "seq2seq_modeling", "token_classification", "question_answering"}:
            raise PlaygroundError("This run's model task is not supported by the playground.")
        return run, path, task

    def infer(self, request: InferenceRequest) -> dict:
        gate = getattr(self.jobs, "resource_gate", None)
        if gate is None:
            raise PlaygroundError("Cannot verify training resource use; inference is unavailable.", 503)
        if not gate.acquire(blocking=False):
            raise PlaygroundError("Model resources are busy. Try again when the current request finishes.", 409)
        model = tokenizer = None
        try:
            run, path, task = self._artifact(request)
            active = self.jobs.has_active_jobs()
            return self._run(request, run, path, task, active)
        finally:
            del model, tokenizer
            gc.collect()
            try:
                import torch
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            finally:
                gate.release()

    def _run(self, request, run, path, task, active):
        import psutil
        import torch
        from accelerate import init_empty_weights, infer_auto_device_map
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoModelForQuestionAnswering, AutoTokenizer

        classes = {"causal_language_modeling": AutoModelForCausalLM, "seq2seq_modeling": AutoModelForSeq2SeqLM, "token_classification": AutoModelForTokenClassification, "question_answering": AutoModelForQuestionAnswering}
        cls = classes[task]
        adapter_file = path / "adapter_config.json"
        adapter = adapter_file.is_file()
        source = run["model_id"] if adapter else str(path)
        if adapter:
            try:
                adapter_config = json.loads(adapter_file.read_text(encoding="utf-8"))
                source = adapter_config.get("base_model_name_or_path") or source
            except (OSError, ValueError) as exc:
                raise PlaygroundError(f"Cannot read adapter configuration: {exc}") from exc
        try:
            config = AutoConfig.from_pretrained(source, trust_remote_code=False)
            with init_empty_weights():
                empty = cls.from_config(config, trust_remote_code=False)
            empty.tie_weights()
            parameter_bytes = sum(p.numel() * 4 for p in empty.parameters())
        except Exception as exc:
            raise PlaygroundError(f"Cannot estimate model memory safely: {exc}") from exc
        if not parameter_bytes:
            raise PlaygroundError("Cannot estimate model memory safely.")
        # Include a conservative float32 attention matrix and KV cache. Architectures
        # without these common dimensions cannot be admitted on a guessed budget.
        layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None) or getattr(config, "num_layers", None)
        hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None) or getattr(config, "d_model", None)
        heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None) or getattr(config, "num_heads", None)
        if not all(isinstance(v, int) and v > 0 for v in (layers, hidden, heads)):
            raise PlaygroundError("Cannot estimate this model's attention memory safely.")
        attention = request.context_window ** 2 * heads * 4
        cache = request.context_window * layers * hidden * 8
        adapter_bytes = sum(item.stat().st_size for item in path.glob("adapter_model.*") if item.is_file()) if adapter else 0
        runtime_bytes = attention + cache + 1024**3 + adapter_bytes * 4
        budgets = memory_budgets(torch, psutil.virtual_memory().available, runtime_bytes, active)
        try:
            device_map = infer_auto_device_map(
                empty, max_memory=budgets, dtype=torch.float32,
                no_split_module_classes=list(empty._no_split_modules or []),
                fallback_allocation=True,
            )
        except Exception as exc:
            raise PlaygroundError(f"Cannot plan model placement safely: {exc}") from exc
        finally:
            del empty
        placement, device = placement_description(device_map)
        if active and any(isinstance(value, int) or str(value).startswith("cuda") for value in device_map.values()):
            raise PlaygroundError("Cannot safely use a GPU while training is active.", 409)
        if not placement or all(str(value) == "disk" for value in device_map.values()):
            raise PlaygroundError("Not enough memory to execute the model, even with disk offload.", 409)
        uses_disk = any(str(value) == "disk" for value in device_map.values())
        offload_root = os.environ.get("MODELFORGE_OFFLOAD_DIR") or tempfile.gettempdir()
        if uses_disk:
            try:
                free_disk = shutil.disk_usage(offload_root).free
            except OSError as exc:
                raise PlaygroundError(f"Cannot access the model offload directory: {exc}", 503) from exc
            if free_disk < (parameter_bytes + adapter_bytes) * 2 + 1024**3:
                raise PlaygroundError("Not enough disk space for safe model offload.", 409)
        offload = None
        try:
            tokenizer_source = str(path) if (path / "tokenizer_config.json").exists() else source
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
            model_limit = getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions", None)
            token_limit = getattr(tokenizer, "model_max_length", None)
            limits = [v for v in (model_limit, token_limit) if isinstance(v, int) and 0 < v < 10_000_000]
            context_limit = min([request.context_window, *limits])
            prompt = request.messages[-1].content
            encoding = None
            if task == "question_answering":
                from utilities.qa_inference import parse_question, decode_answer
                try:
                    question, context = parse_question(request.messages[-1].content)
                except ValueError as exc:
                    raise PlaygroundError(str(exc), 422) from exc
                encoding = tokenizer(question, context, return_tensors="pt", return_offsets_mapping=True, truncation=False)
                inputs = {key: value for key, value in encoding.items() if key != "offset_mapping"}
            elif task == "causal_language_modeling":
                from utilities.chat_format import chat_inputs
                inputs = chat_inputs(tokenizer, [m.model_dump() for m in request.messages], truncation=False)
            else:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            input_tokens = inputs["input_ids"].shape[-1]
            conversation_tokens = input_tokens
            if task == "token_classification":
                history = "\n".join(message.content for message in request.messages)
                conversation_tokens = tokenizer(history, return_tensors="pt", truncation=False)["input_ids"].shape[-1]
            needed = conversation_tokens + (0 if task in {"token_classification", "question_answering"} else request.max_new_tokens)
            if needed > context_limit:
                raise PlaygroundError(f"This conversation needs {needed} tokens, exceeding the {context_limit}-token limit. Start a new chat or choose a larger supported context.", 422)
            offload = tempfile.TemporaryDirectory(prefix="modelforge-offload-", dir=offload_root) if uses_disk else None
            offload_path = offload.name if offload else None
            # Load the declared base explicitly. Reused output folders can
            # contain a stale full-model config/weights alongside a new adapter;
            # passing that mixed directory to AutoModel can load the old model.
            model = cls.from_pretrained(source, config=config, trust_remote_code=False, torch_dtype=torch.float32, device_map=device_map, offload_folder=offload_path)
            if adapter:
                model.load_adapter(str(path), is_trainable=False,
                                   device_map=device_map, disk_offload_folder=offload_path)
            model.eval()
            placement, device = placement_description(getattr(model, "hf_device_map", device_map))
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                if task == "question_answering":
                    result = model(**inputs)
                    text = decode_answer(result, encoding, context)
                elif task == "token_classification":
                    logits = model(**inputs).logits[0]
                    ids = logits.argmax(dim=-1).tolist()
                    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
                    labels = getattr(model.config, "id2label", {})
                    text = "\n".join(f"{token}: {labels.get(label, str(label))}" for token, label in zip(tokens, ids) if token not in tokenizer.all_special_tokens)
                else:
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    result = model.generate(**inputs, max_new_tokens=request.max_new_tokens, do_sample=False, pad_token_id=pad_id)
                    output = result[0] if task == "seq2seq_modeling" else result[0][input_tokens:]
                    text = tokenizer.decode(output, skip_special_tokens=True).strip()
            return {"reply": text or "(No output)", "task": task, "input_tokens": conversation_tokens, "context_limit": context_limit, "device": placement, "training_active": active}
        except PlaygroundError:
            raise
        except Exception as exc:
            raise PlaygroundError(f"Inference failed: {exc}", 500) from exc
        finally:
            # Release device tensors before returning or propagating an error so
            # the resource gate is not released while they are still referenced.
            model = inputs = result = logits = output = None
            gc.collect()
            if offload is not None:
                offload.cleanup()
