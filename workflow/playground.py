"""In-browser inference against a trained adapter.

The previous version of ModelForge opened a playground by spawning a terminal
emulator on the server (`x-terminal-emulator -e python chat_playground.py`).
That works on a desktop and nowhere else -- not over SSH, not in a container,
not on the remote GPU box most people actually train on. This serves the same
capability over the existing websocket transport instead.

The important constraint is physical: a training run already holds the GPU.
Loading a second copy of a model beside it is the fastest way to OOM a machine
mid-run and lose hours of work, so sessions are refused while a job is active.
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Any, Iterator


class PlaygroundUnavailable(RuntimeError):
    """Raised when a session cannot be opened. The message is shown to the user."""


class PlaygroundSession:
    """One loaded model, reusable across messages."""

    def __init__(self, session_id: str, model_path: str) -> None:
        self.id = session_id
        self.model_path = model_path
        self.base_model: str | None = None
        self._lock = threading.Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._task_slug = "causal_language_modeling"
        self._stop = threading.Event()
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoTokenizer

        path = Path(self.model_path)
        if not path.exists():
            raise PlaygroundUnavailable(f"Nothing found at {self.model_path}.")

        descriptor: dict[str, Any] = {}
        descriptor_path = path / "modelforge_config.json"
        if descriptor_path.exists():
            try:
                descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                descriptor = {}

        self._task_slug = descriptor.get("task_slug", self._task_slug)
        model_classes = {
            "question_answering": "AutoModelForQuestionAnswering",
            "causal_language_modeling": "AutoModelForCausalLM",
            "seq2seq_modeling": "AutoModelForSeq2SeqLM",
            "token_classification": "AutoModelForTokenClassification",
        }
        if self._task_slug not in model_classes:
            raise PlaygroundUnavailable(
                f"The saved model has unsupported task '{self._task_slug}'."
            )

        adapter_config = path / "adapter_config.json"
        try:
            if adapter_config.exists():
                import peft

                self.base_model = peft.PeftConfig.from_pretrained(self.model_path).base_model_name_or_path
                peft_class_name = {
                    "question_answering": "AutoPeftModelForQuestionAnswering",
                    "causal_language_modeling": "AutoPeftModelForCausalLM",
                    "seq2seq_modeling": "AutoPeftModelForSeq2SeqLM",
                    "token_classification": "AutoPeftModelForTokenClassification",
                }[self._task_slug]
                peft_class = getattr(peft, peft_class_name, None)
                if peft_class is None:
                    raise PlaygroundUnavailable(
                        f"This PEFT installation cannot load {self._task_slug} adapters."
                    )
                self._model = peft_class.from_pretrained(
                    self.model_path,
                    dtype="auto",
                    device_map="auto",
                )
            else:
                from transformers import (
                    AutoModelForQuestionAnswering,
                    AutoModelForCausalLM,
                    AutoModelForSeq2SeqLM,
                    AutoModelForTokenClassification,
                )

                self.base_model = descriptor.get("base_model") or self.model_path
                model_class = {
                    "question_answering": AutoModelForQuestionAnswering,
                    "causal_language_modeling": AutoModelForCausalLM,
                    "seq2seq_modeling": AutoModelForSeq2SeqLM,
                    "token_classification": AutoModelForTokenClassification,
                }[self._task_slug]
                self._model = model_class.from_pretrained(
                    self.model_path,
                    dtype="auto",
                    device_map="auto",
                )
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the user
            raise PlaygroundUnavailable(f"Could not load this model: {exc}") from exc

        tokenizer_source = self.base_model or self.model_path
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception:  # noqa: BLE001 - adapters often ship no tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model.eval()
        if not torch.cuda.is_available():
            # Generation still works, just slowly; say so rather than hanging silently.
            self.device_note = "No GPU detected. Responses will be slow."
        else:
            self.device_note = None

    def stop(self) -> None:
        self._stop.set()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Iterator[str]:
        """Yield generated text incrementally, so the UI can stream it."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        with self._lock:
            self._stop.clear()
            tokenizer = self._tokenizer

            if self._task_slug == "question_answering":
                from utilities.qa_inference import parse_question, decode_answer
                question, context = parse_question(messages[-1]["content"])
                limit = min(tokenizer.model_max_length, getattr(self._model.config, "max_position_embeddings", 4096), 4096)
                encoding = tokenizer(question, context, return_tensors="pt", return_offsets_mapping=True,
                                     truncation="only_second", max_length=limit)
                inputs = {key: value.to(self._model.device) for key, value in encoding.items() if key != "offset_mapping"}
                with torch.inference_mode():
                    result = self._model(**inputs)
                yield decode_answer(result, encoding, context)
                return

            if self._task_slug == "causal_language_modeling":
                from utilities.chat_format import chat_inputs
                inputs = chat_inputs(tokenizer, messages).to(self._model.device)
            else:
                inputs = tokenizer(messages[-1]["content"], return_tensors="pt").to(self._model.device)
            if self._task_slug == "token_classification":
                with torch.no_grad():
                    logits = self._model(**inputs).logits[0]
                labels = getattr(self._model.config, "id2label", {})
                ids = logits.argmax(dim=-1).tolist()
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
                text = "\n".join(
                    f"{token}: {labels.get(label, str(label))}"
                    for token, label in zip(tokens, ids)
                    if token not in tokenizer.all_special_tokens
                )
                if not self._stop.is_set() and text:
                    yield text
                return

            class StopOnRequest(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return self_stop.is_set()

            self_stop = self._stop
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnRequest()]),
            )

            errors: list[Exception] = []

            def run_generation() -> None:
                try:
                    self._model.generate(**kwargs)
                except Exception as exc:  # Let the websocket surface generation failures.
                    errors.append(exc)
                    streamer.on_finalized_text("", stream_end=True)

            thread = threading.Thread(
                target=run_generation, daemon=True
            )
            with torch.no_grad():
                thread.start()
                for chunk in streamer:
                    if self._stop.is_set():
                        break
                    if chunk:
                        yield chunk
            # Do not unload a session while its generation worker still owns
            # model tensors.  StoppingCriteria makes a user stop request end
            # at the next decoding step.
            thread.join()
            if errors:
                raise errors[0]

    def close(self) -> None:
        self.stop()
        # `generate()` holds this lock until its worker has stopped.  Waiting
        # here avoids clearing model state while a cancellation is in flight.
        with self._lock:
            self._model = None
            self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


class PlaygroundManager:
    """Holds at most one loaded model, because there is only one GPU budget."""

    def __init__(self, jobs: Any) -> None:
        self._jobs = jobs
        self._session: PlaygroundSession | None = None
        self._lock = threading.Lock()
        self._holds_resource_gate = False

    @property
    def _resource_gate(self):
        return getattr(self._jobs, "resource_gate", None)

    def _training_active(self) -> bool:
        threads = getattr(self._jobs, "_threads", {})
        return any(t.is_alive() for t in threads.values())

    def availability(self) -> tuple[bool, str | None]:
        if self._training_active():
            return False, (
                "A training run is using the GPU. Loading a model beside it would run the "
                "card out of memory, so the playground stays closed until the run finishes."
            )
        with self._lock:
            if self._session is not None:
                return True, None
        gate = self._resource_gate
        if gate is not None and not gate.acquire(blocking=False):
            return False, "Model resources are busy. Try again when the current request finishes."
        if gate is not None:
            gate.release()
        return True, None

    def open(self, model_path: str) -> PlaygroundSession:
        with self._lock:
            if self._session is not None:
                if self._session.model_path == model_path:
                    return self._session
                self._session.close()
                self._session = None
                self._release_resource_gate()

            gate = self._resource_gate
            if gate is not None and not gate.acquire(blocking=False):
                raise PlaygroundUnavailable(
                    "Model resources are busy. Try again when the current request finishes."
                )
            # Record ownership immediately so every failure path below releases
            # the admission lock, including a training-race detected here.
            self._holds_resource_gate = gate is not None
            try:
                if self._training_active():
                    raise PlaygroundUnavailable(
                        "A training run is using the GPU. Loading a model beside it would run the "
                        "card out of memory, so the playground stays closed until the run finishes."
                    )
                session = PlaygroundSession(uuid.uuid4().hex, model_path)
                self._session = session
                return session
            except Exception:
                self._release_resource_gate()
                raise

    def get(self, session_id: str) -> PlaygroundSession | None:
        session = self._session
        return session if session and session.id == session_id else None

    @property
    def current(self) -> PlaygroundSession | None:
        return self._session

    def close(self, session_id: str) -> bool:
        with self._lock:
            if self._session and self._session.id == session_id:
                self._session.close()
                self._session = None
                self._release_resource_gate()
                return True
            return False

    def _release_resource_gate(self) -> None:
        if not self._holds_resource_gate:
            return
        gate = self._resource_gate
        self._holds_resource_gate = False
        if gate is not None:
            gate.release()
