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

        adapter_config = path / "adapter_config.json"
        try:
            if adapter_config.exists():
                from peft import AutoPeftModelForCausalLM, PeftConfig

                self.base_model = PeftConfig.from_pretrained(self.model_path).base_model_name_or_path
                self._model = AutoPeftModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype="auto",
                    device_map="auto",
                )
            else:
                from transformers import AutoModelForCausalLM

                self.base_model = descriptor.get("base_model") or self.model_path
                self._model = AutoModelForCausalLM.from_pretrained(
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
        from transformers import TextIteratorStreamer

        with self._lock:
            self._stop.clear()
            tokenizer = self._tokenizer

            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:  # noqa: BLE001 - base models often have no chat template
                prompt = "\n\n".join(m["content"] for m in messages) + "\n"

            inputs = tokenizer(prompt, return_tensors="pt").to(self._model.device)
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
            )

            thread = threading.Thread(
                target=lambda: self._model.generate(**kwargs), daemon=True
            )
            with torch.no_grad():
                thread.start()
                for chunk in streamer:
                    if self._stop.is_set():
                        break
                    if chunk:
                        yield chunk
            thread.join(timeout=1)

    def close(self) -> None:
        self.stop()
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

    def _training_active(self) -> bool:
        threads = getattr(self._jobs, "_threads", {})
        return any(t.is_alive() for t in threads.values())

    def availability(self) -> tuple[bool, str | None]:
        if self._training_active():
            return False, (
                "A training run is using the GPU. Loading a model beside it would run the "
                "card out of memory, so the playground stays closed until the run finishes."
            )
        return True, None

    def open(self, model_path: str) -> PlaygroundSession:
        available, reason = self.availability()
        if not available:
            raise PlaygroundUnavailable(reason or "The playground is unavailable.")
        with self._lock:
            if self._session is not None:
                if self._session.model_path == model_path:
                    return self._session
                self._session.close()
                self._session = None
            session = PlaygroundSession(uuid.uuid4().hex, model_path)
            self._session = session
            return session

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
                return True
            return False
