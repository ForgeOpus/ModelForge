import os
import webbrowser
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple
from providers import SUPPORTED_PROVIDERS
from utilities.tensorboard import (
    DEFAULT_TENSORBOARD_HOST,
    DEFAULT_TENSORBOARD_PORT,
    default_log_dir,
    viewer_log_dir,
)
import torch


class FinetuningStrategy(ABC):
    """Abstract base for all fine-tuning strategies. Subclasses auto-register on definition.

    Required class attributes (validated in __init_subclass__):
        name: str — display name
        slug: str — unique identifier
        description: str — human-readable description
        supported_providers: List[str] — which providers can run this strategy
        supported_tasks: List[str] — task slugs this strategy supports (validated lazily)
        extra_model_copies: int — additional full model weight copies in VRAM during training
        batch_multiplier: float — multiplier on activation/batch memory budget vs SFT baseline of 1.0
    """

    name: str = None
    slug: str = None
    description: str = None

    _registry: Dict[str, type] = {}
    _tensorboard_urls: Dict[str, str] = {}

    supported_providers: Tuple[str, ...] = None

    # --- New required attributes for the recommendation engine ---
    supported_tasks: List[str] = None
    extra_model_copies: int = None
    batch_multiplier: float = None

    device: Literal["cpu", "cuda", "mps"] = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # --- Existing validations (unchanged) ---
        if hasattr(cls, "slug") and cls.slug is not None and isinstance(cls.slug, str):
            if cls.slug in FinetuningStrategy._registry:
                raise ValueError(
                    f"Duplicate slug '{cls.slug}' found for strategy class '{cls.__name__}'. "
                    f"Please ensure that each strategy class has a unique slug."
                )
            FinetuningStrategy._registry[cls.slug] = cls
        else:
            raise ValueError(f"Strategy class '{cls.__name__}' must define a 'slug' attribute")

        if cls.name is None or not isinstance(cls.name, str):
            raise ValueError(
                f"Strategy class '{cls.__name__}' must define a 'name' attribute of type string"
            )
        if cls.description is None or not isinstance(cls.description, str):
            raise ValueError(
                f"Strategy class '{cls.__name__}' must define a 'description' attribute of type string"
            )
        if cls.supported_providers is None:
            raise ValueError(
                f"Strategy class '{cls.__name__}' must define a 'supported_providers' attribute"
            )
        if not isinstance(cls.supported_providers, list) or not all(
            isinstance(p, str) for p in cls.supported_providers
        ):
            raise ValueError(
                f"Strategy class '{cls.__name__}' must have 'supported_providers' as a list of strings"
            )

        # --- New validations ---
        if cls.supported_tasks is None:
            raise ValueError(
                f"Strategy class '{cls.__name__}' must define a 'supported_tasks' attribute"
            )
        if (
            not isinstance(cls.supported_tasks, list)
            or len(cls.supported_tasks) == 0
            or not all(isinstance(t, str) for t in cls.supported_tasks)
        ):
            raise ValueError(
                f"Strategy class '{cls.__name__}' must have 'supported_tasks' as a non-empty list of strings"
            )

        if cls.extra_model_copies is None:
            raise ValueError(
                f"Strategy class '{cls.__name__}' must define an 'extra_model_copies' attribute"
            )
        if not isinstance(cls.extra_model_copies, int) or cls.extra_model_copies < 0:
            raise ValueError(
                f"Strategy class '{cls.__name__}' must have 'extra_model_copies' as an integer >= 0"
            )

        if cls.batch_multiplier is None:
            raise ValueError(
                f"Strategy class '{cls.__name__}' must define a 'batch_multiplier' attribute"
            )
        if not isinstance(cls.batch_multiplier, (int, float)) or cls.batch_multiplier <= 0.0:
            raise ValueError(
                f"Strategy class '{cls.__name__}' must have 'batch_multiplier' as a float > 0.0"
            )

    @abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError("Please add a method to configure trainer parameters for this strategy.")

    @abstractmethod
    def finetune(self, model, data, task_slug: str, provider, peft_approach: Literal["full_finetune", "lora_bf16", "qlora_8bit", "qlora_4bit"], output_dir: str, peft_config=None, callbacks=None, resume_from_checkpoint: str | None = None) -> Any:
        raise NotImplementedError("Please add an entry point method to trigger the fine-tuning process for this strategy.")

    def launch_tensorboard(
        self,
        training_args: Any | None = None,
        log_dir: str | None = None,
        host: str = DEFAULT_TENSORBOARD_HOST,
        port: int = DEFAULT_TENSORBOARD_PORT,
        enabled: bool | None = None,
        open_browser: bool = True,
        trainer: Any | None = None,
    ) -> str | None:
        """Launch TensorBoard via its Python API and return the served URL."""
        if not self._is_main_process():
            return None

        if enabled is None:
            enabled = self._reports_to_tensorboard(getattr(training_args, "report_to", None))
        if not enabled:
            return None

        resolved_log_dir = log_dir or getattr(training_args, "logging_dir", None)
        if resolved_log_dir is None:
            output_dir = getattr(training_args, "output_dir", None)
            if output_dir is None:
                return None
            resolved_log_dir = default_log_dir(output_dir)

        Path(resolved_log_dir).mkdir(parents=True, exist_ok=True)
        if trainer is not None:
            # Transformers 5's TensorBoardCallback ignores args.logging_dir.
            # Set the run's callback directly, avoiding process-global env vars
            # that could leak into a later run.
            from transformers.integrations import TensorBoardCallback

            for callback in trainer.callback_handler.callbacks:
                if isinstance(callback, TensorBoardCallback):
                    callback.logging_dir = str(Path(resolved_log_dir).resolve())
        viewer_dir = viewer_log_dir(str(resolved_log_dir))
        process_key = f"{host}:{port}:{Path(viewer_dir).resolve()}"
        existing_url = self._tensorboard_urls.get(process_key)
        if existing_url is not None:
            if open_browser:
                webbrowser.open(existing_url)
            return existing_url

        tensorboard = self._create_tensorboard_program()
        tensorboard.configure(
            argv=[
                "tensorboard",
                "--logdir",
                viewer_dir,
                "--host",
                host,
                "--port",
                str(port),
            ]
        )
        url = tensorboard.launch()
        self._tensorboard_urls[process_key] = url
        if open_browser:
            webbrowser.open(url)
        print(f"TensorBoard started at {url} for {resolved_log_dir}")
        return url

    @staticmethod
    def _reports_to_tensorboard(report_to: Any) -> bool:
        if report_to is None:
            return False
        if isinstance(report_to, str):
            return report_to.lower() in {"tensorboard", "all"}
        return any(str(target).lower() in {"tensorboard", "all"} for target in report_to)

    @staticmethod
    def _is_main_process() -> bool:
        return all(os.environ.get(name, "0") in {"", "0", "-1"} for name in ("RANK", "LOCAL_RANK", "PROCESS_INDEX"))

    @staticmethod
    def _create_tensorboard_program() -> Any:
        from tensorboard import program

        return program.TensorBoard()

    @classmethod
    @abstractmethod
    def build_recommended_config(cls, peft_tier: str, hardware: Any, chosen_model: Any) -> tuple[Any, Any | None]:
        """Build a recommended training configuration for this strategy.

        Args:
            peft_tier: One of 'full_finetune', 'lora_bf16', 'qlora_8bit', 'qlora_4bit'.
            hardware: HardwareDetector.HardwareSpecs instance.
            chosen_model: RecommendedModel instance (top-ranked Hub result).

        Returns:
            A (strategy_config, peft_config_or_None) tuple of Pydantic BaseModel instances.
            The engine stores these without inspection.
        """
        raise NotImplementedError("Strategy must implement build_recommended_config.")

    def get_supported_providers(self) -> List[str]:
        return self.supported_providers

    def get_name(self) -> str:
        return self.name
