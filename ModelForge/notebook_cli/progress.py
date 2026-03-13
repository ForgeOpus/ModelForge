"""
Notebook-aware progress display for ModelForge CLI.

Automatically uses ipywidgets when running inside a Jupyter/Colab/VSCode
notebook environment, and falls back to tqdm for plain terminal use.
"""
from __future__ import annotations


_NOTEBOOK_OVERRIDE: bool | None = None


def _is_notebook() -> bool:
    """Return True if running inside a Jupyter-compatible notebook kernel."""
    if _NOTEBOOK_OVERRIDE is not None:
        return _NOTEBOOK_OVERRIDE
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def set_notebook_mode(mode: bool) -> None:
    """Override the auto-detected notebook mode."""
    global _NOTEBOOK_OVERRIDE
    _NOTEBOOK_OVERRIDE = mode


class ProgressDisplay:
    """
    Unified progress display that works in notebooks and terminals.

    Usage::

        display = ProgressDisplay()
        display.start("Loading model...")
        display.update(50, "Halfway there")
        display.update(100, "Done!")
        display.close()
    """

    def __init__(self):
        self._notebook = _is_notebook()
        self._bar = None
        self._label = None
        self._tqdm_bar = None
        self._last_value = 0

    def start(self, message: str = "Starting...") -> None:
        """Initialise and display the progress bar."""
        self._last_value = 0
        if self._notebook:
            self._start_widget(message)
        else:
            self._start_tqdm(message)

    def _start_widget(self, message: str) -> None:
        try:
            import ipywidgets as widgets
            from IPython.display import display

            self._bar = widgets.IntProgress(
                value=0,
                min=0,
                max=100,
                description="",
                bar_style="info",
                layout=widgets.Layout(width="100%"),
            )
            self._label = widgets.Label(value=message)
            box = widgets.VBox([self._label, self._bar])
            display(box)
        except ImportError:
            # ipywidgets not available — fall through to tqdm
            self._notebook = False
            self._start_tqdm(message)

    def _start_tqdm(self, message: str) -> None:
        from tqdm import tqdm

        self._tqdm_bar = tqdm(
            total=100,
            desc=message,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}]",
            dynamic_ncols=True,
        )

    def update(self, value: int, message: str = "") -> None:
        """
        Update progress.

        Args:
            value: Progress value 0-100.
            message: Status message to display alongside the bar.
        """
        value = max(0, min(100, value))
        if self._notebook:
            if self._bar is not None:
                self._bar.value = value
                if value >= 100:
                    self._bar.bar_style = "success"
            if self._label is not None and message:
                self._label.value = message
        else:
            if self._tqdm_bar is not None:
                delta = value - self._last_value
                if delta > 0:
                    self._tqdm_bar.update(delta)
                if message:
                    self._tqdm_bar.set_description(message)
        self._last_value = value

    def close(self) -> None:
        """Close/finalize the progress display."""
        if self._notebook:
            if self._bar is not None:
                self._bar.bar_style = "success"
        else:
            if self._tqdm_bar is not None:
                self._tqdm_bar.close()
                self._tqdm_bar = None

    def error(self) -> None:
        """Mark the progress bar as errored."""
        if self._notebook:
            if self._bar is not None:
                self._bar.bar_style = "danger"
            if self._label is not None:
                self._label.value = "Training failed."
        else:
            if self._tqdm_bar is not None:
                self._tqdm_bar.set_description("Training failed")
                self._tqdm_bar.close()
                self._tqdm_bar = None
