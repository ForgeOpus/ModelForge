from __future__ import annotations

import os
import shlex
import re
import uuid
from pathlib import Path


DEFAULT_TENSORBOARD_SUBDIR = "tensorboard"
DEFAULT_TENSORBOARD_HOST = "127.0.0.1"
DEFAULT_TENSORBOARD_PORT = 6006


def default_log_dir(output_dir: str) -> str:
    return str(Path(output_dir) / DEFAULT_TENSORBOARD_SUBDIR)


def viewer_log_dir(log_dir: str) -> str:
    path = Path(log_dir)
    return str(path.parent if re.fullmatch(r"run-(?:\d+|[0-9a-f]{32})", path.name) else path)


def isolated_log_dir(log_dir: str, run_id: int | None = None) -> str:
    """Each launch owns its events, including retries/resumes of earlier runs."""
    return str(Path(viewer_log_dir(log_dir)) / f"run-{run_id if run_id is not None else uuid.uuid4().hex}")


def launch_command(
    log_dir: str,
    host: str = DEFAULT_TENSORBOARD_HOST,
    port: int = DEFAULT_TENSORBOARD_PORT,
) -> str:
    quoted_log_dir = shlex.quote(os.fspath(log_dir))
    quoted_host = shlex.quote(host)
    return f"tensorboard --logdir {quoted_log_dir} --host {quoted_host} --port {port}"
