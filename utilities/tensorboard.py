from __future__ import annotations

import os
import shlex
from pathlib import Path


DEFAULT_TENSORBOARD_SUBDIR = "tensorboard"
DEFAULT_TENSORBOARD_HOST = "127.0.0.1"
DEFAULT_TENSORBOARD_PORT = 6006


def default_log_dir(output_dir: str) -> str:
    return str(Path(output_dir) / DEFAULT_TENSORBOARD_SUBDIR)


def launch_command(
    log_dir: str,
    host: str = DEFAULT_TENSORBOARD_HOST,
    port: int = DEFAULT_TENSORBOARD_PORT,
) -> str:
    quoted_log_dir = shlex.quote(os.fspath(log_dir))
    quoted_host = shlex.quote(host)
    return f"tensorboard --logdir {quoted_log_dir} --host {quoted_host} --port {port}"
