"""Smoke-test a built or installed ModelForge distribution.

``--archives dist`` checks the sdist and wheel produced by
``scripts/build_distribution.sh``. With no arguments, the script must be run
with the Python from a clean environment where the wheel is installed. It
checks console entry points, the packaged frontend bundle, and the FastAPI
routes served by ``modelforge run``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_REF = re.compile(r'(?:src|href)="([^"]+)"')


def assert_archive_has_bundle(names: list[str], label: str) -> None:
    has_index = any(name.endswith("web/frontend_dist/index.html") for name in names)
    has_js = any("frontend_dist/assets/" in name and name.endswith(".js") for name in names)
    if not has_index or not has_js:
        raise SystemExit(
            f"{label} is an incomplete distribution: it must contain "
            "web/frontend_dist/index.html and web/frontend_dist/assets/*.js"
        )


def assert_distribution_archives(dist_dir: Path) -> None:
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(
            f"expected exactly one wheel and one sdist in {dist_dir}, "
            f"found wheels={[path.name for path in wheels]} "
            f"sdists={[path.name for path in sdists]}"
        )
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert_archive_has_bundle(wheel.namelist(), wheels[0].name)
    with tarfile.open(sdists[0]) as sdist:
        assert_archive_has_bundle(sdist.getnames(), sdists[0].name)


def assert_frontend_bundle(dist: Path) -> None:
    index = dist / "index.html"
    if not index.is_file() or index.stat().st_size == 0:
        raise SystemExit(f"Installed package is missing the frontend bundle: {index}")
    text = index.read_text(encoding="utf-8")
    if "ModelForge" not in text:
        raise SystemExit("Installed frontend index.html does not look like the ModelForge app.")
    refs = [
        ref
        for ref in ASSET_REF.findall(text)
        if not ref.startswith(("http://", "https://", "data:", "//"))
    ]
    if not any(ref.startswith("/assets/") and ref.endswith(".js") for ref in refs):
        raise SystemExit(f"Installed index.html does not reference a JS bundle: {refs}")
    for ref in refs:
        target = dist / ref.lstrip("/")
        if not target.is_file() or target.stat().st_size == 0:
            raise SystemExit(f"Installed frontend bundle references a missing file: {ref}")


def _drop_source_tree_from_path() -> None:
    """Import the installed wheel, not this checkout, even if cwd is the repo."""
    repo = REPO_ROOT.resolve()
    script_dir = Path(__file__).resolve().parent
    cleaned: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            continue
        if resolved == repo or resolved == script_dir:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


def assert_installed_package() -> Path:
    _drop_source_tree_from_path()
    from importlib.metadata import entry_points, files

    scripts = {entry.name: entry.value for entry in entry_points(group="console_scripts")}
    if scripts.get("modelforge") != "cli.main:cli":
        raise SystemExit(f"modelforge console script is missing or wrong: {scripts.get('modelforge')}")
    if scripts.get("modelforge-nb") != "notebook_cli.main:main":
        raise SystemExit(
            f"modelforge-nb console script is missing or wrong: {scripts.get('modelforge-nb')}"
        )

    packaged = [str(path) for path in files("modelforge-finetuning") or []]
    assert_archive_has_bundle(packaged, "installed modelforge-finetuning")

    import web.app as app_module

    app_file = Path(app_module.__file__).resolve()
    if REPO_ROOT.resolve() in app_file.parents:
        raise SystemExit(f"imported the source tree at {app_file}, not the installed wheel")
    dist = app_module.frontend_dist_path()
    assert_frontend_bundle(dist)
    return dist


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, str]:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        return exc.code, body


def assert_running_app() -> None:
    port = _free_port()
    token = "ci-smoke-token"
    command = [
        "modelforge",
        "run",
        "--no-browser",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--token",
        token,
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    log = tempfile.TemporaryFile(mode="w+")
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 180
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log.seek(0)
                raise SystemExit(
                    f"modelforge run exited before serving ({process.returncode}):\n{log.read()}"
                )
            try:
                status, _body = _get(f"{base}/api/health")
                if status == 200:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
            time.sleep(0.5)
        else:
            raise SystemExit(f"modelforge run did not become healthy: {last_error}")

        for path in ("/", "/app"):
            status, body = _get(base + path)
            if status != 200 or "ModelForge" not in body:
                raise SystemExit(f"{path} did not serve the frontend (status {status}): {body[:300]}")

        status, body = _get(f"{base}/api/use-cases")
        if status != 401:
            raise SystemExit(f"authenticated route accepted a missing token (status {status})")

        status, body = _get(f"{base}/api/use-cases", {"X-ModelForge-Token": token})
        if status != 200:
            raise SystemExit(f"/api/use-cases failed with status {status}: {body[:300]}")
        payload = json.loads(body)
        if "use_cases" not in payload:
            raise SystemExit(f"/api/use-cases did not return use cases: {body[:300]}")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)


def assert_cli_entry_points() -> None:
    help_result = subprocess.run(["modelforge", "--help"], check=False, text=True, capture_output=True)
    if help_result.returncode != 0 or "usage: modelforge" not in help_result.stdout:
        raise SystemExit(f"modelforge --help failed:\n{help_result.stdout}\n{help_result.stderr}")

    tasks = subprocess.run(["modelforge", "list-tasks"], check=False, text=True, capture_output=True)
    if tasks.returncode != 0 or not tasks.stdout.strip():
        raise SystemExit(f"modelforge list-tasks failed:\n{tasks.stdout}\n{tasks.stderr}")

    history = subprocess.run(["modelforge", "history"], check=False, text=True, capture_output=True)
    if history.returncode != 0:
        raise SystemExit(f"modelforge history failed:\n{history.stdout}\n{history.stderr}")

    notebook = subprocess.run(["modelforge-nb"], input="3\n", check=False, text=True, capture_output=True)
    if notebook.returncode != 0:
        raise SystemExit(f"modelforge-nb failed:\n{notebook.stdout}\n{notebook.stderr}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archives",
        type=Path,
        help="Check the sdist and wheel in this directory instead of an installed environment.",
    )
    args = parser.parse_args(argv)
    if args.archives is not None:
        assert_distribution_archives(args.archives)
        print(f"Distribution archives in {args.archives} contain the frontend bundle.")
        return
    assert_installed_package()
    assert_cli_entry_points()
    assert_running_app()
    print("Installed wheel smoke test passed.")


if __name__ == "__main__":
    main()
