from __future__ import annotations

import importlib.util
import os
import shutil
import site
import socket
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ASSETS = (
    "favicon.svg",
    "favicon.ico",
    "logo.svg",
    "Logo.jpg",
    "apple-touch-icon.png",
    "landing-mock/theme.js",
    "landing-mock/workbench.css",
    "landing-mock/live-run.html",
    "landing-mock/new-run.html",
)


def _load_builder():
    path = REPO_ROOT / "scripts" / "build_frontend.py"
    spec = importlib.util.spec_from_file_location("modelforge_frontend_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frontend_build_refuses_without_npm(monkeypatch):
    module = _load_builder()
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    with pytest.raises(module.FrontendBuildError, match="Refusing to build a wheel"):
        module.build_frontend(REPO_ROOT)


def test_frontend_build_refuses_without_lockfile(tmp_path):
    module = _load_builder()
    source = tmp_path / "incomplete"
    (source / "web" / "frontend").mkdir(parents=True)

    with pytest.raises(module.FrontendBuildError, match="package-lock.json"):
        module.build_frontend(source)


def _python313() -> str:
    if sys.version_info >= (3, 13):
        return sys.executable
    found = shutil.which("python3.13")
    if found:
        return found
    raise AssertionError(
        "The installed-wheel smoke test requires Python 3.13, matching requires-python."
    )


def _copy_clean_source(destination: Path) -> None:
    shutil.copytree(
        REPO_ROOT,
        destination,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "frontend_dist",
            "dist",
            "build",
            "*.egg-info",
            "__pycache__",
            ".pytest_cache",
        ),
    )
    assert not (destination / "web" / "frontend_dist").exists()
    assert (destination / "web" / "frontend" / "package-lock.json").is_file()


def _referenced_assets(html: str) -> list[str]:
    import re

    refs = []
    for ref in re.findall(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", html, flags=re.I):
        if ref.startswith(("http://", "https://", "data:", "mailto:", "#")):
            continue
        relative = ref.split("?", 1)[0].split("#", 1)[0].lstrip("/")
        if relative:
            refs.append(relative)
    return refs


def _has_pip(python: str, env: dict[str, str]) -> bool:
    return (
        subprocess.run(
            [python, "-m", "pip", "--version"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _build_wheel(source: Path, wheel_dir: Path, python: str, env: dict[str, str]) -> None:
    """Build a wheel. uv environments have no pip, so fall back to ``uv build``."""
    if _has_pip(python, env):
        subprocess.check_call(
            [
                python,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-cache-dir",
                "--wheel-dir",
                str(wheel_dir),
                str(source),
            ],
            cwd=source,
            env=env,
        )
        return
    uv = shutil.which("uv")
    if uv is None:
        raise AssertionError(
            f"{python} has no pip module, and uv is not on PATH. "
            "uv-created environments do not include pip."
        )
    subprocess.check_call(
        [uv, "build", "--python", python, "--wheel", "--out-dir", str(wheel_dir)],
        cwd=source,
        env=env,
    )


def _install_wheel(python: str, wheel: Path, env: dict[str, str]) -> None:
    """Install only the project wheel; CI already has its locked dependencies.

    Reinstalling the full PyTorch/CUDA dependency graph in a nested environment
    can exhaust a GitHub runner's disk before the packaging assertions run.
    """
    if _has_pip(python, env):
        subprocess.check_call(
            [python, "-m", "pip", "install", "--no-deps", "--no-cache-dir", str(wheel)],
            env=env,
        )
        return
    uv = shutil.which("uv")
    if uv is None:
        raise AssertionError(
            f"{python} has no pip module, and uv is not on PATH. "
            "Cannot install the wheel into the clean environment."
        )
    subprocess.check_call(
        [uv, "pip", "install", "--python", python, "--no-deps", str(wheel)],
        env=env,
    )


def test_wheel_build_uses_uv_when_the_environment_has_no_pip(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    monkeypatch.setattr(sys.modules[__name__], "_has_pip", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    def capture(command, **_kwargs):
        calls.append(list(command))

    monkeypatch.setattr(subprocess, "check_call", capture)
    source = tmp_path / "source"
    source.mkdir()
    _build_wheel(source, tmp_path / "wheels", sys.executable, {})

    assert calls == [
        ["/usr/bin/uv", "build", "--python", sys.executable, "--wheel", "--out-dir", str(tmp_path / "wheels")]
    ]


def test_wheel_install_does_not_reinstall_locked_dependencies(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    wheel = tmp_path / "modelforge.whl"
    wheel.write_bytes(b"placeholder")

    monkeypatch.setattr(sys.modules[__name__], "_has_pip", lambda *_args, **_kwargs: True)

    def capture(command, **_kwargs):
        calls.append(list(command))

    monkeypatch.setattr(subprocess, "check_call", capture)
    _install_wheel(sys.executable, wheel, {})

    assert calls == [[
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-cache-dir",
        str(wheel),
    ]]


def test_installed_wheel_serves_frontend(tmp_path):
    """Build a wheel from a tree with no prebuilt UI, install it, and serve it."""
    source = tmp_path / "source"
    _copy_clean_source(source)
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    python = _python313()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    _build_wheel(source, wheel_dir, python, env)
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1, wheels
    wheel = wheels[0]

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        index_name = "web/frontend_dist/index.html"
        assert index_name in names
        html = archive.read(index_name).decode("utf-8")
        assert "ModelForge" in html
        assert 'id="root"' in html
        referenced = _referenced_assets(html)
        assert referenced, "built index.html should reference hashed assets"
        for relative in referenced:
            assert f"web/frontend_dist/{relative}" in names, relative
        for relative in PUBLIC_ASSETS:
            assert f"web/frontend_dist/{relative}" in names, relative
        assert not any("node_modules" in name for name in names)
        assert not any(name.startswith("web/frontend/") for name in names)

    venv = tmp_path / "venv"
    _create_venv(python, venv, env)
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    venv_python = bindir / ("python.exe" if os.name == "nt" else "python")
    _install_wheel(str(venv_python), wheel, env)
    # Reuse the dependencies from the pytest environment without reinstalling
    # the multi-gigabyte ML/CUDA graph. The project itself still comes from the
    # wheel installed in the isolated venv.
    dependency_paths = [path for path in site.getsitepackages() if Path(path).is_dir()]
    assert dependency_paths
    env["PYTHONPATH"] = os.pathsep.join(dependency_paths)
    modelforge = bindir / ("modelforge.exe" if os.name == "nt" else "modelforge")
    assert modelforge.is_file()

    port = _free_port()
    process = subprocess.Popen(
        [
            str(modelforge),
            "run",
            "--no-browser",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--token",
            "packaging-smoke",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output: list[str] = []
    reader = threading.Thread(target=_drain, args=(process, output), daemon=True)
    reader.start()
    try:
        base = f"http://127.0.0.1:{port}"
        _wait_until_ready(process, f"{base}/", output)
        root = _get(f"{base}/")
        app = _get(f"{base}/app")
        deep_link = _get(f"{base}/app/history")
        assert root.status == 200
        assert app.status == 200
        assert deep_link.status == 200
        assert "ModelForge" in root.body
        assert app.body == root.body
        assert deep_link.body == root.body
        for relative in _referenced_assets(root.body):
            asset = _get(f"{base}/{relative}")
            assert asset.status == 200, relative
            assert asset.body
        favicon = _get(f"{base}/favicon.svg")
        assert favicon.status == 200
        assert favicon.body
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)
        reader.join(timeout=5)


def _create_venv(python: str, destination: Path, env: dict[str, str]) -> None:
    """Create a venv with pip. Fall back to uv when the interpreter has no ensurepip."""
    try:
        subprocess.check_call([python, "-m", "venv", str(destination)], env=env)
        return
    except subprocess.CalledProcessError:
        uv = shutil.which("uv")
        if uv is None:
            raise
        subprocess.check_call(
            [uv, "venv", "--python", python, "--seed", str(destination)],
            env=env,
        )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Response:
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body


def _get(url: str) -> _Response:
    with urlopen(url, timeout=30) as response:
        return _Response(response.status, response.read().decode("utf-8", errors="replace"))


def _drain(process: subprocess.Popen[str], output: list[str]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        output.append(line)


def _wait_until_ready(process: subprocess.Popen[str], url: str, output: list[str]) -> None:
    deadline = time.time() + 240
    while time.time() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"modelforge run exited {process.returncode}:\n{''.join(output)}"
            )
        try:
            response = _get(url)
        except (URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(0.5)
            continue
        if response.status == 200 and "ModelForge" in response.body:
            return
        time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for {url} to serve the UI.\n{''.join(output)}")
