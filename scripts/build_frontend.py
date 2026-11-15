"""Build the browser UI into web/frontend_dist using the committed lockfile.

Package and release builds call this before collecting package data. A missing
Node.js install or a failed frontend build aborts the build so a wheel is not
published without a UI.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

_ASSET_REF = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


class FrontendBuildError(RuntimeError):
    """The frontend bundle could not be produced."""


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_frontend(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else repo_root()
    frontend = root / "web" / "frontend"
    dist = root / "web" / "frontend_dist"
    lockfile = frontend / "package-lock.json"
    if not lockfile.is_file():
        _fail(
            f"Locked frontend manifest not found at {lockfile}. "
            "Build from a source tree that includes web/frontend and package-lock.json. "
            "Refusing to build a wheel without web/frontend_dist/index.html."
        )

    npm = shutil.which("npm")
    if npm is None:
        _fail(
            "Node.js and npm are required to package the ModelForge browser UI. "
            "Install Node.js (npm is included), ensure npm is on PATH, and retry the package build. "
            "Refusing to build a wheel without web/frontend_dist/index.html."
        )

    _run(npm, "ci", cwd=frontend)
    _run(npm, "run", "build", cwd=frontend)
    index = dist / "index.html"
    if not index.is_file():
        _fail(
            "Frontend build finished but web/frontend_dist/index.html is missing. "
            "Refusing to build a wheel without the browser UI."
        )
    _assert_public_assets(frontend / "public", dist)
    _assert_referenced_assets(index, dist)
    return dist


def _assert_public_assets(public: Path, dist: Path) -> None:
    if not public.is_dir():
        _fail(f"Frontend public directory is missing: {public}.")
    missing = [
        path.relative_to(public).as_posix()
        for path in public.rglob("*")
        if path.is_file() and not (dist / path.relative_to(public)).is_file()
    ]
    if missing:
        _fail(
            "Frontend build did not copy required public assets into web/frontend_dist: "
            + ", ".join(missing)
            + ". Refusing to build a wheel without the browser UI."
        )


def _assert_referenced_assets(index: Path, dist: Path) -> None:
    missing: list[str] = []
    for ref in _ASSET_REF.findall(index.read_text(encoding="utf-8")):
        if ref.startswith(("http://", "https://", "data:", "mailto:", "#")):
            continue
        relative = ref.split("?", 1)[0].split("#", 1)[0].lstrip("/")
        if relative and not (dist / relative).is_file():
            missing.append(ref)
    if missing:
        _fail(
            "Frontend build output is incomplete. index.html references missing files: "
            + ", ".join(missing)
            + ". Refusing to build a wheel without the browser UI."
        )


def _run(npm: str, *args: str, cwd: Path) -> None:
    command = [npm, *args]
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        _fail(
            f"Frontend build failed ({' '.join(command)} exited {exc.returncode}) in {cwd}. "
            "Refusing to build a wheel without web/frontend_dist/index.html."
        )
    except OSError as exc:
        _fail(
            f"Could not run {' '.join(command)}: {exc}. "
            "Install Node.js and npm, then retry the package build. "
            "Refusing to build a wheel without web/frontend_dist/index.html."
        )


def _fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise FrontendBuildError(message)


if __name__ == "__main__":
    try:
        built = build_frontend()
    except FrontendBuildError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Frontend bundle written to {built}")
