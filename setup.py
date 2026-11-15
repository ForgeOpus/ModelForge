"""Build the locked frontend bundle before setuptools collects package data."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

try:
    from setuptools.command.editable_wheel import editable_wheel as _editable_wheel
except ImportError:  # pragma: no cover - setuptools>=64 provides this command
    _editable_wheel = None

ROOT = Path(__file__).resolve().parent
_FRONTEND_BUILT = False


def _load_builder():
    path = ROOT / "scripts" / "build_frontend.py"
    if not path.is_file():
        raise RuntimeError(
            f"Missing {path}. Source distributions must include scripts/build_frontend.py "
            "so the wheel build can run the locked frontend build."
        )
    spec = importlib.util.spec_from_file_location("modelforge_frontend_build", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frontend build script at {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_frontend() -> Path:
    global _FRONTEND_BUILT
    module = _load_builder()
    if _FRONTEND_BUILT and (ROOT / "web" / "frontend_dist" / "index.html").is_file():
        return ROOT / "web" / "frontend_dist"
    dist = module.build_frontend(ROOT)
    _FRONTEND_BUILT = True
    return dist


def _frontend_files() -> list[str]:
    dist = ROOT / "web" / "frontend_dist"
    return [
        path.relative_to(ROOT / "web").as_posix()
        for path in dist.rglob("*")
        if path.is_file()
    ]


def _include_frontend(distribution) -> None:
    files = _frontend_files()
    if not files:
        raise RuntimeError(
            "Frontend build produced no files under web/frontend_dist. "
            "Refusing to build a wheel without the browser UI."
        )
    package_data = distribution.package_data or {}
    package_data["web"] = files
    distribution.package_data = package_data


class build_py(_build_py):
    def finalize_options(self):
        _build_frontend()
        _include_frontend(self.distribution)
        super().finalize_options()

    def run(self):
        super().run()
        if self.editable_mode or not self.build_lib:
            return
        package_dir = Path(self.build_lib) / "web"
        package_dir.mkdir(parents=True, exist_ok=True)
        # MANIFEST.in includes the frontend source so the sdist can run npm.
        # The wheel should contain only the generated bundle.
        bundled_source = package_dir / "frontend"
        if bundled_source.exists():
            shutil.rmtree(bundled_source)
        source = ROOT / "web" / "frontend_dist"
        destination = package_dir / "frontend_dist"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


cmdclass = {"build_py": build_py}

if _editable_wheel is not None:

    class editable_wheel(_editable_wheel):
        def run(self):
            _build_frontend()
            super().run()

    cmdclass["editable_wheel"] = editable_wheel

setup(cmdclass=cmdclass)
