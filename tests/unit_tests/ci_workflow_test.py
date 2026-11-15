import re
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
BUILD_SCRIPT = ROOT / "scripts" / "build_distribution.sh"
SMOKE_SCRIPT = ROOT / "scripts" / "distribution_smoke.py"
PINNED_ACTION = re.compile(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}\Z")


def _load_smoke():
    import importlib.util

    spec = importlib.util.spec_from_file_location("distribution_smoke", SMOKE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ci_workflow_contract():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert re.search(r"(?m)^on:\n  pull_request:\n  push:\n    branches:\n      - main\n", text)
    uses = re.findall(r"uses:\s+(\S+)", text)
    assert uses, "workflow must pin third-party actions"
    assert all(PINNED_ACTION.fullmatch(use) for use in uses), uses
    assert not re.search(r"uses:\s+\S+@v\d", text)

    assert "python-version-file: .python-version" in text
    assert "node-version-file: .node-version" in text
    assert "cache-dependency-glob: uv.lock" in text
    assert "cache-dependency-path: web/frontend/package-lock.json" in text
    assert "uv sync --frozen --extra dev" in text
    assert "uv run pytest" in text
    assert "npm ci" in text
    assert "npm run build" in text
    assert "bash scripts/build_distribution.sh" in text
    assert "scripts/distribution_smoke.py" in text
    assert "--extra gpu" not in text
    assert 'CUDA_VISIBLE_DEVICES: ""' in text
    assert 'HF_HUB_OFFLINE: "1"' in text
    for job in ("python:", "frontend:", "distribution:"):
        assert job in text


def test_release_script_is_the_supported_build_path():
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "npm ci --prefix web/frontend" in text
    assert "npm run build --prefix web/frontend" in text
    assert "uv build" in text
    assert "distribution_smoke.py --archives dist" in text
    assert "set -euo pipefail" in text


def test_incomplete_archives_fail(tmp_path):
    smoke = _load_smoke()
    wheel = tmp_path / "modelforge_finetuning-4.0.0-py3-none-any.whl"
    sdist = tmp_path / "modelforge_finetuning-4.0.0.tar.gz"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("web/app.py", "print('no ui')\n")
    with tarfile.open(sdist, "w:gz"):
        pass

    with pytest.raises(SystemExit, match="incomplete distribution"):
        smoke.assert_distribution_archives(tmp_path)


def test_complete_archives_pass(tmp_path):
    smoke = _load_smoke()
    wheel = tmp_path / "modelforge_finetuning-4.0.0-py3-none-any.whl"
    sdist = tmp_path / "modelforge_finetuning-4.0.0.tar.gz"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("web/frontend_dist/index.html", "<title>ModelForge</title>")
        archive.writestr("web/frontend_dist/assets/index-abc.js", "console.log(1)")
    with tarfile.open(sdist, "w:gz") as archive:
        index = tmp_path / "index.html"
        script = tmp_path / "index-abc.js"
        index.write_text("<title>ModelForge</title>", encoding="utf-8")
        script.write_text("console.log(1)", encoding="utf-8")
        archive.add(index, arcname="modelforge_finetuning-4.0.0/web/frontend_dist/index.html")
        archive.add(script, arcname="modelforge_finetuning-4.0.0/web/frontend_dist/assets/index-abc.js")

    smoke.assert_distribution_archives(tmp_path)


def test_smoke_finds_console_scripts_beside_the_interpreter(tmp_path, monkeypatch):
    smoke = _load_smoke()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    executable = bindir / "python"
    executable.write_text("", encoding="utf-8")
    script = bindir / "modelforge"
    script.write_text("", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setattr(smoke.sys, "executable", str(executable))
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: None)

    assert smoke._console_script("modelforge") == str(script)
    with pytest.raises(SystemExit, match="modelforge-nb"):
        smoke._console_script("modelforge-nb")


def test_frontend_bundle_requires_referenced_assets(tmp_path):
    smoke = _load_smoke()
    (tmp_path / "index.html").write_text(
        '<title>ModelForge</title><script src="/assets/app.js"></script>',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="missing file"):
        smoke.assert_frontend_bundle(tmp_path)

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log(1)", encoding="utf-8")
    smoke.assert_frontend_bundle(tmp_path)
