"""
ModelForge Notebook CLI — public API.

Provides an interactive wizard-style interface that calls the ModelForge
Python APIs directly, without going through the REST/web layer.

Install the required extras with:
    pip install modelforge-finetuning[cli]

Usage from a notebook cell:
    from ModelForge.notebook_cli import run_cli
    run_cli()

Usage from a terminal:
    modelforge-nb
"""
from .wizard import ModelForgeWizard, main as run_cli

__all__ = ["ModelForgeWizard", "run_cli"]
