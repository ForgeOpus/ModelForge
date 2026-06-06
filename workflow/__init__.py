"""Shared workflow layer for ModelForge UI surfaces.

Imports are intentionally light so recommendation/config utilities do not load
trainer dependencies until a job actually starts.
"""

from workflow.service import WorkflowService

__all__ = ["WorkflowService"]
