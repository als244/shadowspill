"""Preserve task-local Triton compiler choices across disposable caches.

External operator selection and dispatch belong to provider-owned preparation.
"""

from ._scope import capture_choices, use_choices

__all__ = ["capture_choices", "use_choices"]
