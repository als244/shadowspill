"""Preserve task-local provider choices across disposable compiler caches.

Existing Triton record keys and the Program's historical ``triton_choices``
binding remain compatible. Other backends use namespaced, versioned records.
"""

from ._scope import capture_choices, use_choices

__all__ = ["capture_choices", "use_choices"]
