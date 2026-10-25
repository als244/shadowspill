"""Preserve task-local Triton choices across disposable compiler caches."""

from .triton import capture_choices, use_choices

__all__ = ["capture_choices", "use_choices"]
