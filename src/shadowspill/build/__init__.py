"""Build phase entry points. Internal compiler passes remain implementation details."""

from .forward import build_forward

__all__ = ["build_forward"]
