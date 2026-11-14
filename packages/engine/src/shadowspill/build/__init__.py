"""Build phase entry points. Internal compiler passes remain implementation details."""

from .forward import build_forward
from .step import build_step
from .training import build_forward_backward

__all__ = ["build_forward", "build_forward_backward", "build_step"]
