"""OLMoE: GQA attention followed by a routed SwiGLU expert block, with a
softmax router that selects top-k without renormalizing."""

from .config import Config

__all__ = ["Config"]
