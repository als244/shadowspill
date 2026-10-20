"""Dense Qwen 3.5: a repeating hybrid of Gated DeltaNet and gated GQA, each
block followed by a dense SwiGLU MLP; no routed experts."""

from .config import Config

__all__ = ["Config"]
