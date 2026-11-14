"""Llama 3: GQA attention and a dense SwiGLU MLP in every decoder block,
with RMSNorm and RoPE; separate embedding and output-head weights."""

from .config import Config

__all__ = ["Config"]
