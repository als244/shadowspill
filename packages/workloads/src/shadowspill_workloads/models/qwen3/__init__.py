"""Qwen 3: GQA attention with per-head query and key RMSNorm and a dense
SwiGLU MLP in every decoder block, with RMSNorm and RoPE; the attention
width is its own, and the smaller sizes tie the embedding and the head."""

from .config import Config

__all__ = ["Config"]
