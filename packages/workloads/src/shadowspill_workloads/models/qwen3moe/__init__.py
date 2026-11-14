"""Qwen 3 MoE: GQA attention followed by routed experts in every block,
with normalized top-k routing and optional expert parallelism."""

from .config import Config

__all__ = ["Config"]
