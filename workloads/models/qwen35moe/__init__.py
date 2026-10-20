"""Qwen 3.5 MoE: three Gated DeltaNet blocks then gated GQA, repeating,
with routed experts and a gated shared expert after every mixer."""

from .config import Config

__all__ = ["Config"]
