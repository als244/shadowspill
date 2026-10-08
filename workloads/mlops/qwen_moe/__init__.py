"""Qwen3-30B-A3B and Qwen3.5-35B-A3B language-model workloads."""

from .config import Qwen30BConfig, Qwen35BConfig
from .model import Qwen30B, Qwen35B

__all__ = ["Qwen30B", "Qwen30BConfig", "Qwen35B", "Qwen35BConfig"]
