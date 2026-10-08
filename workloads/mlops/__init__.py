"""Reference models using the separately installed :mod:`mlops` package."""

from .llama3 import Llama3, Llama3Config
from .olmoe import OLMoE, OLMoEConfig
from .qwen35 import Qwen35, Qwen35Config
from .qwen_moe import Qwen30B, Qwen30BConfig, Qwen35B, Qwen35BConfig

__all__ = [
    "Llama3",
    "Llama3Config",
    "OLMoE",
    "OLMoEConfig",
    "Qwen30B",
    "Qwen30BConfig",
    "Qwen35",
    "Qwen35B",
    "Qwen35BConfig",
    "Qwen35Config",
]
