"""Reference models using the separately installed :mod:`mlops` package."""

from .llama3 import Llama3, Llama3Config
from .olmoe import OLMoE, OLMoEConfig
from .qwen3_moe import Qwen3MoE, Qwen3MoEConfig
from .qwen35 import Qwen35, Qwen35Config
from .qwen35_moe import Qwen35MoE, Qwen35MoEConfig

__all__ = [
    "Llama3",
    "Llama3Config",
    "OLMoE",
    "OLMoEConfig",
    "Qwen3MoE",
    "Qwen3MoEConfig",
    "Qwen35",
    "Qwen35Config",
    "Qwen35MoE",
    "Qwen35MoEConfig",
]
