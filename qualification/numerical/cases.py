"""Deterministic, exact-scale model cases shared by eager and planned workers."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import Any, Literal

import mlops
import torch
import torch.nn as nn
from mlops.dispatch import use_implementations

from models.mlops import (
    Llama3 as MlopsLlama3,
)
from models.mlops import (
    OLMoE as MlopsOLMoE,
)
from models.mlops import (
    Qwen35 as MlopsQwen35,
)
from models.pytorch import (
    Llama3 as PyTorchLlama3,
)
from models.pytorch import (
    Llama3Config,
    OLMoEConfig,
    Qwen35Config,
)
from models.pytorch import (
    OLMoE as PyTorchOLMoE,
)
from models.pytorch import (
    Qwen35 as PyTorchQwen35,
)

ModelImplementation = Literal["pytorch", "mlops"]

_IMPLEMENTATIONS = {
    "llama3": {
        "embedding": "builtin.embedding.deterministic",
        "rms_norm": "builtin.rms_norm.triton",
        "rope": "builtin.rope.triton_table",
        "flash_attention": "builtin.flash_attention.aten",
        "swiglu": "builtin.swiglu.triton",
        "head_loss": "builtin.head_loss.chunked",
    },
    "qwen35": {
        "embedding": "builtin.embedding.deterministic",
        "rms_norm": "builtin.rms_norm.triton",
        "gated_rms_norm": "fla.gated_rms_norm",
        "partial_rope": "builtin.partial_rope.triton",
        "flash_attention": "builtin.flash_attention.aten",
        "causal_conv_silu": "fla.causal_conv_silu",
        "l2_norm": "fla.l2_norm",
        "linear_attention": "fla.linear_attention.gated_delta_rule",
        "swiglu": "builtin.swiglu.triton",
        "head_loss": "builtin.head_loss.chunked",
    },
    "olmoe": {
        "embedding": "builtin.embedding.deterministic",
        "rms_norm": "builtin.rms_norm.triton",
        "rope": "builtin.rope.triton_table",
        "flash_attention": "builtin.flash_attention.aten",
        "moe": "builtin.moe.grouped_gemm_composed",
        "head_loss": "builtin.head_loss.chunked",
    },
}


@dataclass(frozen=True, slots=True)
class NumericalCase:
    family: str
    model_implementation: ModelImplementation
    model: nn.Module
    microbatches: list[list[Any]]

    def objective(self, model: nn.Module, *values: Any) -> torch.Tensor:
        tokens, targets, sequence_lengths = values
        if self.family == "olmoe":
            return model.loss(
                tokens,
                targets,
                seq_lens=sequence_lengths,
                aux_coef=0.01,
            )
        return model.loss(tokens, targets, seq_lens=sequence_lengths)

    def implementations(self) -> AbstractContextManager[Any]:
        if self.model_implementation == "pytorch":
            return nullcontext()
        return use_implementations(_IMPLEMENTATIONS[self.family])

    @staticmethod
    def optimizer(parameters: Any) -> torch.optim.Optimizer:
        values = list(parameters)
        groups = (
            {
                "params": [item for item in values if item.ndim >= 2],
                "weight_decay": 0.1,
            },
            {"params": [item for item in values if item.ndim < 2], "weight_decay": 0.0},
        )
        return mlops.optim.AdamW(groups, lr=3e-4)


def build_case(
    family: str,
    *,
    model_implementation: ModelImplementation = "pytorch",
    seed: int = 20_260_811,
) -> NumericalCase:
    """Build model and CPU examples in one reproducible RNG order."""

    if model_implementation not in {"pytorch", "mlops"}:
        raise ValueError(f"unknown model implementation {model_implementation!r}")
    torch.manual_seed(seed)
    if family == "llama3":
        config = replace(Llama3Config.numerical(), max_seq_len=192)
        model_type = PyTorchLlama3 if model_implementation == "pytorch" else MlopsLlama3
        model: nn.Module = model_type(config).to(torch.bfloat16)
    elif family == "qwen35":
        config = replace(Qwen35Config.numerical(), max_seq_len=192)
        model_type = (
            PyTorchQwen35 if model_implementation == "pytorch" else MlopsQwen35
        )
        model = model_type(config).to(torch.bfloat16)
    elif family == "olmoe":
        config = replace(OLMoEConfig.numerical(), max_seq_len=192)
        model_type = PyTorchOLMoE if model_implementation == "pytorch" else MlopsOLMoE
        model = model_type(config).to(torch.bfloat16)
    else:
        raise ValueError(f"unknown numerical family {family!r}")
    microbatches = [
        [
            torch.randint(config.vocab_size, (1, 64)),
            torch.randint(config.vocab_size, (1, 64)),
            (13, 19, 32),
        ],
        [
            torch.randint(config.vocab_size, (1, 96)),
            torch.randint(config.vocab_size, (1, 96)),
            (17, 31, 48),
        ],
    ]
    return NumericalCase(family, model_implementation, model, microbatches)


DEFAULT_DEVICE_BUDGETS = {
    "llama3": 10 << 30,
    "qwen35": 10 << 30,
    "olmoe": 10 << 30,
}

__all__ = [
    "DEFAULT_DEVICE_BUDGETS",
    "ModelImplementation",
    "NumericalCase",
    "build_case",
]
