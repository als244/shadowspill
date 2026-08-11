"""Deterministic, exact-scale model cases shared by eager and planned workers."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

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

_DEFAULT_DATA_GEOMETRY: tuple[dict[str, Any], ...] = (
    {
        "token_shape": [1, 64],
        "sequence_lengths": [13, 19, 32],
    },
    {
        "token_shape": [1, 96],
        "sequence_lengths": [17, 31, 48],
    },
)

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
    model_config: Mapping[str, Any] | None = None,
    data_geometry: Sequence[Mapping[str, Any]] | None = None,
    case_factory: str | None = None,
    case_options: Mapping[str, Any] | None = None,
) -> NumericalCase:
    """Build model and CPU examples in one reproducible RNG order."""

    if model_implementation not in {"pytorch", "mlops"}:
        raise ValueError(f"unknown model implementation {model_implementation!r}")
    if case_factory is not None:
        module_name, separator, attribute = case_factory.partition(":")
        if separator == "" or not module_name or not attribute:
            raise ValueError("case_factory must use the form 'module:function'")
        factory = getattr(importlib.import_module(module_name), attribute)
        if not callable(factory):
            raise TypeError(
                f"verification case factory is not callable: {case_factory}"
            )
        case = factory(
            model_name=family,
            model_implementation=model_implementation,
            seed=seed,
            model_config=dict(model_config or {}),
            data_geometry=tuple(dict(item) for item in (data_geometry or ())),
            case_options=dict(case_options or {}),
        )
        required = (
            "family",
            "model_implementation",
            "model",
            "microbatches",
            "objective",
            "optimizer",
            "implementations",
        )
        missing = [name for name in required if not hasattr(case, name)]
        if missing:
            raise TypeError(
                f"verification case factory {case_factory} omitted: "
                + ", ".join(missing)
            )
        return cast(NumericalCase, case)
    if case_options:
        raise ValueError("case_options require a custom case_factory")
    torch.manual_seed(seed)
    if family == "llama3":
        config = replace(Llama3Config.numerical(), max_seq_len=192)
        model_type = PyTorchLlama3 if model_implementation == "pytorch" else MlopsLlama3
    elif family == "qwen35":
        config = replace(Qwen35Config.numerical(), max_seq_len=192)
        model_type = (
            PyTorchQwen35 if model_implementation == "pytorch" else MlopsQwen35
        )
    elif family == "olmoe":
        config = replace(OLMoEConfig.numerical(), max_seq_len=192)
        model_type = PyTorchOLMoE if model_implementation == "pytorch" else MlopsOLMoE
    else:
        raise ValueError(f"unknown numerical family {family!r}")
    if model_config:
        try:
            config = replace(config, **dict(model_config))
        except TypeError as exc:
            raise ValueError(f"invalid {family} model_config: {exc}") from exc
    model: nn.Module = model_type(config).to(torch.bfloat16)
    selected_geometry = data_geometry or _DEFAULT_DATA_GEOMETRY
    microbatches: list[list[Any]] = []
    for index, item in enumerate(selected_geometry):
        geometry = dict(item)
        unknown = set(geometry) - {"token_shape", "sequence_lengths"}
        if unknown:
            raise ValueError(
                f"data_geometry[{index}] has unknown fields: {sorted(unknown)}"
            )
        shape_value = geometry.get("token_shape")
        sequence_value = geometry.get("sequence_lengths")
        if (
            not isinstance(shape_value, Sequence)
            or isinstance(shape_value, (str, bytes))
            or not shape_value
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in shape_value
            )
        ):
            raise ValueError(
                f"data_geometry[{index}].token_shape must contain positive integers"
            )
        shape = tuple(int(value) for value in shape_value)
        if (
            not isinstance(sequence_value, Sequence)
            or isinstance(sequence_value, (str, bytes))
            or not sequence_value
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in sequence_value
            )
        ):
            raise ValueError(
                f"data_geometry[{index}].sequence_lengths must contain "
                "positive integers"
            )
        sequence_lengths = tuple(int(value) for value in sequence_value)
        elements = 1
        for extent in shape:
            elements *= extent
        if sum(sequence_lengths) != elements:
            raise ValueError(
                f"data_geometry[{index}] sequence lengths sum to "
                f"{sum(sequence_lengths)}, expected {elements} from token_shape"
            )
        if max(sequence_lengths) > config.max_seq_len:
            raise ValueError(
                f"data_geometry[{index}] exceeds max_seq_len={config.max_seq_len}"
            )
        microbatches.append(
            [
                torch.randint(config.vocab_size, shape),
                torch.randint(config.vocab_size, shape),
                sequence_lengths,
            ]
        )
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
