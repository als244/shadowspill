"""Deterministic, exact-scale model cases shared by eager and planned workers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast

import torch
import torch.nn as nn

from workloads.precision import TrainingDtypes
from workloads.providers import MODEL_DTYPES, ModelImplementation

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


class SupportsLoss(Protocol):
    """The objective a workload exposes, whatever arguments it takes."""

    def loss(self, *values: Any, **options: Any) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class NumericalCase:
    family: str
    model_implementation: ModelImplementation
    model: nn.Module
    microbatches: list[list[Any]]
    dtypes: TrainingDtypes = field(default_factory=TrainingDtypes)

    def objective(self, model: nn.Module, *values: Any) -> torch.Tensor:
        tokens, targets, sequence_lengths = values
        workload = cast(SupportsLoss, model)
        if self.family == "olmoe":
            return workload.loss(
                tokens,
                targets,
                seq_lens=sequence_lengths,
                aux_coef=0.01,
            )
        return workload.loss(tokens, targets, seq_lens=sequence_lengths)

    @contextmanager
    def implementations(self, *, deterministic: bool = False) -> Iterator[None]:
        """Request reproducible kernels without overriding provider selection.

        mlops selects supported implementations from the inputs and device.
        Deterministic accumulation is needed for numerical comparisons; it
        remains optional for performance measurements.
        """
        if self.model_implementation == "mlops":
            from mlops.dispatch import deterministic_kernels

            with deterministic_kernels(deterministic):
                yield
        else:
            yield

    @property
    def optimizer(self) -> Any:
        return self.dtypes.optimizer()


def build_case(
    family: str,
    *,
    model_implementation: ModelImplementation = "pytorch",
    seed: int = 20_260_811,
    model_config: Mapping[str, Any] | None = None,
    data_geometry: Sequence[Mapping[str, Any]] | None = None,
    model_dtype: str | None = None,
    master_dtype: str | None = None,
    grad_dtype: str | None = None,
    opt_state_dtype: str | None = None,
) -> NumericalCase:
    """Build model and CPU examples in one reproducible RNG order."""

    if model_implementation not in {"pytorch", "mlops"}:
        raise ValueError(f"unknown model implementation {model_implementation!r}")
    if model_dtype is not None and model_dtype not in MODEL_DTYPES:
        raise ValueError(f"model_dtype must be one of {MODEL_DTYPES}")
    dtypes = TrainingDtypes(
        model_dtype or "bfloat16",
        master_dtype or "none",
        grad_dtype or "parameter",
        opt_state_dtype or "bfloat16",
    )
    torch.manual_seed(seed)
    from importlib import import_module

    from workloads.pytorch import Llama3Config, OLMoEConfig, Qwen35Config

    library = import_module("workloads." + model_implementation)
    presets = {"llama3": Llama3Config, "qwen35": Qwen35Config, "olmoe": OLMoEConfig}
    classes = {"llama3": "Llama3", "qwen35": "Qwen35", "olmoe": "OLMoE"}
    if family not in presets:
        raise ValueError(f"unknown numerical family {family!r}")
    config = replace(presets[family].numerical(), max_seq_len=192)
    model_type = getattr(library, classes[family])
    if model_config:
        try:
            config = replace(config, **dict(model_config))
        except TypeError as exc:
            raise ValueError(f"invalid {family} model_config: {exc}") from exc
    # The branch above pairs each config with the class that reads it.
    model: nn.Module = model_type(cast(Any, config)).to(
        getattr(torch, model_dtype or "bfloat16")
    )
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
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
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
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
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
    return NumericalCase(family, model_implementation, model, microbatches, dtypes)


__all__ = ["ModelImplementation", "NumericalCase", "build_case"]
