"""Translate standalone MLOps declarations into ShadowSpill build controls.

Operator names, geometry, precision eligibility and kernel preparation belong
to MLOps. This module only adapts its records; it selects no workload policy and
does not import MLOps until a factory is called.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import torch

from shadowspill import OperatorPreparation
from shadowspill.pytorch import (
    AxisFrom,
    ChunkAxis,
    ChunkLocal,
    ExecutionLifetimes,
    GradientDestinationContext,
    GradientDestinationMapping,
    GradientDestinationRule,
    LayoutIndependentOperator,
    LifetimeOperation,
    LifetimeRule,
    LifetimeSelection,
    OperatorRule,
    ReadonlyOperator,
)


def _axis(value: Any) -> AxisFrom | ChunkAxis | ChunkLocal | None:
    from mlops.contracts import types

    if value is None:
        return None
    if isinstance(value, types.AxisFrom):
        return AxisFrom(value.input_index, value.trailing_dims)
    if isinstance(value, types.ChunkAxis):
        return ChunkAxis(value.dim, value.before, value.after)
    if isinstance(value, types.ChunkLocal):
        return ChunkLocal(value.dim)
    raise TypeError(f"unsupported MLOps axis contract: {type(value).__name__}")


def _chunk_rules(contracts: Iterable[Any]) -> tuple[OperatorRule, ...]:
    return tuple(
        OperatorRule(
            item.target,
            item.version,
            tuple(_axis(value) for value in item.output_axes),
            item.reductions,
        )
        for item in contracts
    )


def operator_chunk_rules(batch_size: int = 1) -> tuple[OperatorRule, ...]:
    """Common MLOps axis/reduction facts; callers choose regions and chunk sizes."""
    from mlops.contracts import operator_chunk_contracts

    return _chunk_rules(operator_chunk_contracts(batch_size))


def sonicmoe_chunk_rules(
    *, compute_precision: str = "bf16"
) -> tuple[OperatorRule, ...]:
    """Sonic BF16 or frozen block-FP8 contracts, including local routing state."""
    from mlops.contracts import sonicmoe_chunk_contracts

    return _chunk_rules(sonicmoe_chunk_contracts(compute_precision=compute_precision))


def mhc_chunk_rules(*, batch_size: int = 1) -> tuple[OperatorRule, ...]:
    """Fused mHC contracts, including the provider's invocation alignment."""
    from mlops.contracts import mhc_chunk_contracts

    return _chunk_rules(mhc_chunk_contracts(batch_size=batch_size))


def _destination_mapping(
    context: GradientDestinationContext, *, resolve: Callable[[Any], Any]
) -> GradientDestinationMapping | None:
    result = resolve(context)
    if result is None:
        return None
    return GradientDestinationMapping(
        result.output_indices, ((),) * len(result.output_indices), result.keywords
    )


def sonicmoe_gradient_destination_rules() -> tuple[GradientDestinationRule, ...]:
    """Adapt provider-certified replacements; unsupported outputs keep ordinary adds."""
    from mlops.contracts import sonicmoe_gradient_contracts

    return tuple(
        GradientDestinationRule(
            item.target,
            item.replacement,
            item.output_indices,
            item.version,
            partial(_destination_mapping, resolve=item.resolve),
        )
        for item in sonicmoe_gradient_contracts()
    )


def fla_execution_lifetimes(
    *,
    rules: tuple[LifetimeRule, ...] = (),
    select: Callable[[LifetimeOperation], LifetimeSelection | None] | None = None,
) -> ExecutionLifetimes:
    """Attach provider facts to caller-selected movement and compaction policies."""
    from mlops.contracts import fla_value_contracts

    contracts = fla_value_contracts()
    return ExecutionLifetimes(
        rules=rules,
        select=select,
        readonly=tuple(ReadonlyOperator(c.target, c.version) for c in contracts),
        layout=tuple(
            LayoutIndependentOperator(c.target, c.version, c.inputs) for c in contracts
        ),
    )


def sonicmoe_preparation(*, cache_root: str | None = None) -> OperatorPreparation:
    """Explicit supported Sonic binary preparation; construction is CPU-only."""
    from mlops.contracts import sonicmoe_preparation as description

    item = description(cache_root=cache_root)
    return OperatorPreparation(item.factory, item.options)


def operator_preparations(
    *, dtype: torch.dtype = torch.bfloat16, cache_root: str | None = None
) -> tuple[OperatorPreparation, ...]:
    """Supported provider preparations, with device selection deferred to build."""
    from mlops.contracts import operator_preparations as descriptions

    return tuple(
        OperatorPreparation(item.factory, item.options)
        for item in descriptions(dtype=dtype, cache_root=cache_root)
    )
