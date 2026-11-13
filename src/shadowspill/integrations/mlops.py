"""Turn MLOps operator metadata into ShadowSpill build options.

This module runs during build configuration, never inside a training task.
It does four small translations:

* Output slicing and sum reductions -> OperatorRule for chunked execution.
* Supported gradient writes -> GradientDestinationRule for fused accumulation.
* Read-only inputs and layout support -> ExecutionLifetimes for safe movement/copies.
* Kernel preparation factories -> OperatorPreparation for saving/loading kernels.

MLOps supplies the operator facts. The caller still chooses checkpoint boundaries,
chunk sizes, recomputation and memory placement. No kernels or tensor computation
are implemented here, and MLOps is imported only when a helper is called.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from mlops.operator_metadata import (
        AxisFromInput,
        ChunkLocalOutput,
        FactoredAxis,
        GradientAccumulation,
        OperatorChunking,
    )


# 1. Chunked execution: how to slice outputs and combine partial results.


def operator_chunk_rules(batch_size: int = 1) -> tuple[OperatorRule, ...]:
    """Describe common MLOps outputs, including flattened batch/sequence layouts.

    This tells the compiler how an output relates to its sliced inputs. It does
    not choose which input axis to split or how large a chunk should be.
    """
    from mlops.operator_metadata import operator_chunking

    descriptions = operator_chunking(batch_size=batch_size)
    return _make_chunk_rules(descriptions)


def sonicmoe_chunk_rules(
    *, compute_precision: str = "bf16"
) -> tuple[OperatorRule, ...]:
    """Describe Sonic outputs and summed gradients for BF16 or frozen block-FP8."""
    from mlops.operator_metadata import sonicmoe_chunking

    descriptions = sonicmoe_chunking(compute_precision=compute_precision)
    return _make_chunk_rules(descriptions)


def mhc_chunk_rules(*, batch_size: int = 1) -> tuple[OperatorRule, ...]:
    """Describe fused mHC outputs; the provider checks each chunk's alignment."""
    from mlops.operator_metadata import mhc_chunking

    descriptions = mhc_chunking(batch_size=batch_size)
    return _make_chunk_rules(descriptions)


def _make_chunk_rules(
    descriptions: Sequence[OperatorChunking],
) -> tuple[OperatorRule, ...]:
    rules = []
    for description in descriptions:
        output_axes = tuple(
            _make_output_axis(output_axis) for output_axis in description.output_axes
        )
        rules.append(
            OperatorRule(
                target=description.target,
                version=description.version,
                output_axes=output_axes,
                reductions=description.reductions,
            )
        )
    return tuple(rules)


def _make_output_axis(
    description: AxisFromInput | FactoredAxis | ChunkLocalOutput | None,
) -> AxisFrom | ChunkAxis | ChunkLocal | None:
    from mlops.operator_metadata import AxisFromInput, ChunkLocalOutput, FactoredAxis

    if description is None:
        # Shared output, or an additive result listed in the rule's reductions.
        return None
    if isinstance(description, AxisFromInput):
        # Example: normalized activations preserve their input's sequence axis.
        return AxisFrom(
            input_index=description.input_index,
            trailing_dims=description.trailing_dims,
        )
    if isinstance(description, FactoredAxis):
        # Example: a physical [batch * sequence, hidden] output has before=batch.
        return ChunkAxis(
            dim=description.dim,
            before=description.before,
            after=description.after,
        )
    if isinstance(description, ChunkLocalOutput):
        # Sorted expert rows/maps belong to this chunk and cannot escape it.
        return ChunkLocal(dim=description.dim)
    raise TypeError(f"unsupported MLOps output-axis description: {description!r}")


# 2. Gradient accumulation: replace eligible products with destination writes.


def sonicmoe_gradient_destination_rules() -> tuple[GradientDestinationRule, ...]:
    """Let eligible Sonic backwards write directly into an existing gradient.

    MLOps decides which dtype, layout and scaling paths its kernels support.
    ShadowSpill proves destination ownership and performs the graph replacement.
    """
    from mlops.operator_metadata import sonicmoe_gradient_accumulation

    rules = []
    for description in sonicmoe_gradient_accumulation():
        rules.append(
            GradientDestinationRule(
                target=description.target,
                replacement=description.replacement,
                output_indices=description.output_indices,
                version=description.version,
                mapper=_make_gradient_mapper(description),
            )
        )
    return tuple(rules)


def _make_gradient_mapper(
    description: GradientAccumulation,
) -> Callable[[GradientDestinationContext], GradientDestinationMapping | None]:
    def select_destinations(
        context: GradientDestinationContext,
    ) -> GradientDestinationMapping | None:
        selection = description.resolve(context)
        if selection is None:
            # The original backward and ordinary gradient addition remain.
            return None

        # These kernels accept the existing destination tensors directly.
        # No reshape/view needs to be applied to a destination before the call.
        destination_views = tuple(() for _ in selection.output_indices)
        return GradientDestinationMapping(
            output_indices=selection.output_indices,
            views=destination_views,
            keywords=selection.keywords,
        )

    return select_destinations


# 3. Value lifetimes: attach operator facts to the caller's scheduling choices.


def fla_execution_lifetimes(
    *,
    rules: tuple[LifetimeRule, ...] = (),
    select: Callable[[LifetimeOperation], LifetimeSelection | None] | None = None,
) -> ExecutionLifetimes:
    """Supply FLA input-layout facts for caller-selected movement/compaction.

    The caller's rules/select decide what to move or copy. MLOps only identifies
    read-only operators and input ports that accept the same values in a new
    layout. ShadowSpill must still prove aliases, effects and storage safety.
    """
    from mlops.operator_metadata import fla_input_layouts

    readonly_operators = []
    supported_layouts = []
    for description in fla_input_layouts():
        readonly_operators.append(
            ReadonlyOperator(target=description.target, version=description.version)
        )
        supported_layouts.append(
            LayoutIndependentOperator(
                target=description.target,
                version=description.version,
                inputs=description.inputs,
            )
        )
    return ExecutionLifetimes(
        rules=rules,
        select=select,
        readonly=tuple(readonly_operators),
        layout=tuple(supported_layouts),
    )


# 4. Prepared kernels: forward the provider's factory and options unchanged.


def sonicmoe_preparation(*, cache_root: str | None = None) -> OperatorPreparation:
    """Request saved BF16 Sonic kernels; construction does not access the GPU."""
    from mlops.operator_metadata import sonicmoe_preparation as describe_preparation

    description = describe_preparation(cache_root=cache_root)
    return OperatorPreparation(factory=description.factory, options=description.options)


def operator_preparations(
    *, dtype: torch.dtype = torch.bfloat16, cache_root: str | None = None
) -> tuple[OperatorPreparation, ...]:
    """Request available BF16 provider preparations; build checks device support."""
    from mlops.operator_metadata import operator_preparations as describe_preparations

    preparations = []
    for description in describe_preparations(dtype=dtype, cache_root=cache_root):
        preparations.append(
            OperatorPreparation(
                factory=description.factory, options=description.options
            )
        )
    return tuple(preparations)
