"""Bind one structural graph-pair graph_pairs to a stage occurrence."""

from __future__ import annotations

import torch

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import rebind_backward_input_provenance
from shadowspill.pytorch.capture.artifacts import AotGraphPair

from ..partition.artifacts import StageExample
from .artifacts import GraphPairVariant, TaskGraphPairs


def rebind_task_graph_pairs(
    graph_pairs: TaskGraphPairs,
    example: StageExample,
) -> TaskGraphPairs:
    """Replace occurrence-local values while preserving structural graph code."""

    return TaskGraphPairs(
        structural_contract=graph_pairs.structural_contract,
        root_output_indices=graph_pairs.root_output_indices,
        variants=tuple(
            GraphPairVariant(
                item.option_id,
                item.memory_budget,
                _rebind_graph_pair(
                    item.pair,
                    example,
                    graph_pairs.root_output_indices,
                ),
                item.accumulates,
            )
            for item in graph_pairs.variants
        ),
        reference_option_id=graph_pairs.reference_option_id,
    )


def _rebind_graph_pair(
    pair: AotGraphPair,
    example: StageExample,
    roots: tuple[int, ...],
) -> AotGraphPair:
    forward_arguments: list[torch.Tensor] = []
    for position in pair.forward.tensor_argument_positions:
        try:
            value = example.inputs[position]
        except IndexError as exc:
            raise CaptureError(
                "reused stage forward argument positions changed"
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise CaptureError("reused stage tensor argument became static")
        forward_arguments.append(value.detach())
    if len(forward_arguments) != pair.forward.argument_count:
        raise CaptureError("reused stage forward tensor argument count changed")
    forward_provenance = tuple(
        example.stage.input_provenance[position]
        for position in pair.forward.tensor_argument_positions
    )
    forward = pair.forward.rebind_examples(
        tuple(forward_arguments),
        input_provenance=forward_provenance,
    )
    backward = pair.backward.rebind_examples(
        pair.backward.example_arguments,
        input_provenance=rebind_backward_input_provenance(pair, forward),
    )
    if len(roots) < pair.specialized_unit_tangent_count:
        raise CaptureError("specialized tangent count exceeds stage roots")
    return AotGraphPair(
        forward=forward,
        backward=backward,
        recomputation=pair.recomputation,
        saved_value_count=pair.saved_value_count,
        specialized_unit_tangent_count=pair.specialized_unit_tangent_count,
    )


__all__ = ["rebind_task_graph_pairs"]
