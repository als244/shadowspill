"""Bind one structural `TaskGraphPairs` to a stage occurrence."""

from __future__ import annotations

from dataclasses import replace

import torch

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import (
    physical_input_provenance,
    rebind_backward_input_provenance,
)
from shadowspill.pytorch.capture.artifacts import AotGraphPair
from shadowspill.pytorch.representations import component_at

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
    components = pair.forward.input_components or tuple(
        (position, ()) for position in pair.forward.tensor_argument_positions
    )
    for position, path in components:
        try:
            value = example.inputs[position]
        except IndexError as exc:
            raise CaptureError(
                "reused stage forward argument positions changed"
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise CaptureError("reused stage tensor argument became static")
        forward_arguments.append(component_at(value, path).detach())
    if len(forward_arguments) != pair.forward.argument_count:
        raise CaptureError("reused stage forward tensor argument count changed")
    forward_provenance = (
        physical_input_provenance(example.inputs, example.stage.input_provenance)
        if pair.forward.input_components
        else tuple(
            example.stage.input_provenance[position]
            for position in pair.forward.tensor_argument_positions
        )
    )
    forward = pair.forward.rebind_examples(
        tuple(forward_arguments),
        input_provenance=forward_provenance,
    )
    forward = replace(forward, input_components=pair.forward.input_components)
    backward = pair.backward.rebind_examples(
        pair.backward.example_arguments,
        input_provenance=rebind_backward_input_provenance(pair, forward),
    )
    if len(roots) < pair.specialized_unit_tangent_count:
        raise CaptureError("specialized tangent count exceeds stage roots")
    return AotGraphPair(
        forward=forward,
        backward=backward,
        retention=pair.retention,
        saved_value_count=pair.saved_value_count,
        specialized_unit_tangent_count=pair.specialized_unit_tangent_count,
        gradient_provenance=tuple(
            provenance
            for value, provenance in zip(
                example.inputs, example.stage.input_provenance, strict=True
            )
            if isinstance(value, torch.Tensor)
        )
        if pair.gradient_provenance
        else (),
    )


__all__ = ["rebind_task_graph_pairs"]
