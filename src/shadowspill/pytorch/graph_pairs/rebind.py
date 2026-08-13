"""Bind one structural graph-pair portfolio to a stage occurrence."""

from __future__ import annotations

import torch

from shadowspill.pytorch.capture.aot import rebind_backward_input_provenance
from shadowspill.pytorch.capture.artifacts import AotGraphPair

from ..contracts import CaptureError
from ..partition.artifacts import StageExample
from .artifacts import GraphPairPortfolio, GraphPairVariant


def rebind_graph_pair_portfolio(
    portfolio: GraphPairPortfolio,
    example: StageExample,
) -> GraphPairPortfolio:
    """Replace occurrence-local values while preserving structural graph code."""

    return GraphPairPortfolio(
        structural_abi=portfolio.structural_abi,
        root_output_indices=portfolio.root_output_indices,
        variants=tuple(
            GraphPairVariant(
                item.option_id,
                item.memory_budget,
                _rebind_graph_pair(
                    item.pair,
                    example,
                    portfolio.root_output_indices,
                ),
            )
            for item in portfolio.variants
        ),
        reference_option_id=portfolio.reference_option_id,
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


__all__ = ["rebind_graph_pair_portfolio"]
