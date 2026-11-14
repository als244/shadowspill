"""Capture inference stages and expand physical tensor representations."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch
from torch.fx import GraphModule

from shadowspill.pytorch.capture.storage import (
    ExplicitMutation,
)
from shadowspill.pytorch.representations import (
    component_at,
    is_wrapper,
    map_tensor,
    tensor_components,
)

from .geometry import TaskInputProvenance
from .graph import GraphArtifact

if TYPE_CHECKING:
    from shadowspill.pytorch.partition import PartitionedExport


def capture_forward_stage_artifacts(
    partitioned: PartitionedExport,
) -> tuple[GraphArtifact, ...]:
    """Capture one structural inference contract for each stage occurrence.

    Partitioning itself stops at :class:`Stage`. This conversion belongs to
    graph capture because it adds the geometry-dependent task contract consumed by
    compilation and profiling.
    """

    from ..provenance import physical_input_provenance

    contracts: dict[str, GraphArtifact] = {}
    artifacts: list[GraphArtifact] = []
    for example in partitioned.stages:
        stage = example.stage
        key = GraphArtifact.input_compatibility_digest(
            graph_module=stage.graph_module,
            example_inputs=example.inputs,
            explicit_mutations=stage.mutations,
            input_provenance=stage.input_provenance,
        )
        cached = contracts.get(key)
        if cached is None:
            result = capture_inference_artifact(
                graph_module=stage.graph_module,
                example_inputs=example.inputs,
                explicit_mutations=stage.mutations,
                input_provenance=stage.input_provenance,
            )
            contracts[key] = result
        else:
            components = cached.input_components or tuple(
                (position, ()) for position in cached.tensor_argument_positions
            )
            arguments = tuple(
                component_at(cast(torch.Tensor, example.inputs[position]), path)
                for position, path in components
            )
            provenance = (
                physical_input_provenance(
                    example.inputs, example.stage.input_provenance
                )
                if cached.input_components
                else tuple(
                    example.stage.input_provenance[position]
                    for position in cached.tensor_argument_positions
                )
            )
            assert provenance is not None
            # Inputs and observed values belong to this occurrence; consumers
            # belong to the already verified physical graph.
            provenance = tuple(
                replace(current, consumer_targets=prior.consumer_targets)
                for current, prior in zip(
                    provenance, cached.input_provenance, strict=True
                )
            )
            result = cached.rebind_examples(arguments, input_provenance=provenance)
        artifacts.append(result)
    return tuple(artifacts)


def capture_inference_artifact(
    *,
    graph_module: GraphModule,
    example_inputs: tuple[object, ...],
    explicit_mutations: tuple[ExplicitMutation, ...] = (),
    input_provenance: tuple[TaskInputProvenance, ...] | None = None,
) -> GraphArtifact:
    """Lower logical representations only after inference semantics are captured."""
    components: tuple[tuple[int, tuple[str, ...]], ...] = ()
    if any(is_wrapper(value) for value in example_inputs):
        from torch.fx.experimental.proxy_tensor import make_fx

        from ..provenance import physical_input_provenance

        logical = example_inputs
        graph = graph_module
        components = tuple(
            (index, path)
            for index, value in enumerate(logical)
            if isinstance(value, torch.Tensor)
            for path, _ in tensor_components(value)
        )
        leaves = tuple(
            component
            for value in logical
            if isinstance(value, torch.Tensor)
            for _, component in tensor_components(value)
        )

        def execute(*values: torch.Tensor) -> object:
            mapped = {id(old): new for old, new in zip(leaves, values, strict=True)}
            return graph(
                *(
                    map_tensor(value, lambda leaf: mapped[id(leaf)])
                    if isinstance(value, torch.Tensor)
                    else value
                    for value in logical
                )
            )

        with torch.no_grad():
            graph_module = make_fx(
                execute, tracing_mode="fake", _allow_non_fake_inputs=True
            )(*leaves)
        explicit_mutations = tuple(
            replace(item, input_position=components.index((item.input_position, ())))
            for item in explicit_mutations
        )
        input_provenance = physical_input_provenance(logical, input_provenance)
        example_inputs = leaves
    result = GraphArtifact.capture(
        kind="inference",
        graph_module=graph_module,
        example_inputs=example_inputs,
        explicit_mutations=explicit_mutations,
        input_provenance=input_provenance,
    )
    return replace(result, input_components=components)
