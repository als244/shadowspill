"""Preserve logical output geometry while allocating distributed storage padding."""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace

from torch._guards import detect_fake_mode

from shadowspill.pytorch.operators.materialize import PAD_STORAGE

from .artifacts import GraphArtifact
from .gradients.nodes import _record_value
from .storage import ExplicitMutation
from .torch_deprecations import copy_graph_module


def pad_output_storage(
    artifact: GraphArtifact, multiples: Mapping[int, int]
) -> GraphArtifact:
    """Allocate equal-shard padding inside the measured task, keeping tensor shapes."""
    if not multiples:
        return artifact
    graph_module = copy_graph_module(artifact.graph_module)
    graph = graph_module.graph
    output = next(node for node in graph.nodes if node.op == "output")
    leaves = list(output.args[0])
    mode = detect_fake_mode(artifact.example_arguments)
    changed = False
    for index, multiple in multiples.items():
        produced = leaves[index]
        if produced is None:
            continue
        value = produced.meta["val"]
        elements = value.untyped_storage().nbytes() // value.element_size()
        padded = (elements + multiple - 1) // multiple * multiple
        if padded == elements:
            continue
        with graph.inserting_before(output):
            node = graph.call_function(PAD_STORAGE, args=(produced, padded))
        with mode if mode is not None else nullcontext():
            _record_value(node, PAD_STORAGE(value, padded))
        leaves[index] = node
        changed = True
    if not changed:
        return artifact
    output.args = (tuple(leaves),)
    graph.lint()
    graph_module.recompile()
    result = GraphArtifact.capture(
        kind=artifact.kind,
        graph_module=graph_module,
        example_inputs=artifact.example_arguments,
        input_provenance=artifact.input_provenance,
        explicit_mutations=tuple(
            ExplicitMutation(
                m.input_position, m.replacement_output_leaf, m.argument_name
            )
            for m in artifact.storage_contract.mutations
            if m.replacement_output_leaf is not None
        ),
    )
    return replace(result, input_components=artifact.input_components)
