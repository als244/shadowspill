"""Slice explicit forward/VJP graphs while retaining their public contracts."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import Any

import torch
from torch._guards import detect_fake_mode
from torch.fx import Interpreter, Node
from torch.utils._pytree import tree_flatten, tree_map

from shadowspill.artifacts._values import digest
from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.gradients.accumulation import (
    accumulate_gradient_outputs,
)
from shadowspill.pytorch.capture.gradients.destinations import GradientDestinationRule
from shadowspill.pytorch.capture.storage import ExplicitMutation
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module
from shadowspill.task.inputs import TaskInputRole
from shadowspill.task.storage import StorageRootKind

from .broadcasts import align_broadcasts
from .contract import materialized_storage
from .geometry import ChunkAxis, ChunkTask, Layout, LocalLayout
from .policy import AxisChunks
from .rules import infer


class _Values(Interpreter):
    def run_node(self, node: Node) -> Any:
        value = super().run_node(node)
        node.meta["val"] = value
        return value


def describe(layout: Layout) -> dict[str, Any] | str | None:
    return asdict(layout) if isinstance(layout, (ChunkAxis, LocalLayout)) else layout


def chunk_artifact(
    artifact: GraphArtifact,
    inputs: tuple[Layout, ...],
    rows: int,
    config: AxisChunks,
    *,
    fuse_narrow_gradient_accumulation: bool = True,
    gradient_destination_rules: Sequence[GradientDestinationRule] = (),
    stage_id: str = "chunk",
    checkpoint_stage_id: str | None = None,
    variant: str | None = None,
) -> GraphArtifact:
    """Build reusable chunk graphs without differentiating an outer loop."""
    if len(inputs) != artifact.argument_count:
        raise CaptureError("chunk input mapping differs from task argument count")
    if artifact.input_components and any(
        path and layout is not None
        for (_, path), layout in zip(artifact.input_components, inputs, strict=True)
    ):
        raise CaptureError(
            "axis chunking of wrapped tensor components needs a representation rule"
        )
    explicit = {
        m.input_position
        for m in artifact.storage_contract.mutations
        if m.replacement_output_leaf is not None
    }
    for mutation in artifact.storage_contract.mutations:
        if (
            artifact.kind != "backward"
            or artifact.input_provenance[mutation.input_position].role
            is not TaskInputRole.GRADIENT
            or (
                mutation.input_position not in explicit
                and inputs[mutation.input_position] is not None
            )
        ):
            raise CaptureError(
                "axis chunking only supports explicit backward gradient mutations"
            )
    graph = copy_graph_module(artifact.graph_module)
    mode = detect_fake_mode(artifact.example_arguments)
    if mode is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode, torch.no_grad():
        full_values, _ = tree_flatten(_Values(graph).run(*artifact.example_arguments))
    for node in tuple(graph.graph.nodes):
        if str(node.target) != "aten._assert_tensor_metadata.default" or node.users:
            continue
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size")
        stride = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride")
        if size is None and stride is None:
            # Full capture checked these static dtype/device/layout facts.
            # Axis slicing preserves them; they need not keep dead values live.
            # Size/stride assertions remain because chunk geometry changes.
            graph.graph.erase_node(node)
    # Functionalized graphs can retain unused routing statistics. Only drop
    # explicit pure calls; a discarded random/effectful call still matters.
    graph.graph.eliminate_dead_code(
        is_impure_node=lambda node: (
            node.op != "call_function"
            or not (
                isinstance(node.target, torch._ops.OpOverload)
                or node.target is operator.getitem
            )
            or node.is_impure(impure_random=True)
        )
    )
    graph.recompile()
    literals = []
    for index, value in enumerate(full_values):
        if not isinstance(value, torch.Tensor) and value is not None:
            if not isinstance(value, (str, bool, int, float)):
                raise CaptureError(
                    "axis chunking requires portable scalar output literals"
                )
            literals.append((index, value))
    storage = artifact.storage_contract
    composed_storage = materialized_storage(artifact, full_values)
    layouts: dict[Node, Any] = {}
    rules = {rule.target: rule for rule in config.rules}
    outputs: tuple[Layout, ...] = ()
    broadcasts: dict[Node, ChunkAxis] = {}
    while True:
        positions = iter(inputs)
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                layouts[node] = next(positions)
            elif node.op == "output":
                outputs = tuple(
                    tree_flatten(
                        tree_map(
                            lambda n: layouts[n] if isinstance(n, Node) else None,
                            node.args[0],
                        )
                    )[0]
                )
            elif node.op == "call_function":
                try:
                    layouts[node] = (
                        broadcasts[node]
                        if node in broadcasts
                        else infer(node, layouts, rows, rules)
                    )
                except CaptureError as error:
                    operands = [(n.name, layouts.get(n)) for n in node.all_input_nodes]
                    raise CaptureError(
                        f"chunking {artifact.kind} node {node.name}: {error}; "
                        f"input layouts {operands}"
                    ) from error
            else:
                raise CaptureError(
                    f"axis chunking requires explicit operators, got {node.op}"
                )
        found = align_broadcasts(graph.graph, layouts, rows)
        if not found:
            break
        broadcasts.update(found)
    if any(isinstance(layout, LocalLayout) for layout in outputs):
        leaves, _ = tree_flatten(
            next(n for n in graph.graph.nodes if n.op == "output").args[0]
        )
        escaping = [
            (i, str(leaves[i]))
            for i, axis in enumerate(outputs)
            if isinstance(axis, LocalLayout)
        ]
        raise CaptureError(
            "chunk-local scratch cannot escape or be saved across the stage boundary; "
            f"restore axis order inside the region and select recompute: {escaping}"
        )
    children = []
    accumulating_children = []
    for root in storage.roots:
        views = [view for view in storage.output_views if view.root_id == root.root_id]
        if len(views) > 1 and any(outputs[v.leaf_index] == "sum" for v in views):
            raise CaptureError("aliased chunk reductions require one canonical output")
    accumulation_leaves = tuple(
        view.leaf_index
        for view in storage.output_views
        if artifact.kind == "backward"
        and rows > config.chunk_size
        and outputs[view.leaf_index] == "sum"
        and storage.roots[view.root_id].kind is StorageRootKind.FRESH
        and not any(
            size > 1 and stride == 0
            for size, stride in zip(view.shape, view.stride, strict=True)
        )
    )
    sizes = tuple(
        dict.fromkeys((min(rows, config.chunk_size), rows % config.chunk_size))
    )
    for size in sizes:
        if size == 0:
            continue
        child = copy_graph_module(graph)
        for old, node in zip(graph.graph.nodes, child.graph.nodes, strict=True):
            layout = layouts.get(old)
            if config.domain is not None:
                # Preserve proven coordinates through inner cuts, including
                # broadcasts of shared cotangents. Their producer can have no
                # sliced input from which an inner region could infer a domain.
                domains = dict(node.meta.get("execution_domains", {}))
                if isinstance(layout, ChunkAxis):
                    domains[config.domain] = layout
                else:
                    domains.pop(config.domain, None)
                node.meta["execution_domains"] = domains
            if isinstance(layout, LocalLayout):
                layout = layout.axis
            target = str(old.target)
            if isinstance(layout, ChunkAxis) and target in {
                "aten.view.default",
                "aten.reshape.default",
                "aten._unsafe_view.default",
                "aten.expand.default",
                "aten.broadcast_to.default",
                "aten.slice_backward.default",
                "aten.select_backward.default",
            }:
                node.args = (
                    node.args[0],
                    layout.shape(tuple(old.meta["val"].shape), size),
                    *node.args[2:],
                )
        child.graph.lint()
        child.recompile()
        with mode, torch.no_grad():
            arguments = tuple(
                axis.slice(value, rows, 0, size).clone(
                    memory_format=torch.preserve_format
                )
                if isinstance(axis, ChunkAxis) and isinstance(value, torch.Tensor)
                else value
                for value, axis in zip(artifact.example_arguments, inputs, strict=True)
            )
            try:
                child_values, _ = tree_flatten(_Values(child).run(*arguments))
                for index, (full, value, layout) in enumerate(
                    zip(full_values, child_values, outputs, strict=True)
                ):
                    if isinstance(full, torch.Tensor):
                        expected = (
                            layout.shape(tuple(full.shape), size)
                            if isinstance(layout, ChunkAxis)
                            else tuple(full.shape)
                        )
                        if (
                            not isinstance(value, torch.Tensor)
                            or tuple(value.shape) != expected
                            or value.dtype != full.dtype
                        ):
                            raise CaptureError(
                                f"output {index} violates its declared chunk geometry: "
                                f"expected {expected} with {full.dtype}"
                            )
                    elif type(value) is not type(full) or value != full:
                        raise CaptureError(
                            f"output {index} changes its shared scalar literal"
                        )
                captured = GraphArtifact.capture(
                    kind=artifact.kind,
                    graph_module=child,
                    example_inputs=arguments,
                    input_provenance=artifact.input_provenance,
                    explicit_mutations=tuple(
                        ExplicitMutation(
                            m.input_position, m.replacement_output_leaf, m.argument_name
                        )
                        for m in artifact.storage_contract.mutations
                        if m.replacement_output_leaf is not None
                    ),
                )
            except Exception as error:
                raise CaptureError(f"row chunk shape {size} failed: {error}") from error
        if not accumulation_leaves or size == min(rows, config.chunk_size):
            children.append((size, captured))
        # Do not package a specialization the complete task never launches:
        # build freezes kernels from the actual first and subsequent chunks.
        if accumulation_leaves and (
            size == rows % config.chunk_size or rows >= 2 * config.chunk_size
        ):
            accumulating_children.append(
                (
                    size,
                    accumulate_gradient_outputs(
                        captured,
                        accumulation_leaves,
                        fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
                        gradient_destination_rules=gradient_destination_rules,
                    ),
                )
            )
    if config.regions is not None:
        from .regions import region_artifact

        assert config.domain is not None
        inherited = tuple(
            {config.domain: layout} if isinstance(layout, ChunkAxis) else {}
            for layout in inputs
        )
        reductions = tuple(
            frozenset((config.domain,)) if layout == "sum" else frozenset()
            for layout in inputs
        )

        def compose(child: GraphArtifact) -> GraphArtifact:
            assert config.regions is not None
            assert config.domain is not None
            extra = child.argument_count - len(inputs)
            return region_artifact(
                child,
                config.regions,
                stage_id=stage_id,
                checkpoint_stage_id=checkpoint_stage_id,
                variant=variant,
                input_domains=(*inherited, *({} for _ in range(extra))),
                input_reductions=(*reductions, *(frozenset() for _ in range(extra))),
                domain_rules={config.domain: config.rules},
                fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
                gradient_destination_rules=gradient_destination_rules,
                # Sliced activation cotangents are assembled in their own
                # coordinates; only shared/reduced outputs may be donated to
                # a canonical destination across these inner regions.
                gradient_leaves=tuple(
                    i
                    for i, layout in enumerate(outputs)
                    if not isinstance(layout, (ChunkAxis, LocalLayout))
                ),
            )

        children = [(size, compose(child)) for size, child in children]
        accumulating_children = [
            (size, compose(child)) for size, child in accumulating_children
        ]
    identity = {
        "schema": 5,
        "graph": artifact.compatibility_digest,
        "rows": rows,
        "chunk_size": config.chunk_size,
        "domain": config.domain,
        "inputs": [describe(x) for x in inputs],
        "outputs": [describe(x) for x in outputs],
        "rules": [asdict(rule) for rule in config.rules],
        "children": [(size, child.compatibility_digest) for size, child in children],
        "accumulation_leaves": list(accumulation_leaves),
        "accumulating_children": [
            (size, child.compatibility_digest) for size, child in accumulating_children
        ],
        "storage_contract": composed_storage.compatibility_digest,
        "output_literals": literals,
    }
    if config.regions is not None:
        identity["child_regions"] = {
            name: [
                (size, child.regions.identity)
                for size, child in group
                if child.regions is not None
            ]
            for name, group in (
                ("creators", children),
                ("accumulators", accumulating_children),
            )
        }
    return replace(
        artifact,
        compatibility_digest=digest(identity),
        chunking=ChunkTask(
            rows,
            config.chunk_size,
            inputs,
            outputs,
            tuple(children),
            tuple(accumulating_children),
            identity,
            composed_storage,
        ),
    )
