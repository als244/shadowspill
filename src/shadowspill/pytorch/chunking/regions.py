"""Execution regions within one explicit checkpoint task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any

import torch
from torch._guards import detect_fake_mode
from torch.fx import Node
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_flatten, tree_map

from shadowspill.artifacts._values import digest
from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact, TaskInputProvenance
from shadowspill.pytorch.capture.gradients import GradientDestinationRule
from shadowspill.pytorch.capture.live_storage import live_storage_identity
from shadowspill.pytorch.capture.storage import ExplicitMutation
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module
from shadowspill.pytorch.partition.policy import resolve_partition_assignments
from shadowspill.pytorch.partition.sources import stage_value_source
from shadowspill.pytorch.partition.split import observe_split_root
from shadowspill.task.inputs import TaskInputRole
from shadowspill.task.storage import StorageRootKind, TaskStorageContract

from .contract import boundary_metadata, materialized_storage
from .geometry import REDUCED, ChunkAxis, Layout
from .policy import ChunkRegions, OperatorRule


@dataclass(frozen=True, slots=True)
class RegionTask:
    children: tuple[GraphArtifact, ...]
    identity: dict[str, Any]
    storage_contract: TaskStorageContract
    input_domains: tuple[dict[str, ChunkAxis], ...]
    output_domains: tuple[dict[str, ChunkAxis], ...]
    domain_rules: dict[str, tuple[OperatorRule, ...]]
    lifetime_report: dict[str, Any] | None = None


def region_artifact(
    artifact: GraphArtifact,
    policy: ChunkRegions,
    *,
    stage_id: str,
    variant: str | None,
    checkpoint_stage_id: str | None = None,
    input_domains: tuple[Mapping[str, ChunkAxis], ...] = (),
    input_reductions: tuple[frozenset[str], ...] = (),
    domain_rules: Mapping[str, tuple[OperatorRule, ...]] | None = None,
    fuse_narrow_gradient_accumulation: bool = True,
    gradient_destination_rules: Sequence[GradientDestinationRule] = (),
    gradient_leaves: tuple[int, ...] | None = None,
) -> GraphArtifact:
    """Split after differentiation; child edges never become checkpoint saves."""
    from shadowspill.pytorch.capture.artifacts import TensorGeometry
    from shadowspill.pytorch.partition.metadata import module_paths

    from .domains import inherit_nested_rules, merge_rules, whole_output_domains
    from .inputs import resolve_input_axes
    from .policy import AxisChunks, StageInfo
    from .transform import chunk_artifact

    contracts = dict(domain_rules or {})

    if any(path for _, path in artifact.input_components):
        raise CaptureError("execution regions require ordinary tensor components")
    graph = copy_graph_module(artifact.graph_module)
    _shared_provenance(graph, artifact)
    lifetime_identity = lifetime_report = None
    if policy.lifetimes is not None:
        from .lifetimes import apply_lifetimes

        lifetime_identity, lifetime_report = apply_lifetimes(
            graph,
            artifact,
            policy.lifetimes,
            partition=policy.partition,
            stage_id=stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            variant=variant,
            gradient_leaves=gradient_leaves,
        )
    from .accumulation import route_gradient_destinations

    route_gradient_destinations(
        graph,
        artifact,
        fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
        gradient_destination_rules=gradient_destination_rules,
        gradient_leaves=gradient_leaves,
    )
    assignments, _ = resolve_partition_assignments(graph, graph, policy.partition)
    root = split_module(
        graph, graph, lambda node: assignments[node], keep_original_order=True
    )
    mode = detect_fake_mode(artifact.example_arguments)
    if mode is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode, torch.no_grad():
        split = observe_split_root(root, artifact.example_arguments)
        full_values, _ = tree_flatten(graph(*artifact.example_arguments))
    parent_roles = {
        live_storage_identity(value): provenance
        for value, provenance in zip(
            artifact.example_arguments, artifact.input_provenance, strict=True
        )
        if isinstance(value, torch.Tensor)
    }
    children = []
    instructions: list[dict[str, Any]] = []
    next_slot = artifact.argument_count
    output_slots: list[list[int]] = []
    domains = (
        [dict(item) for item in input_domains]
        if input_domains
        else [{} for _ in range(artifact.argument_count)]
    )
    if len(domains) != artifact.argument_count:
        raise CaptureError("execution row-domain input arity differs from the task")
    reductions = (
        list(input_reductions)
        if input_reductions
        else [frozenset() for _ in range(artifact.argument_count)]
    )
    if len(reductions) != artifact.argument_count:
        raise CaptureError("execution reduction input arity differs from the task")

    def source_slot(source: Any) -> int:
        if source.root_input_index is not None:
            return int(source.root_input_index)
        return int(
            output_slots[source.producer_stage_index][source.producer_output_index]
        )

    for index, record in enumerate(split.stages):
        if any(not isinstance(value, torch.Tensor) for value in record.inputs):
            raise CaptureError(
                "execution cuts must pass flat tensors; "
                "keep tuple unpacking with its producer"
            )
        tensor_inputs = tuple(
            value for value in record.inputs if isinstance(value, torch.Tensor)
        )
        placeholders = [
            n for n in record.graph_module.graph.nodes if n.op == "placeholder"
        ]
        provenance = tuple(
            node.meta.get("execution_provenance")
            or parent_roles.get(live_storage_identity(value))
            or TaskInputProvenance(
                TaskInputRole.ACTIVATION
                if value.is_floating_point()
                else TaskInputRole.CONTROL
            )
            for node, value in zip(placeholders, tensor_inputs, strict=True)
        )
        values, _ = tree_flatten(record.output)
        # Mark explicit gradient handoffs when both ports remain in this region.
        # Schema mutations without a returned port remain observable mutations.
        explicit = []
        for mutation in artifact.storage_contract.mutations:
            if mutation.replacement_output_leaf is None:
                continue
            parent_value = artifact.example_arguments[mutation.input_position]
            assert isinstance(parent_value, torch.Tensor)
            for position, value in enumerate(tensor_inputs):
                if live_storage_identity(value) != live_storage_identity(parent_value):
                    continue
                for leaf, output in enumerate(values):
                    if isinstance(output, torch.Tensor) and live_storage_identity(
                        output
                    ) == live_storage_identity(value):
                        explicit.append(
                            ExplicitMutation(position, leaf, mutation.argument_name)
                        )
                        break
        with mode, torch.no_grad():
            child = GraphArtifact.capture(
                kind=artifact.kind,
                graph_module=record.graph_module,
                example_inputs=record.inputs,
                input_provenance=provenance,
                explicit_mutations=tuple(explicit),
            )
            # A destination created inside the parent has no parent input
            # mutation to inherit. Once a cut exposes it as a child input,
            # retain its returned alias as an explicit handoff too. Sliced
            # gradient destinations need that return to assemble their updates.
            inherited_count = len(explicit)
            handed_off = {m.input_position for m in explicit}
            for mutation in child.storage_contract.mutations:
                position = mutation.input_position
                if (
                    artifact.kind != "backward"
                    or mutation.replacement_output_leaf is not None
                    or position in handed_off
                    or provenance[position].role is not TaskInputRole.GRADIENT
                ):
                    continue
                for view in child.storage_contract.output_views:
                    storage_root = child.storage_contract.roots[view.root_id]
                    if (
                        storage_root.kind is StorageRootKind.INPUT
                        and storage_root.source_input == position
                    ):
                        explicit.append(
                            ExplicitMutation(
                                position,
                                view.leaf_index,
                                str(placeholders[position].target),
                            )
                        )
                        handed_off.add(position)
                        break
            if len(explicit) > inherited_count:
                child = GraphArtifact.capture(
                    kind=artifact.kind,
                    graph_module=record.graph_module,
                    example_inputs=record.inputs,
                    input_provenance=provenance,
                    explicit_mutations=tuple(explicit),
                )
        input_slots = [source_slot(source) for source in record.input_sources]
        inherited = [domains[slot] for slot in input_slots]
        config = policy.chunking.configure_stage(
            StageInfo(
                f"{stage_id}/region_{index}",
                module_paths(child.graph_module),
                tuple(TensorGeometry.from_tensor(x) for x in tensor_inputs),
                tuple(x.role for x in provenance),
                child.operator_targets,
                False,
                artifact.kind,
                checkpoint_stage_id or stage_id,
                child.compatibility_digest,
                tuple(MappingProxyType(dict(item)) for item in inherited),
            )
        )
        if config is not None:
            if not isinstance(config, AxisChunks):
                raise TypeError("configure_stage must return AxisChunks or None")
            if variant is not None and variant not in config.variants:
                raise CaptureError(
                    f"execution region {index} excludes outer variant {variant}; "
                    "set ChunkRegions.variants to the intended checkpoint choices"
                )
            if config.domain is not None:
                merge_rules(contracts, config.domain, config.rules)
                declared = dict(config.inputs)
                for position, available in enumerate(inherited):
                    axis = available.get(config.domain)
                    if axis is None:
                        continue
                    selected = declared.get(position, axis)
                    selected = (
                        ChunkAxis(selected) if isinstance(selected, int) else selected
                    )
                    from .rules import _same_axis

                    if (
                        not -tensor_inputs[position].ndim
                        <= selected.dim
                        < tensor_inputs[position].ndim
                    ):
                        raise CaptureError(
                            f"invalid chunk dimension for execution input {position}"
                        )
                    normalized = replace(
                        selected, dim=selected.dim % tensor_inputs[position].ndim
                    )
                    if not _same_axis(
                        normalized, axis, tuple(tensor_inputs[position].shape)
                    ):
                        raise CaptureError(
                            f"execution input {position} disagrees with "
                            f"chunk domain {config.domain!r}"
                        )
                    declared[position] = axis
                config = replace(config, inputs=declared)
            axes, rows = resolve_input_axes(child, config)
            # A subtotal from an earlier region contributes once to this
            # region's loop. A mutation destination instead persists through
            # every iteration and must never be replaced with a zero buffer.
            destinations = {m.input_position for m in child.storage_contract.mutations}
            layouts: tuple[Layout, ...] = tuple(
                REDUCED
                if config.domain in reductions[slot] and position not in destinations
                else axis
                for position, (slot, axis) in enumerate(
                    zip(input_slots, axes, strict=True)
                )
            )
            child = chunk_artifact(
                child,
                layouts,
                rows,
                config,
                fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
                gradient_destination_rules=gradient_destination_rules,
                stage_id=f"{stage_id}/region_{index}",
                checkpoint_stage_id=checkpoint_stage_id or stage_id,
                variant=variant,
            )
            if config.domain is not None:
                for slot, axis in zip(input_slots, axes, strict=True):
                    if isinstance(axis, ChunkAxis):
                        domains[slot][config.domain] = axis
        inherit_nested_rules(child, contracts)
        children.append(child)
        slots = list(range(next_slot, next_slot + child.output_count))
        next_slot += child.output_count
        output_slots.append(slots)
        output_domains: list[dict[str, ChunkAxis]] = [{} for _ in slots]
        output_reductions: list[frozenset[str]] = [frozenset() for _ in slots]
        if config is not None and config.domain is not None:
            assert child.chunking is not None
            for leaf, (item, layout) in enumerate(
                zip(output_domains, child.chunking.outputs, strict=True)
            ):
                if isinstance(layout, ChunkAxis):
                    item[config.domain] = layout
                elif layout == REDUCED:
                    output_reductions[leaf] = frozenset((config.domain,))
        elif config is None:
            output_domains = whole_output_domains(child, inherited, contracts)
            output_reductions = _shared_reduction_outputs(
                child,
                tuple(reductions[slot] for slot in input_slots),
                inherited,
            )
        # The enclosing chunk transform has already proved coordinates for
        # internal values. A broadcast from a shared scalar/vector has no
        # sliced input, but its output still needs slicing at the next cut.
        output_nodes, _ = tree_flatten(
            next(n for n in record.graph_module.graph.nodes if n.op == "output").args[0]
        )
        for node, known in zip(output_nodes, output_domains, strict=True):
            if isinstance(node, Node):
                for name, axis in node.meta.get("execution_domains", {}).items():
                    if name in contracts:
                        from .rules import _same_axis

                        if name in known and not _same_axis(
                            known[name], axis, tuple(node.meta["val"].shape)
                        ):
                            raise CaptureError(
                                f"execution output disagrees with chunk domain {name!r}"
                            )
                        known[name] = axis
        domains.extend(output_domains)
        reductions.extend(output_reductions)
        instructions.append(
            {
                "inputs": input_slots,
                "outputs": slots,
                "boundary": boundary_metadata(child, values),
            }
        )
        if child.chunking is not None:
            instructions[-1]["row_chunks"] = child.chunking.identity
    output_node = next(n for n in root.graph.nodes if n.op == "output")
    leaves, _ = tree_flatten(output_node.args[0])
    outputs = []
    for leaf in leaves:
        if isinstance(leaf, Node):
            outputs.append(
                {
                    "slot": source_slot(
                        stage_value_source(
                            leaf,
                            placeholder_index=split.placeholder_index,
                            stage_index_by_node=split.stage_index_by_node,
                        )
                    )
                }
            )
        else:
            outputs.append({"literal": leaf})
    last_use = {}
    for index, instruction in enumerate(instructions):
        for slot in (*instruction["inputs"], *instruction["outputs"]):
            last_use[slot] = index
    for binding in outputs:
        if "slot" in binding:
            last_use[binding["slot"]] = len(instructions)
    for index, instruction in enumerate(instructions):
        instruction["release"] = [
            slot for slot, last in last_use.items() if last == index
        ]
    storage = materialized_storage(artifact, full_values)
    parent_inputs = tuple(domains[: artifact.argument_count])
    parent_outputs = tuple(
        dict(domains[item["slot"]]) if "slot" in item else {} for item in outputs
    )
    identity = {
        "schema": 1,
        "graph": artifact.compatibility_digest,
        "children": [child.compatibility_digest for child in children],
        "regions": instructions,
        "outputs": outputs,
        "slots": next_slot,
        "boundary": boundary_metadata(artifact, full_values),
        "input_domains": [
            {name: asdict(axis) for name, axis in item.items()}
            for item in parent_inputs
        ],
        "output_domains": [
            {name: asdict(axis) for name, axis in item.items()}
            for item in parent_outputs
        ],
        "domain_rules": {
            name: [asdict(rule) for rule in rules]
            for name, rules in sorted(contracts.items())
        },
    }
    if lifetime_identity is not None:
        identity["lifetimes"] = lifetime_identity
    return replace(
        artifact,
        compatibility_digest=digest(identity),
        regions=RegionTask(
            tuple(children),
            identity,
            storage,
            parent_inputs,
            parent_outputs,
            contracts,
            lifetime_report,
        ),
    )


def _shared_reduction_outputs(
    artifact: GraphArtifact,
    inputs: tuple[frozenset[str], ...],
    domains: list[dict[str, ChunkAxis]],
) -> list[frozenset[str]]:
    """Preserve linear subtotals through whole regions until aggregation."""
    from .rules import infer

    result: list[frozenset[str]] = [frozenset() for _ in range(artifact.output_count)]
    names = set().union(*inputs) - set().union(*(set(item) for item in domains))
    destinations = {m.input_position for m in artifact.storage_contract.mutations}
    for name in sorted(names):
        positions = iter(
            REDUCED if name in item and i not in destinations else None
            for i, item in enumerate(inputs)
        )
        layouts: dict[Node, Any] = {}
        for node in artifact.graph_module.graph.nodes:
            if node.op == "placeholder":
                layouts[node] = next(positions)
            elif node.op == "call_function":
                # There is no axis-bearing operand in this domain. The extent
                # cannot affect a shared linear operation's layout inference.
                try:
                    layouts[node] = infer(node, layouts, 1, {})
                except CaptureError:
                    # This region runs once on completed values. An operation
                    # without an additive contract ends subtotal provenance;
                    # it does not need to be separable to execute whole.
                    layouts[node] = tree_map(lambda _: None, node.meta.get("val"))
            elif node.op == "output":
                outputs, _ = tree_flatten(
                    tree_map(
                        lambda n, layouts=layouts: (
                            layouts[n] if isinstance(n, Node) else None
                        ),
                        node.args[0],
                    )
                )
                for leaf, layout in enumerate(outputs):
                    if layout == REDUCED:
                        result[leaf] |= frozenset((name,))
    return result


def _shared_provenance(graph: torch.fx.GraphModule, artifact: GraphArtifact) -> None:
    """Casts and derived shared weights remain shared at an execution cut.

    Storage identity alone loses this fact for a cast or contiguous copy. A
    projected feature extent can equal the sequence length; geometry is not
    evidence that this derived parameter contains token rows.
    """
    inputs = iter(artifact.input_provenance)
    shared = {TaskInputRole.PARAMETER, TaskInputRole.BUFFER, TaskInputRole.CONSTANT}
    for node in graph.graph.nodes:
        if node.op == "placeholder":
            node.meta["execution_provenance"] = next(inputs)
        elif node.op == "call_function" and node.all_input_nodes:
            provenance = [
                n.meta.get("execution_provenance") for n in node.all_input_nodes
            ]
            if all(p is not None and p.role in shared for p in provenance):
                roles = {p.role for p in provenance}
                role = (
                    TaskInputRole.PARAMETER
                    if TaskInputRole.PARAMETER in roles
                    else TaskInputRole.BUFFER
                    if TaskInputRole.BUFFER in roles
                    else TaskInputRole.CONSTANT
                )
                node.meta["execution_provenance"] = TaskInputProvenance(role)
