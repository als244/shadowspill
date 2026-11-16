"""Example execution policy for token-independent text regions.

Checkpoint partitioning stays with the caller. Operator contracts and expert
targets are explicit inputs; unknown opaque operations are not certified here.
This workload helper uses the public generic axes/regions/lifetimes APIs.
"""

import operator
from dataclasses import dataclass

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from shadowspill.pytorch import (
    AxisChunks,
    AxisFrom,
    ChunkAxis,
    ChunkLocal,
    ChunkRegions,
    ExecutionLifetimes,
    LifetimeSelection,
)
from shadowspill.pytorch.partition import IsolateOperations
from shadowspill.task.inputs import TaskInputRole

from .chunking import is_barrier


def keep_local_intervals(graph, assignments, rules):
    """Merge cuts that would expose provider-local routing coordinates."""
    nodes = tuple(graph.graph.nodes)
    positions = {node: i for i, node in enumerate(nodes)}
    by_target = {rule.target: rule for rule in rules}
    local, starts = {}, {}
    for node in nodes:
        rule = by_target.get(str(node.target))
        if node.op == "placeholder":
            local[node] = False
        elif node.target is operator.getitem and node.args[0] in local:
            value = local[node.args[0]]
            local[node] = (
                value[node.args[1]] if isinstance(value, list | tuple) else value
            )
        elif rule is not None:
            flags = [
                isinstance(axis, ChunkLocal)
                or (
                    isinstance(axis, AxisFrom)
                    and any(
                        tree_flatten(local.get(node.args[axis.input_index], False))[0]
                    )
                )
                for axis in rule.output_axes
            ]
            _, spec = tree_flatten(node.meta.get("val"))
            local[node] = tree_unflatten(flags, spec)
        else:
            depends = any(
                flag
                for parent in node.all_input_nodes
                for flag in tree_flatten(local.get(parent, False))[0]
            )
            local[node] = tree_map(
                lambda value, depends=depends: (
                    depends if isinstance(value, torch.Tensor) else False
                ),
                node.meta.get("val"),
            )
        if any(tree_flatten(local[node])[0]) and node.users:
            starts[positions[node]] = max(positions[user] for user in node.users)
        # Stage ports are tensor leaves. Keep a tuple-producing operator with
        # its unpacking, even when a shared gradient join requested a cut in
        # between. Individual unpacked tensors may cross a later boundary.
        if isinstance(node.meta.get("val"), tuple | list):
            unpacked, pending = set(), [node]
            while pending:
                for user in pending.pop().users:
                    if user.target is operator.getitem and user not in unpacked:
                        unpacked.add(user)
                        pending.append(user)
            if unpacked:
                starts[positions[node]] = max(
                    starts.get(positions[node], -1),
                    *(positions[user] for user in unpacked),
                )
    active_end, previous, index, result = -1, None, -1, {}
    for position, node in enumerate(nodes):
        if node.name in assignments:
            requested = assignments[node.name]
            if index < 0 or (requested != previous and position > active_end):
                index += 1
            result[node.name] = index
            previous = requested
        active_end = max(active_end, starts.get(position, -1))
    return result


@dataclass(frozen=True)
class TextExecutionBoundaries:
    sequence_length: int
    rules: tuple
    inner_targets: tuple[str, ...] | None = None

    def assign_stages(self, graph_module, module):
        targets = self.inner_targets
        if targets is None:
            targets = tuple(
                str(n.target)
                for n in graph_module.graph.nodes
                if is_barrier(str(n.target)) or str(n.target) == "aten.add_.Tensor"
            )
        assigned = IsolateOperations(targets, base="whole").assign_stages(
            graph_module, module
        )
        # Joins of shared values and completed reductions run once, outside
        # token loops. This covers both parameter adjoints and forward loss /
        # metric combinations. Token-valued joins still carry their axis.
        previous, index, result = None, -1, {}
        for node in graph_module.graph.nodes:
            if node.name not in assigned:
                continue
            value = node.meta.get("val")
            shared = (
                str(node.target)
                in ("aten.add.Tensor", "aten.sub.Tensor", "aten.cat.default")
                and hasattr(value, "shape")
                and not any(
                    int(dim) >= self.sequence_length
                    and int(dim) % self.sequence_length == 0
                    for dim in value.shape
                )
            )
            key = ("combine", node.name) if shared else ("base", assigned[node.name])
            if key != previous:
                index += 1
            result[node.name], previous = index, key
        return keep_local_intervals(graph_module, result, self.rules)


@dataclass(frozen=True)
class TextExecutionChunks:
    sequence_length: int
    batch_size: int
    chunk_size: int
    expert_chunk_size: int
    rules: tuple
    expert_targets: tuple[str, ...]
    inner_targets: tuple[str, ...] = ()
    variants: tuple[str, ...] = ("recompute",)
    inner: bool = False

    def configure_stage(self, stage):
        if stage.terminal or any(
            is_barrier(op) or "embedding" in op or "packed_sequence" in op
            for op in stage.operators
        ):
            return None
        axes = {}
        batch, sequence = self.batch_size, self.sequence_length
        for index, (geometry, role) in enumerate(
            zip(stage.inputs, stage.input_roles, strict=True)
        ):
            if stage.input_domains and "tokens" in stage.input_domains[index]:
                axes[index] = stage.input_domains[index]["tokens"]
                continue
            if (
                self.inner
                or geometry is None
                or role
                in (
                    TaskInputRole.PARAMETER,
                    TaskInputRole.BUFFER,
                    # Parameter gradients and shared-statistic tangents may
                    # have a feature extent equal to the sequence length.
                    # Only inherited forward coordinates prove token slices.
                    TaskInputRole.GRADIENT,
                    TaskInputRole.TANGENT,
                )
            ):
                continue
            # Integer/control tensors can carry token coordinates too (for
            # example packed RoPE positions). Slice matching token geometry;
            # shared parameter/buffer tables and complete sequence mixers stay
            # outside this inference. CONTROL does not imply a shared axis.
            shape = geometry.shape
            if len(shape) > 1 and shape[:2] == (batch, sequence):
                axes[index] = ChunkAxis(1)
            elif len(shape) == 4 and shape[0] == batch and shape[2] == sequence:
                axes[index] = ChunkAxis(2)
            else:
                dims = [
                    dim for dim, size in enumerate(shape) if size == batch * sequence
                ]
                if len(dims) == 1:
                    axes[index] = ChunkAxis(dims[0], before=batch)
        if not axes:
            return None
        expert = any(op in self.expert_targets for op in stage.operators)
        size = self.expert_chunk_size if expert else self.chunk_size
        regions = None
        if not self.inner and expert and self.inner_targets and size > self.chunk_size:
            from dataclasses import replace

            regions = ChunkRegions(
                TextExecutionBoundaries(size, self.rules, self.inner_targets),
                replace(self, inner=True),
                variants=self.variants,
            )
        return AxisChunks(
            size, axes, self.variants, self.rules, domain="tokens", regions=regions
        )


def move_without_replay(info):
    """Move eligible ATen work toward use, without repeating arithmetic."""
    if info.kind != "backward":
        return None
    return LifetimeSelection(
        move=info.target.startswith("aten.") and info.effect_fence is None,
        barrier=is_barrier(info.target),
    )


_COMPACT_VIEWS = frozenset(
    "aten." + name
    for name in (
        "alias.default",
        "detach.default",
        "view.default",
        "_unsafe_view.default",
        "permute.default",
        "t.default",
        "transpose.int",
        "select.int",
        "slice.Tensor",
        "split.Tensor",
        "split_with_sizes.default",
        "unbind.int",
        "unsqueeze.default",
        "squeeze.default",
        "squeeze.dim",
        "squeeze.dims",
        "expand.default",
    )
) | {"<built-in function getitem>"}


def compact_without_replay(info):
    """Move work once and compact internal primal views with proven consumers."""
    if info.kind != "backward":
        return None
    return LifetimeSelection(
        move=info.target.startswith("aten.") and info.effect_fence is None,
        barrier=is_barrier(info.target),
        compact=info.phase == "primal" and info.target in _COMPACT_VIEWS,
        replay=False,
    )


def text_execution_chunks(*, lifetimes: ExecutionLifetimes | None = None, **options):
    """Select text chunks with compact value boundaries and no extra replay.

    The default supplies explicit FLA provider contracts, leaving unknown
    providers as fences. Pass an ExecutionLifetimes policy to replace this
    preset; an empty policy disables these lifetime transformations. Chunking
    and checkpoint retention remain separate controls.
    """
    if lifetimes is None:
        from shadowspill.integrations.mlops import fla_execution_lifetimes

        lifetimes = fla_execution_lifetimes(select=compact_without_replay)
    chunks = TextExecutionChunks(**options)
    return ChunkRegions(
        TextExecutionBoundaries(chunks.sequence_length, chunks.rules),
        chunks,
        variants=chunks.variants,
        lifetimes=lifetimes,
    )
