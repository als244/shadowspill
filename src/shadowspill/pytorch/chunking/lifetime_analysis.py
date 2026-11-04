"""Semantic ownership and effect fences for future execution scheduling.

This analysis does not rewrite a task. Dispatcher contracts supply ownership;
fresh fake execution supplies geometry. Captured allocation addresses are never
used to merge intermediate roots. The byte inventory excludes inputs, opaque
workspace and allocator overhead, so it is not physical admission evidence.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch.fx import GraphModule, Node
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.schema import operator_alias_contract
from shadowspill.pytorch.capture.storage import capture_task_storage_contract
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module
from shadowspill.task.storage import StorageRootKind, TaskStorageContract

from .lifetime_policy import ReadonlyOperator

# These exact ATen overloads always alias their labelled input. A dispatcher
# alias annotation alone permits a fresh return (e.g. reshape/contiguous), so it
# is not enough to certify a lifetime transformation. Unknown forms are refused.
_ALWAYS_ALIAS = frozenset(
    "aten." + name
    for name in (
        "alias.default",
        "as_strided.default",
        "detach.default",
        "diagonal.default",
        "expand.default",
        "permute.default",
        "select.int",
        "slice.Tensor",
        "split.Tensor",
        "split_with_sizes.default",
        "squeeze.default",
        "squeeze.dim",
        "squeeze.dims",
        "t.default",
        "transpose.int",
        "unbind.int",
        "unsqueeze.default",
        "view.default",
        "view.dtype",
        "_reshape_alias.default",
        "add_.Tensor",
        "copy_.default",
        "zero_.default",
    )
)


@dataclass(frozen=True)
class LifetimeGraph:
    """Read/write roots and external alias obligations in the original graph."""

    storage: TaskStorageContract
    values: Mapping[Node, frozenset[int]]
    reads: Mapping[Node, frozenset[int]]
    writes: Mapping[Node, frozenset[int]]
    fences: Mapping[Node, str]
    readonly_contracts: Mapping[str, str]
    external_roots: frozenset[int]
    order: tuple[Node, ...]

    def live_roots(self, barrier: Node) -> tuple[int, ...]:
        """Internal storage roots live immediately before a selected operation."""
        rank = {node: i for i, node in enumerate(self.order)}
        if barrier not in rank:
            raise ValueError("lifetime barrier does not belong to the analyzed graph")
        names = {node.name: node for node in self.order}
        last: dict[int, int] = {}
        for node, roots in self.reads.items():
            for root in roots:
                last[root] = max(last.get(root, -1), rank[node])
        return tuple(
            root.root_id
            for root in self.storage.roots
            if root.kind is StorageRootKind.FRESH
            and root.producer_node in names
            and rank[names[root.producer_node]]
            < rank[barrier]
            <= last.get(root.root_id, -1)
        )

    def live_bytes(self, barrier: Node) -> int:
        """Semantic storage spans, excluding fixed inputs and opaque workspace."""
        return sum(
            self.storage.roots[root].minimum_span_bytes
            for root in self.live_roots(barrier)
        )


def _effects(
    nodes: Sequence[Node], contracts: Mapping[str, ReadonlyOperator]
) -> tuple[dict[Node, str], dict[str, str]]:
    fences, resolved = {}, {}
    for node in nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op == "call_function" and node.target is operator.getitem:
            # Structural tuple/list extraction is qualified below. Tensor
            # indexing or arbitrary Python objects are not structural extraction.
            source = node.args[0]
            value = source.meta.get("val") if isinstance(source, Node) else source
            if not isinstance(value, (tuple, list)) or not isinstance(
                node.args[1], int
            ):
                raise CaptureError("lifetime analysis requires structural getitem")
            continue
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            # Do not evaluate arbitrary Python while obtaining fake geometry.
            raise CaptureError(
                "lifetime analysis requires dispatcher operations and structural "
                f"getitem; unsupported operation: {node.op} {node.target}"
            )
        target = str(node.target)
        mutation = node.target._schema.is_mutable
        impure = node.is_impure(impure_random=True)
        if target in contracts:
            if mutation or impure:
                raise CaptureError(
                    f"readonly contract cannot override mutation/RNG/effects: {target}"
                )
            resolved[target] = contracts[target].version
        if mutation:
            fences[node] = "ordered mutation"
        elif impure:
            fences[node] = "RNG or other observable effect"
        elif node.target.namespace != "aten" and target not in contracts:
            fences[node] = "opaque operator requires a readonly effect contract"
    return fences, resolved


def _check_aliases(graph: GraphModule) -> None:
    for node in graph.graph.nodes:
        if node.op == "call_function" and node.target is operator.getitem:
            source, index = node.args
            value = source.meta.get("val") if isinstance(source, Node) else source
            assert isinstance(value, (tuple, list))  # Qualified by _effects.
            if index < 0:
                index += len(value)
            if not 0 <= index < len(value):
                raise CaptureError("lifetime tuple/list index is outside its return")
            node.args = (source, index)
            continue
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            continue
        if node.target is torch.ops.aten._unsafe_view.default:
            # _unsafe_view deliberately lacks the normal alias annotation, but
            # always aliases storage. Normalize only the inspection copy; neither
            # the executable nor the general task storage contract is changed.
            node.target = torch.ops.aten.view.default
        alias = operator_alias_contract(node.target._schema)
        if (
            any(result.labels for result in alias.returns)
            and str(node.target) not in _ALWAYS_ALIAS
        ):
            raise CaptureError(
                "lifetime analysis needs an always-alias storage contract for "
                f"{node.target}; conditional or unknown aliases cannot be assumed"
            )
    graph.recompile()


def analyze_lifetimes(
    graph: GraphModule,
    example_inputs: tuple[object, ...],
    *,
    readonly: tuple[ReadonlyOperator, ...] = (),
) -> LifetimeGraph:
    """Analyze all intermediate storage without changing the executable graph.

    Unknown opaque effects are fences. Conditional or uncertified aliases are
    an explicit refusal, including when their operator is declared readonly.
    Input alias groups come from the existing task contract capture path.
    """
    contracts: dict[str, ReadonlyOperator] = {}
    for contract in readonly:
        previous = contracts.setdefault(contract.target, contract)
        if previous != contract:
            raise ValueError(f"conflicting readonly contracts for {contract.target}")
    originals = tuple(graph.graph.nodes)
    fences, resolved = _effects(originals, contracts)
    inspection = copy_graph_module(graph)
    _check_aliases(inspection)
    copied = tuple(inspection.graph.nodes)
    output = next(node for node in copied if node.op == "output")
    tensors, owners = [], []

    def expose(node: Node, value: object, owner: Node) -> None:
        if isinstance(value, torch.Tensor):
            tensors.append(node)
            owners.append(owner)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if isinstance(child, (list, tuple)):
                    # The task contract's result identity is a single return
                    # index. Nested fresh-return paths need a stronger contract;
                    # treating their unpacking as the allocation would hide the
                    # earlier lifetime at the opaque producer.
                    raise CaptureError(
                        "nested lifetime return trees require an explicit "
                        "storage contract; flat tuple/list returns are supported"
                    )
                with inspection.graph.inserting_before(output):
                    selected = inspection.graph.call_function(
                        operator.getitem, (node, index)
                    )
                selected.meta["val"] = child
                expose(selected, child, owner)
        elif value is not None and not isinstance(
            value, (int, float, bool, torch.SymInt, torch.SymFloat, torch.SymBool)
        ):
            raise CaptureError(f"unsupported lifetime value tree: {type(value)}")

    for original, node in zip(originals, copied, strict=True):
        if node.op != "output":
            if "val" not in node.meta:
                schema = getattr(node.target, "_schema", None)
                if schema is None or schema.returns:
                    raise CaptureError(
                        f"missing captured lifetime geometry: {node.name}"
                    )
                # Generated in-place gradient producers have no return value
                # and may carry no val metadata. Their dispatcher contract is
                # sufficient to establish that there is no output geometry;
                # their reads/writes and effect fence still participate below.
                node.meta["val"] = None
            expose(node, node.meta["val"], original)
    output.args = (tuple(tensors),)
    inspection.graph.lint()
    inspection.recompile()
    storage = capture_task_storage_contract(inspection, example_inputs)
    by_node: dict[Node, set[int]] = defaultdict(set)
    for owner, view in zip(owners, storage.output_views, strict=True):
        by_node[owner].add(view.root_id)
    values = {node: frozenset(by_node[node]) for node in originals}
    reads, writes = {}, {}
    for node in originals:
        # Unpacking one tuple/list return must not extend all sibling lifetimes.
        reads[node] = (
            values[node]
            if node.op == "call_function" and node.target is operator.getitem
            else frozenset().union(*(values[parent] for parent in node.all_input_nodes))
        )
        written: set[int] = set()
        schema = getattr(node.target, "_schema", None)
        if schema is not None:
            for index, argument in enumerate(operator_alias_contract(schema).arguments):
                if not argument.is_write:
                    continue
                value = (
                    node.args[index]
                    if index < len(node.args)
                    else node.kwargs.get(argument.name)
                )
                leaves, _ = tree_flatten(value)
                for leaf in leaves:
                    if isinstance(leaf, Node):
                        written.update(values[leaf])
        writes[node] = frozenset(written)
    original_output = next(node for node in originals if node.op == "output")
    active: set[Node] = set()
    pending = [original_output, *fences]
    while pending:
        node = pending.pop()
        if node not in active:
            active.add(node)
            pending.extend(node.all_input_nodes)
    reads = {
        node: roots if node in active else frozenset() for node, roots in reads.items()
    }
    return LifetimeGraph(
        storage,
        MappingProxyType(values),
        MappingProxyType(reads),
        MappingProxyType(writes),
        MappingProxyType(fences),
        MappingProxyType(resolved),
        reads[original_output],
        originals,
    )
