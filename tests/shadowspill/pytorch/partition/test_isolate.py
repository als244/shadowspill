"""Sequence barriers preserve dataflow and compose with user stage boundaries."""

import operator

import torch
from torch import nn
from torch.fx import Graph, GraphModule

from shadowspill.pytorch.partition import IsolateOperations
from shadowspill.pytorch.partition.policy import resolve_partition_assignments


def test_operator_and_delayed_tuple_users_form_one_contiguous_barrier():
    graph = Graph()
    value = graph.placeholder("value")
    before = graph.call_function(torch.ops.aten.sin.default, (value,))
    pair = graph.call_function(torch.ops.aten.sort.default, (before,))
    independent = graph.call_function(torch.ops.aten.cos.default, (value,))
    ordered = graph.call_function(operator.getitem, (pair, 0))
    indices = graph.call_function(operator.getitem, (pair, 1))
    after = graph.call_function(torch.ops.aten.add.Tensor, (ordered, independent))
    graph.output((after, indices))
    module = GraphModule(nn.Module(), graph)
    original = str(graph)
    assignments, _ = resolve_partition_assignments(
        module, module, IsolateOperations(("aten.sort.default",), base="whole")
    )
    assert assignments[before] != assignments[pair]
    assert (
        len({assignments[node] for node in (pair, independent, ordered, indices)}) == 1
    )
    assert assignments[after] != assignments[pair]
    assert str(graph) == original


def test_overlapping_tuple_barriers_merge_without_reordering_the_graph():
    graph = Graph()
    value = graph.placeholder("value")
    first = graph.call_function(torch.ops.aten.sort.default, (value,))
    second = graph.call_function(torch.ops.aten.max.dim, (value, 0))
    left = graph.call_function(operator.getitem, (first, 0))
    right = graph.call_function(operator.getitem, (second, 0))
    graph.output((left, right))
    module = GraphModule(nn.Module(), graph)
    assignments, _ = resolve_partition_assignments(
        module,
        module,
        IsolateOperations(("aten.sort.default", "aten.max.dim"), base="whole"),
    )
    assert len(set(assignments.values())) == 1
    x = torch.randn(5, 3)
    torch.testing.assert_close(module(x), (x.sort().values, x.max(0).values))


def test_isolation_keeps_custom_base_boundaries_around_each_occurrence():
    graph = Graph()
    value = graph.placeholder("value")
    nodes = []
    for operation in (
        torch.ops.aten.sin.default,
        torch.ops.aten.cos.default,
        torch.ops.aten.sin.default,
        torch.ops.aten.cos.default,
    ):
        value = graph.call_function(operation, (value,))
        nodes.append(value)
    graph.output(value)
    module = GraphModule(nn.Module(), graph)

    class Base:
        def assign_stages(self, graph_module, source):
            return {node.name: i // 2 for i, node in enumerate(nodes)}

    assignments, _ = resolve_partition_assignments(
        module, module, IsolateOperations(("aten.cos.default",), base=Base())
    )
    assert len(set(assignments.values())) == 4
