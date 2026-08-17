from __future__ import annotations

import io

import pytest
import torch
from torch.fx import Graph, GraphModule

from shadowspill.pytorch.compilation.fx_graph import SerializedFxGraph
from shadowspill.pytorch.contracts import CaptureError


def _factory_output_graph() -> GraphModule:
    graph = Graph()
    value = graph.placeholder("value")
    index = graph.call_function(
        torch.ops.aten.arange.default,
        args=(4,),
        kwargs={"device": torch.device("cpu")},
    )
    doubled = graph.call_function(torch.ops.aten.add.Tensor, args=(value, value))
    graph.output((doubled, index))
    return GraphModule({}, graph)


def test_serialized_fx_graph_preserves_input_independent_tensor_nodes() -> None:
    original = _factory_output_graph()
    record = SerializedFxGraph.capture(original)
    storage = io.BytesIO()
    torch.save(record, storage)
    storage.seek(0)
    loaded = torch.load(storage, weights_only=False)
    assert isinstance(loaded, SerializedFxGraph)

    restored = loaded.restore()
    assert tuple(node.op for node in restored.graph.nodes) == tuple(
        node.op for node in original.graph.nodes
    )
    assert tuple(str(node.target) for node in restored.graph.nodes) == tuple(
        str(node.target) for node in original.graph.nodes
    )
    assert not any(node.op == "get_attr" for node in restored.graph.nodes)
    expected = original(torch.arange(4, dtype=torch.float32))
    actual = restored(torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(actual, expected)


def test_serialized_fx_graph_rejects_tensor_literals() -> None:
    graph = Graph()
    value = graph.placeholder("value")
    literal = torch.ones(4)
    result = graph.create_node(
        "call_function",
        torch.ops.aten.add.Tensor,
        (value, literal),
        {},
    )
    graph.output(result)
    module = GraphModule({}, graph)

    with pytest.raises(CaptureError, match="literal Tensor"):
        SerializedFxGraph.capture(module)
