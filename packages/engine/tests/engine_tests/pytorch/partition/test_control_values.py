"""Control-value derivation is bounded by the live dependency set, not depth."""

import weakref

import torch
from torch.fx import Graph, GraphModule

from shadowspill.pytorch.partition import values
from shadowspill.pytorch.partition.artifacts import StageRecord, StageValueSource
from shadowspill.pytorch.partition.split import SplitExportGraph


def test_long_control_dependency_chain_releases_snapshots(monkeypatch):
    graph = Graph()
    weight, x = graph.placeholder("weight"), graph.placeholder("x")
    graph.output((graph.call_function(torch.ops.aten.add.Tensor, (weight, x)),))
    adding = GraphModule({}, graph)
    float_value = torch.ones(1)
    stages = []
    count = 1100
    for index in range(count):
        previous = (
            StageValueSource(root_input_index=1)
            if index == 0
            else StageValueSource(
                producer_stage_index=index - 1, producer_output_index=0
            )
        )
        stages.append(
            StageRecord(
                f"stage_{index}",
                adding,
                (float_value, float_value),
                (StageValueSource(root_input_index=0), previous),
                (float_value,),
            )
        )
    graph = Graph()
    x = graph.placeholder("x")
    graph.output(
        (
            graph.call_function(
                torch.ops.aten._to_copy.default, (x,), {"dtype": torch.int64}
            ),
        )
    )
    integer = torch.ones(1, dtype=torch.int64)
    stages.append(
        StageRecord(
            "control",
            GraphModule({}, graph),
            (float_value,),
            (
                StageValueSource(
                    producer_stage_index=count - 1, producer_output_index=0
                ),
            ),
            (integer,),
        )
    )
    graph = Graph()
    graph.output((graph.placeholder("control"),))
    stages.append(
        StageRecord(
            "consumer",
            GraphModule({}, graph),
            (integer,),
            (StageValueSource(producer_stage_index=count, producer_output_index=0),),
            (integer,),
        )
    )
    split = SplitExportGraph(stages[-1].graph_module, tuple(stages), {}, {})
    snapshot = values._snapshot
    live, peak = [], 0

    def observe(value):
        nonlocal peak
        result = snapshot(value)
        live[:] = [ref for ref in live if ref() is not None]
        live.append(weakref.ref(result))
        peak = max(peak, len(live))
        return result

    monkeypatch.setattr(values, "_snapshot", observe)
    result = values.derive_authentic_control_values(split, (float_value, float_value))
    assert set(result) == {(count, 0)}
    torch.testing.assert_close(result[(count, 0)], torch.tensor([count + 1]))
    # One previous value and its successor; no full-chain host retention.
    assert peak <= 2


def _unused_branch(x):
    raise AssertionError("An unrelated branch was executed")


def test_branched_controls_share_a_producer_and_skip_unneeded_work():
    graph = Graph()
    x = graph.placeholder("x")
    left = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
    right = graph.call_function(torch.ops.aten.sub.Tensor, (x, 1))
    unused = graph.call_function(_unused_branch, (x,))
    graph.output((left, right, unused))
    producer = GraphModule({}, graph)
    float_value = torch.tensor([2.0])
    integer = torch.tensor([2], dtype=torch.int64)

    graph = Graph()
    x = graph.placeholder("x")
    graph.output(
        (
            graph.call_function(
                torch.ops.aten._to_copy.default, (x,), {"dtype": torch.int64}
            ),
        )
    )
    cast = GraphModule({}, graph)
    graph = Graph()
    x, y = graph.placeholder("x"), graph.placeholder("y")
    graph.output((graph.call_function(torch.ops.aten.add.Tensor, (x, y)),))
    add = GraphModule({}, graph)
    graph = Graph()
    graph.output((graph.placeholder("a"), graph.placeholder("b")))
    consumer = GraphModule({}, graph)

    def source(stage, output=0):
        return StageValueSource(
            producer_stage_index=stage, producer_output_index=output
        )

    stages = (
        StageRecord(
            "producer",
            producer,
            (float_value,),
            (StageValueSource(root_input_index=0),),
            (float_value,) * 3,
        ),
        StageRecord("left_control", cast, (float_value,), (source(0),), (integer,)),
        StageRecord(
            "join",
            add,
            (float_value, float_value),
            (source(0), source(0, 1)),
            (float_value,),
        ),
        StageRecord("joined_control", cast, (float_value,), (source(2),), (integer,)),
        StageRecord(
            "consumer",
            consumer,
            (integer, integer),
            (source(1), source(3)),
            (integer, integer),
        ),
    )
    split = SplitExportGraph(consumer, stages, {}, {})
    result = values.derive_authentic_control_values(split, (float_value,))
    torch.testing.assert_close(result[(1, 0)], torch.tensor([3]))
    torch.testing.assert_close(result[(3, 0)], torch.tensor([4]))
