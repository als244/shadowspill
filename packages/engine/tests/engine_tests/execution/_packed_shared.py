"""Owned global records through the real saved complete-step execution path."""

from contextlib import ExitStack

import torch

from shadowspill.parallelism import InputFieldPlacement, InputPlacement
from shadowspill.training.inputs.materialize import materialize_buffer
from shadowspill.training.inputs.packing import pack_batch
from shadowspill.training.inputs.routing import route_buffer
from shadowspill.training.inputs.shapes import BufferShape, PackingSchedule
from shadowspill.training.inputs.validation import validate_batch
from shadowspill.training.observations import to_host
from shadowspill_client import BatchConfig, Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.payload import snapshot_batch


def qualify_packed_family(executor, oracle, snapshot, fill):
    executor.initialize(fill, chunk_bytes=32)
    paths = (("inputs", "x"), ("loss_inputs", "target"), ("loss_inputs", "mask"))
    schema = InputSchema(
        [TensorField(path, "float32", ("positions", 4), "pack", 0) for path in paths],
        [
            Capacity("positions", "positions", "sum", paths[0], 0, scheduled=True),
            Capacity("length", "positions", "max", paths[0], 0),
            Capacity("records", "records", "count"),
        ],
    )
    limits = BatchConfig(limits={"positions": 5, "length": 3, "records": 3})
    placement = InputPlacement.independent(
        1, fields=[InputFieldPlacement(path, "replicate") for path in paths]
    )
    schedule = PackingSchedule(
        "complete", [[BufferShape(3, {path: (n, 4) for path in paths}) for n in (2, 3)]]
    )
    columns = [
        torch.cat([args[i] for args in oracle["examples"]]).numpy() for i in range(3)
    ]
    records = [
        Datum(
            {"x": columns[0][start:stop]},
            {"target": columns[1][start:stop], "mask": columns[2][start:stop]},
        )
        for start, stop in ((0, 1), (1, 4), (4, 5))
    ]
    with snapshot_batch(records, BufferBudget(2 << 20)) as owned, ExitStack() as stack:
        # All original storage is reusable immediately after snapshot completion.
        for column in columns:
            column.fill(-100)
        validated = validate_batch(owned, schema, limits)
        packed = pack_batch(validated, schema, limits, placement, schedule)
        assert packed.records == ((0, 2), (1,))
        budget = BufferBudget(2 << 20)
        buffers = [
            stack.enter_context(materialize_buffer(packed, schema, i, budget))
            for i in range(2)
        ]
        calls = []
        for buffer in buffers:
            routed = route_buffer(buffer, placement, 0)
            assert routed.contributes_loss
            # Torch is already installed in this execution worker. The arrays
            # are read-only routing views; use the owned writable backing for
            # its CPU tensor views, which Executor stages without changing it.
            calls.append(tuple(torch.from_numpy(buffer.fields[path]) for path in paths))
        scale_name = executor.plans["complete"].program.bindings[
            "gradient_scale_control"
        ]
        result = executor.run(
            plan="complete", args=calls, hyperparams={scale_name: oracle["split_scale"]}
        ).result()
        predictions = to_host(tuple(item["prediction"] for item in result.metrics))
        executor.synchronize()
        del result
        restored = [None] * len(records)
        for buffer, output in zip(buffers, predictions, strict=True):
            for region in buffer.regions:
                restored[region.datum] = output[region.fields[paths[0]]].clone()
        for index, output in enumerate(restored):
            original = torch.from_numpy(owned.datums[index].inputs["x"].array().copy())
            expected = (
                original @ oracle["initial"]["weight"] + oracle["initial"]["bias"]
            )
            torch.testing.assert_close(output, expected)
        actual = snapshot(executor)
        for name, value in oracle["split"].items():
            torch.testing.assert_close(actual[name].reshape(value.shape), value)
    assert budget.used == (0, 0)
