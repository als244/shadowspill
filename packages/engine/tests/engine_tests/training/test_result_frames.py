"""Prepared output mappings drive bounded files, preserving caller order."""

import io
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pytest

from engine_tests.training.test_outputs import assemble
from engine_tests.training.test_packing import batch, limits, schema, shape
from shadowspill.parallelism import GroupRef, InputPlacement, OutputPlacement
from shadowspill.training.inputs.frames import InputFrame
from shadowspill.training.inputs.materialize import materialize_buffer
from shadowspill.training.outputs import OutputLayout
from shadowspill.training.result_frames import ResultFrames
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.inputs.packing import pack_batch
from shadowspill_client.inputs.shapes import PackingSchedule
from shadowspill_client.protocol.components import OutputSet, TensorOutput
from shadowspill_client.protocol.payload_io import read_payload


def worker_frames(values):
    frames, sources = {}, {}
    for rank in sorted({rank for _, rank in values}):
        buffers = tuple(sorted(buffer for buffer, owner in values if owner == rank))
        fields = {
            ("inputs", str(index), "values", name): value
            for index, buffer in enumerate(buffers)
            for name, value in values[buffer, rank].items()
        }
        frame = InputFrame(
            {path: value.shape for path, value in fields.items()},
            {path: value.dtype.name for path, value in fields.items()},
        )
        stream = io.BytesIO()
        frame.write(stream, fields, BufferBudget(4 << 20), "worker")
        sources[rank] = stream.getvalue()
        frames[rank] = SimpleNamespace(frame=frame, buffers=buffers)
    return frames, lambda rank: io.BytesIO(sources[rank])


def check(tmp_path, layout, assignment, workers):
    expected = assemble(layout, assignment, workers)
    frames, open_rank = worker_frames(workers)
    memory, disk = BufferBudget(16 << 20, 10), BufferBudget(16 << 20, 10)
    result = ResultFrames(layout, assignment, root=tmp_path, memory=memory, disk=disk)
    try:
        result.start()
        result.assemble(frames, open_rank)
        assert result.assignment is None
        for name, frame in result.frames.items():
            wire = io.BytesIO()
            reference = result.write(name, wire, "result")
            wire.seek(0)
            with read_payload(wire, reference, memory, limits=frame.size) as decoded:
                values = decoded.datums[0].inputs
                for (output, datum), value in expected.items():
                    if name == output:
                        key = "global" if datum is None else str(datum)
                        np.testing.assert_array_equal(values[key].array(), value)
                assert len(values) == sum(output == name for output, _ in expected)
    finally:
        result.close()
    assert memory.used == disk.used == (0, 0)


@pytest.mark.parametrize("mode", ["leader", "sum", "concat"])
def test_streamed_outputs_restore_reordered_packed_datums(tmp_path, mode):
    bound = limits(5, 3, records=3)
    schedule = PackingSchedule("unequal", [[shape(2), shape(3)]])
    placement = InputPlacement(
        GroupRef("pair", ((1, 0),)),
        outputs={"prediction": OutputPlacement(mode, 1 if mode == "concat" else None)},
    )
    outputs = OutputSet(
        tensors={
            "prediction": TensorOutput(
                "float32", (2, "positions"), "aligned", "positions"
            )
        }
    )
    layout = OutputLayout(schema(), outputs, schedule, placement)
    with batch([1, 3, 1], bound) as data, ExitStack() as stack:
        packed = pack_batch(data, schema(), bound, placement, schedule)
        assert packed.records == ((0, 2), (1,))
        workers = {}
        for index in range(2):
            buffer = stack.enter_context(
                materialize_buffer(packed, schema(), index, BufferBudget(2 << 20))
            )
            full = np.stack(
                [buffer.fields[("inputs", "x")], buffer.fields[("inputs", "x")] + 1]
            )
            if mode == "concat":
                left, right = np.split(full, [full.shape[1] // 2], axis=1)
            elif mode == "sum":
                left, right = full * 0.25, full * 0.75
            else:
                left, right = full, full + 1000
            workers[index, 1] = {"prediction": left}
            workers[index, 0] = {"prediction": right}
        check(tmp_path, layout, packed, workers)


@pytest.mark.parametrize("reduction", ["sum", "min", "max", "identical"])
def test_global_reduction_follows_complete_worker_sum(tmp_path, reduction):
    bound = limits(4, 2, records=2)
    schedule = PackingSchedule("two", [[shape(2), shape(2)]])
    placement = InputPlacement(
        GroupRef("pair", ((1, 0),)),
        loss="partitioned",
        outputs={"global": OutputPlacement("sum")},
    )
    layout = OutputLayout(
        schema(),
        OutputSet(
            tensors={
                "global": TensorOutput("float32", (), "global", reduction=reduction)
            }
        ),
        schedule,
        placement,
    )
    with batch([1, 2], bound) as data:
        packed = pack_batch(data, schema(), bound, placement, schedule)
        workers = {
            (buffer, rank): {"global": np.asarray(value, np.float32)}
            for buffer, rank, value in (
                (0, 1, 10),
                (0, 0, 40),
                (1, 1, 30),
                (1, 0, 20 if reduction == "identical" else 30),
            )
        }
        check(tmp_path, layout, packed, workers)


def test_empty_aligned_outputs_and_zero_length_worker_pieces_keep_shapes(tmp_path):
    bound = limits(1, 1, records=2)
    schedule = PackingSchedule("one", [[shape(1)]])
    placement = InputPlacement(
        GroupRef("four", ((0, 1, 2, 3),)), outputs={"x": OutputPlacement("concat", 0)}
    )
    layout = OutputLayout(
        schema(),
        OutputSet(
            tensors={
                "x": TensorOutput("float32", ("positions",), "aligned", "positions")
            }
        ),
        schedule,
        placement,
    )
    with batch([0, 1], bound) as data:
        packed = pack_batch(data, schema(), bound, placement, schedule)
        workers = {
            (0, rank): {"x": np.asarray([] if rank < 3 else [10], np.float32)}
            for rank in range(4)
        }
        check(tmp_path, layout, packed, workers)


def test_output_scratch_capacity_failure_leaves_a_reachable_metadata_owner(tmp_path):
    bound = limits(1, 1, records=1)
    schedule = PackingSchedule("one", [[shape(1)]])
    placement = InputPlacement.independent(1)
    layout = OutputLayout(
        schema(),
        OutputSet(
            tensors={
                "x": TensorOutput("float32", ("positions",), "aligned", "positions")
            }
        ),
        schedule,
        placement,
    )
    with batch([1], bound) as data:
        packed = pack_batch(data, schema(), bound, placement, schedule)
        memory, disk = BufferBudget(4 << 20, 10), BufferBudget(1)
        result = ResultFrames(layout, packed, root=tmp_path, memory=memory, disk=disk)
        with pytest.raises(ValueError, match="staging bytes"):
            result.start()
        assert memory.used[0] > 0 and disk.used == (0, 0)
        result.close()
        assert memory.used == (0, 0)
