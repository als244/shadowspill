"""Output coordinates restore caller order through padding and worker placement."""

import json
from contextlib import ExitStack

import numpy as np
import pytest

from engine_tests.training.test_packing import batch, limits, schema, shape
from shadowspill.parallelism import GroupRef, InputPlacement, OutputPlacement
from shadowspill.training.inputs.materialize import materialize_buffer
from shadowspill.training.inputs.packing import pack_batch
from shadowspill.training.inputs.shapes import BufferShape, PackingSchedule
from shadowspill.training.inputs.validation import validate_batch
from shadowspill.training.outputs import OutputLayout
from shadowspill_client import BatchConfig, Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.definitions import OutputSet, TensorOutput
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.payload import snapshot_batch


def assemble(layout, assignment, workers):
    """Small host-array consumer; production artifacts will stream these regions."""
    buffers = {}
    for piece in layout.pieces(assignment):
        dest = piece.destination
        source = workers[piece.buffer, piece.logical_rank][dest.name]
        assert source.shape == piece.source_shape
        key = dest.name, piece.buffer, dest.datum
        target = buffers.setdefault(key, np.zeros(dest.shape, dtype=source.dtype))
        if piece.worker_reduction == "sum":
            target[piece.destination_axes] += source[piece.source_axes]
        else:
            target[piece.destination_axes] = source[piece.source_axes]
    result = {}
    for (name, _buffer, datum), value in buffers.items():
        key = name, datum
        if key not in result:
            result[key] = value.copy()
        else:
            reduction = layout.global_reductions[name]
            if reduction == "sum":
                result[key] += value
            elif reduction == "min":
                np.minimum(result[key], value, out=result[key])
            elif reduction == "max":
                np.maximum(result[key], value, out=result[key])
            else:
                np.testing.assert_array_equal(result[key], value)
    return result


@pytest.mark.parametrize("mode", ["leader", "sum", "concat"])
def test_aligned_outputs_change_axis_preserve_datums_and_reduce_cooperating_copies(
    mode,
):
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
                split = full.shape[1] // 2
                left, right = full[:, :split], full[:, split:]
            elif mode == "sum":
                left, right = full * 0.25, full * 0.75
            else:
                left, right = full, full + 1000
            workers[index, 1] = {"prediction": left}
            workers[index, 0] = {"prediction": right}
        actual = assemble(layout, packed, workers)
        for index, values in enumerate(data.fields):
            original = values[("inputs", "x")].source.array()
            np.testing.assert_array_equal(
                actual["prediction", index], np.stack([original, original + 1])
            )
        assert [d.datum for d in layout.destinations(packed)] == [0, 1, 2]
        assert layout.metric_workers(0) == (1,)
        restored = OutputLayout.from_dict(json.loads(json.dumps(layout.to_dict())))
        assert restored.to_dict() == layout.to_dict()


def test_stacked_variable_axes_scalar_datums_and_worker_slicing_record_axis():
    source = ("inputs", "pixels")
    definition = InputSchema(
        [TensorField(source, "float32", ("height", 2))],
        [
            Capacity("records", "records", "count"),
            Capacity("height", "pixels", "max", source, 0),
        ],
    )
    bound = BatchConfig(limits={"records": 3, "height": 4})
    schedule = PackingSchedule("stacked", [[BufferShape(3, {source: (3, 4, 2)})]])
    outputs = OutputSet(
        tensors={
            "pixels": TensorOutput("float32", ("height", 2), "aligned", "height"),
            "rows": TensorOutput("float32", ("height",), "datum"),
            "summary": TensorOutput("float32", (), "datum"),
        }
    )
    placement = InputPlacement(
        GroupRef("pair", ((0, 1),)),
        outputs={name: OutputPlacement("concat", 0) for name in outputs.tensors},
    )
    layout = OutputLayout(definition, outputs, schedule, placement)
    datums = [
        Datum({"pixels": np.arange(n * 2, dtype=np.float32).reshape(n, 2) + index})
        for index, n in enumerate((1, 3, 0))
    ]
    with snapshot_batch(datums, BufferBudget(1 << 20)) as snapshot:
        data = validate_batch(snapshot, definition, bound)
        packed = pack_batch(data, definition, bound, placement, schedule)
        with materialize_buffer(packed, definition, 0, BufferBudget(2 << 20)) as buffer:
            pixels = buffer.fields[source]
            full = {
                "pixels": pixels,
                "rows": pixels.sum(axis=2),
                "summary": pixels.sum(axis=(1, 2)),
            }
            workers = {
                (0, rank): {name: value[start:stop] for name, value in full.items()}
                for rank, start, stop in ((0, 0, 1), (1, 1, 3))
            }
            actual = assemble(layout, packed, workers)
        for index, datum in enumerate(datums):
            original = datum.inputs["pixels"]
            np.testing.assert_array_equal(actual["pixels", index], original)
            np.testing.assert_array_equal(actual["rows", index], original.sum(axis=1))
            np.testing.assert_array_equal(actual["summary", index], original.sum())


def test_global_worker_sum_precedes_separate_microbatch_max():
    bound = limits(4, 2, records=2)
    schedule = PackingSchedule("two", [[shape(2), shape(2)]])
    placement = InputPlacement(
        GroupRef("pair", ((1, 0),)),
        loss="partitioned",
        outputs={"global": OutputPlacement("sum")},
    )
    outputs = OutputSet(
        tensors={"global": TensorOutput("float32", (), "global")},
        metrics={"valid": "sum"},
    )
    layout = OutputLayout(
        schema(), outputs, schedule, placement, global_reductions={"global": "max"}
    )
    with batch([1, 2], bound) as data:
        packed = pack_batch(data, schema(), bound, placement, schedule)
        workers = {
            (index, rank): {"global": np.asarray(value, dtype=np.float32)}
            for index, rank, value in ((0, 1, 10), (0, 0, 40), (1, 1, 30), (1, 0, 30))
        }
        actual = assemble(layout, packed, workers)
        assert actual["global", None] == 60  # Not max of individual worker values.
        assert layout.metric_workers(0) == (1, 0)
    with pytest.raises(ValueError, match="reduction for every global"):
        OutputLayout(schema(), outputs, schedule, placement)


def test_empty_worker_slices_and_empty_aligned_outputs_retain_declared_shapes():
    bound = limits(1, 1, records=2)
    schedule = PackingSchedule("one", [[shape(1)]])
    placement = InputPlacement(
        GroupRef("four", ((0, 1, 2, 3),)), outputs={"x": OutputPlacement("concat", 0)}
    )
    output = OutputSet(
        tensors={"x": TensorOutput("float32", ("positions",), "aligned", "positions")}
    )
    layout = OutputLayout(schema(), output, schedule, placement)
    with batch([0, 1], bound) as data:
        packed = pack_batch(data, schema(), bound, placement, schedule)
        workers = {
            (0, rank): {"x": np.asarray([] if rank < 3 else [10], dtype=np.float32)}
            for rank in range(4)
        }
        actual = assemble(layout, packed, workers)
        assert actual["x", 0].shape == (0,)
        np.testing.assert_array_equal(actual["x", 1], [10])
        assert [d.shape for d in layout.destinations(packed)] == [(0,), (1,)]


def test_alignment_ambiguity_and_missing_cooperation_rules_fail_before_execution():
    definition = InputSchema(
        [
            TensorField(("inputs", "packed"), "float32", ("n",), "pack", 0),
            TensorField(("inputs", "stacked"), "float32", ("n",)),
        ]
    )
    schedule = PackingSchedule(
        "mixed",
        [[BufferShape(2, {("inputs", "packed"): (8,), ("inputs", "stacked"): (2, 4)})]],
    )
    outputs = OutputSet(
        tensors={"result": TensorOutput("float32", ("n",), "aligned", "n")}
    )
    with pytest.raises(ValueError, match="unambiguous"):
        OutputLayout(definition, outputs, schedule, InputPlacement.independent(1))
    layout = OutputLayout(
        definition,
        outputs,
        schedule,
        InputPlacement.independent(1),
        sources={"result": ("inputs", "packed")},
    )
    assert layout.physical_shape("result", 0) == (8,)
    with pytest.raises(ValueError, match="incompatible alignment"):
        OutputLayout(
            definition,
            outputs,
            schedule,
            InputPlacement.independent(1),
            sources={"result": ("inputs", "missing")},
        )
    with pytest.raises(ValueError, match="cooperating-worker placement"):
        OutputLayout(
            definition,
            outputs,
            schedule,
            InputPlacement(GroupRef("two", ((0, 1),))),
            sources={"result": ("inputs", "packed")},
        )
    with pytest.raises(ValueError, match="does not own"):
        layout.worker_shape("result", 0, 3)
