"""All CPU preprocessing finishes within admission before any worker consumes."""

import io
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

from engine_tests.fixtures.definition_operations import records
from engine_tests.training.test_input_staging import classifier_staging
from shadowspill.training.inputs.adapters import InputBinding
from shadowspill.training.inputs.frames import FRAME_CHUNK_BYTES, InputFrame
from shadowspill.training.inputs.packing import pack_batch
from shadowspill.training.inputs.spool import InputSpool
from shadowspill.training.inputs.staging import admit_input_staging
from shadowspill.training.inputs.validation import validate_batch
from shadowspill.training.inputs.values import VALIDATION_SCRATCH_BYTES
from shadowspill_client import Execution
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.payload import snapshot_batch
from shadowspill_client.protocol.payload_io import read_payload


class ChunkWriter(io.BytesIO):
    def __init__(self, budget, *, partial=False, fail=False):
        super().__init__()
        self.budget, self.partial, self.fail = budget, partial, fail
        self.largest = self.peak = 0

    def write(self, value):
        assert self.budget.used[1] > 0
        self.peak = max(self.peak, self.budget.used[0])
        self.largest = max(self.largest, len(value))
        assert len(value) <= FRAME_CHUNK_BYTES
        if self.fail:
            return 0
        return super().write(value[:7] if self.partial else value)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice", "reverse"])
@pytest.mark.parametrize("dtype", ["float64", "bfloat16", "bool"])
def test_borrowed_frames_roundtrip_c_order_without_full_tensor_scratch(
    layout, dtype, monkeypatch
):
    nditer = np.nditer

    @contextmanager
    def bounded(*args, **kwargs):
        with nditer(*args, **kwargs) as iterator:

            def chunks():
                for chunk in iterator:
                    assert chunk.flags.c_contiguous
                    assert chunk.nbytes <= FRAME_CHUNK_BYTES
                    yield chunk

            yield chunks()

    monkeypatch.setattr(np, "nditer", bounded)
    width = 384
    array = np.arange(width * width, dtype=np.int64).reshape(width, width)
    array = array.astype("uint16" if dtype == "bfloat16" else dtype)
    if layout == "transpose":
        array = array.T
    elif layout == "slice":
        array = array[:, ::2]
    elif layout == "reverse":
        array = array[::-1, ::-1]
    fields = {
        ("inputs", "nested", "bits"): array,
        ("loss_inputs", "empty"): np.zeros((0, 3), dtype=np.float32),
        ("loss_inputs", "scalar"): np.array(9, dtype=np.int64),
    }
    frame = InputFrame(
        {path: value.shape for path, value in fields.items()},
        {
            ("inputs", "nested", "bits"): dtype,
            ("loss_inputs", "empty"): "float32",
            ("loss_inputs", "scalar"): "int64",
        },
    )
    budget = BufferBudget(frame.write_bytes)
    writer = ChunkWriter(budget, partial=True)
    ref = frame.write(writer, fields, budget, "frame")
    assert writer.peak == frame.write_bytes and budget.used == (0, 0)
    assert len(writer.getvalue()) == frame.size.wire == ref.size_bytes
    assert frame.write_bytes < array.nbytes
    with read_payload(
        io.BytesIO(writer.getvalue()),
        ref,
        BufferBudget(frame.size.staging),
        limits=frame.size,
        datum_count=1,
    ) as decoded:
        assert decoded.header == frame.header
        tensor = decoded.datums[0].inputs["nested"]["bits"]
        assert tensor.dtype == dtype
        np.testing.assert_array_equal(tensor.array(), array)
        assert decoded.datums[0].loss_inputs["empty"].shape == (0, 3)
        assert decoded.datums[0].loss_inputs["scalar"].array().item() == 9


def test_frame_rejects_abi_mismatch_budget_and_short_writes_without_leaking():
    path = ("inputs", "x")
    frame = InputFrame({path: (3,)}, {path: "float32"})
    good = {path: np.ones(3, dtype=np.float32)}
    budget = BufferBudget(frame.write_bytes)
    writer = ChunkWriter(budget)
    for bad in ({}, {path: np.ones(3)}, {path: np.ones(4, dtype=np.float32)}):
        with pytest.raises(ValueError, match="frame layout"):
            frame.write(writer, bad, budget, "bad")
        assert writer.tell() == 0 and budget.used == (0, 0)
    with pytest.raises(ValueError, match="staging bytes"):
        frame.write(writer, good, BufferBudget(frame.write_bytes - 1), "large")
    assert writer.tell() == 0
    with pytest.raises(OSError, match="no progress"):
        frame.write(ChunkWriter(budget, fail=True), good, budget, "failed")
    assert budget.used == (0, 0)
    with pytest.raises(ValueError, match="overlap"):
        InputFrame(
            {path: (), (*path, "nested"): ()}, {path: "bool", (*path, "nested"): "bool"}
        )


def pipeline(*, fill=None):
    schema, batch, placement, schedule, bindings, plan = classifier_staging(metadata=8)
    if fill is not None:
        bindings = {
            key: InputBinding(binding.layout, replace(binding.adapter, fill=fill))
            for key, binding in bindings.items()
        }
        plan = admit_input_staging(
            schema, batch, placement, schedule, bindings, Execution()
        )
    return schema, batch, placement, schedule, bindings, plan


def test_spool_finishes_each_rank_once_and_reuses_ready_frames_with_owned_cleanup(
    tmp_path,
):
    calls = []

    def fill(context, fields):
        calls.append((context.routed.owner.location, context.layout.logical_rank))
        fields["inputs", "sequence_metadata"][:] = len(calls)

    schema, batch, placement, schedule, bindings, plan = pipeline(fill=fill)
    budget = BufferBudget(plan.staging_bytes, max_requests=1)
    with plan.reserve(budget) as reservation:
        arena = reservation.budget
        with (
            snapshot_batch(records(), arena) as owned,
            arena.reserve(plan.workspace_bytes + VALIDATION_SCRATCH_BYTES) as workspace,
        ):
            validated = validate_batch(owned, schema, batch)
            workspace.shrink(plan.workspace_bytes)
            assignment = pack_batch(validated, schema, batch, placement, schedule)
            used = arena.used
            with InputSpool(assignment, bindings, arena, root=tmp_path) as spool:
                assert len(calls) == len(spool.frames) == 4
                assert spool.size_bytes == plan.spool_bytes
                assert arena.used == used
                # Both worker passes use completed files; no fill hook runs again.
                for _ in range(2):
                    for index, frame in enumerate(spool.frames.values(), 1):
                        with frame.read(arena) as decoded:
                            np.testing.assert_array_equal(
                                frame.binding.frame_arrays(decoded)[
                                    "inputs", "sequence_metadata"
                                ],
                                np.full(8, index),
                            )
                            assert arena.used[0] <= plan.staging_bytes
                assert len(calls) == 4 and arena.used == used
                paths = [frame.path for frame in spool.frames.values()]
            assert all(not path.exists() for path in paths)
        assert arena.used == (0, 0)
    assert list(tmp_path.iterdir()) == [] and budget.used == (0, 0)


@pytest.mark.parametrize("failure", ["adapter", "disk"])
def test_later_preprocessing_failure_discards_complete_earlier_frames(
    tmp_path, monkeypatch, failure
):
    seen = []

    def fill(context, fields):
        seen.append(context.layout.logical_rank)
        if failure == "adapter" and len(seen) == 3:
            raise ValueError("invalid later media item")

    original = InputFrame.write

    def write(self, *args):
        if failure == "disk" and len(seen) == 3:
            raise OSError("storage full")
        return original(self, *args)

    monkeypatch.setattr(InputFrame, "write", write)
    schema, batch, placement, schedule, bindings, plan = pipeline(fill=fill)
    with plan.reserve(BufferBudget(plan.staging_bytes)) as reservation:
        arena = reservation.budget
        with (
            snapshot_batch(records(), arena) as owned,
            arena.reserve(plan.workspace_bytes + VALIDATION_SCRATCH_BYTES) as workspace,
        ):
            validated = validate_batch(owned, schema, batch)
            workspace.shrink(plan.workspace_bytes)
            assignment = pack_batch(validated, schema, batch, placement, schedule)
            used = arena.used
            with pytest.raises(
                (ValueError, OSError), match=r"later media|storage full"
            ):
                InputSpool(assignment, bindings, arena, root=tmp_path)
            assert len(seen) == 3 and arena.used == used
            assert list(tmp_path.iterdir()) == []
        assert arena.used == (0, 0)


def test_ready_frame_corruption_is_detected_before_exposing_arrays(tmp_path):
    schema, batch, placement, schedule, bindings, plan = pipeline()
    with plan.reserve(BufferBudget(plan.staging_bytes)) as reservation:
        arena = reservation.budget
        with (
            snapshot_batch(records(), arena) as owned,
            arena.reserve(plan.workspace_bytes + VALIDATION_SCRATCH_BYTES) as workspace,
        ):
            validated = validate_batch(owned, schema, batch)
            workspace.shrink(plan.workspace_bytes)
            assignment = pack_batch(validated, schema, batch, placement, schedule)
            with InputSpool(assignment, bindings, arena, root=tmp_path) as spool:
                frame = next(iter(spool.frames.values()))
                original = frame.path.read_bytes()
                frame.path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                before = arena.used
                with pytest.raises(ProtocolError, match="checksum"):
                    frame.read(arena)
                assert arena.used == before
                frame.path.write_bytes(original[:-1])
                with pytest.raises(ProtocolError, match="declared size"):
                    frame.read(arena)
                assert arena.used == before
