"""Complete-call envelopes bound real ownership and cannot deadlock on themselves."""

import io
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from engine_tests.fixtures.definition_operations import definitions, records
from shadowspill.parallelism import GroupRef, InputFieldPlacement, InputPlacement
from shadowspill.training.inputs.adapters import InputAdapter, InputBinding, InputLayout
from shadowspill.training.inputs.coverage_bounds import CoverageError
from shadowspill.training.inputs.materialize import materialize_buffer
from shadowspill.training.inputs.packing import pack_batch
from shadowspill.training.inputs.routing import route_buffer
from shadowspill.training.inputs.shapes import BufferShape, PackingSchedule
from shadowspill.training.inputs.staging import admit_input_staging, input_envelope
from shadowspill.training.inputs.validation import combined_schema, validate_batch
from shadowspill.training.inputs.values import VALIDATION_SCRATCH_BYTES
from shadowspill_client import BatchConfig, Datum, Execution
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.payload import MEDIA_TYPE, snapshot_batch
from shadowspill_client.protocol.payload_io import read_payload
from shadowspill_client.protocol.requests import ArtifactRef


class ObservedBudget(BufferBudget):
    def __init__(self, maximum):
        super().__init__(maximum, max_requests=4)
        self.peak = 0

    def reserve(self, size, *, timeout=None):
        result = super().reserve(size, timeout=0)
        self.peak = max(self.peak, self.used[0])
        return result


def test_complete_envelope_bounds_exhaustive_raw_dtypes_nested_and_optional_fields():
    schema = InputSchema(
        [
            TensorField(("inputs", 'imä\\"ge', "values"), "float32", ("n", 2)),
            TensorField(("loss_inputs", "unused"), "float32", ("n", 2)),
            TensorField(
                ("loss_inputs", "weight"), "float32", required=False, default=1
            ),
        ],
        [
            Capacity("records", "items", "count"),
            Capacity("tokens", "positions", "sum", ("inputs", 'imä\\"ge', "values"), 0),
        ],
    )
    batch = BatchConfig(limits={"records": 3, "tokens": 4})
    paths = [("inputs", 'imä\\"ge', "values")]
    bound = input_envelope(schema, batch, paths)
    # Aligned optional fields are fully included, even for a selective forward.
    assert bound.snapshot.tensors == (4 * 2 * 2 + 3) * 8
    cases = 0
    for count in range(4):
        for lengths in itertools.product(range(5), repeat=count):
            if sum(lengths) > 4:
                continue
            for optional in (False, True):
                datums = [
                    Datum(
                        {'imä\\"ge': {"values": np.ones((n, 2), dtype=np.float64)}},
                        {"unused": np.zeros((n, 2), dtype=np.int64), "weight": 1.0}
                        if optional
                        else {},
                    )
                    for n in lengths
                ]
                budget = ObservedBudget(bound.snapshot.staging)
                with snapshot_batch(datums, budget) as snapshot:
                    validate_batch(snapshot, schema, batch, required_paths=paths)
                    assert budget.peak <= bound.snapshot.staging
                    assert budget.used[0] <= bound.snapshot.retained
                    assert len(snapshot.header) <= bound.snapshot.header
                    assert snapshot.size_bytes <= bound.snapshot.wire
                    reference = ArtifactRef(
                        "batch", snapshot.sha256, snapshot.size_bytes, MEDIA_TYPE
                    )
                    decoded_budget = ObservedBudget(bound.snapshot.staging)
                    with read_payload(
                        io.BytesIO(b"".join(snapshot.chunks())),
                        reference,
                        decoded_budget,
                    ) as decoded:
                        assert decoded.tensors == snapshot.tensors
                        assert decoded_budget.peak <= bound.snapshot.staging
                assert budget.used == decoded_budget.used == (0, 0)
                cases += 1
    assert cases > 100


def test_omittable_capacity_is_not_a_bound_for_other_supplied_aligned_fields():
    bounded, extra = ("inputs", "bounded"), ("inputs", "extra")
    schema = InputSchema(
        [
            TensorField(bounded, "float32", ("n",)),
            TensorField(extra, "float32", ("n",)),
        ],
        [
            Capacity("records", "items", "count"),
            Capacity("n", "positions", "sum", bounded, 0),
        ],
    )
    batch = BatchConfig(limits={"records": 1, "n": 1})
    # This input is legal for an operation that only consumes extra: the omitted
    # bounded field contributes nothing to capacity validation.
    with snapshot_batch(
        [Datum({"extra": np.zeros(32)})], BufferBudget(1 << 20)
    ) as owned:
        assert (
            validate_batch(owned, schema, batch, required_paths=[extra]).capacities["n"]
            == 0
        )
    with pytest.raises(CoverageError, match="unbounded dimension"):
        input_envelope(schema, batch, [extra])
    # Even when neither is selected, every known supplied leaf must be bounded.
    with pytest.raises(CoverageError, match="unbounded dimension"):
        input_envelope(schema, batch, [])
    assert input_envelope(schema, batch, [bounded]).snapshot.tensors == 16


def test_unused_fixed_sum_does_not_bound_record_count_and_limits_reject_without_arrays(
    monkeypatch,
):
    path = ("inputs", "optional")
    schema = InputSchema(
        [TensorField(path, "float32", (2,))],
        [Capacity("size", "elements", "sum", path, 0)],
    )
    with pytest.raises(CoverageError, match="Datum count is unbounded"):
        input_envelope(schema, BatchConfig(limits={"size": 2}), [])
    assert (
        input_envelope(schema, BatchConfig(limits={"size": 2}), [path]).max_datums == 1
    )

    def forbidden(*args, **kwargs):
        pytest.fail("allocated an array while admitting metadata")

    monkeypatch.setattr(np, "empty", forbidden)
    monkeypatch.setattr(np, "zeros", forbidden)
    zero = InputSchema(
        [TensorField(("inputs", "zero"), "float32", (0,))],
        [Capacity("records", "items", "count")],
    )
    with pytest.raises(CoverageError, match="metadata cannot fit"):
        input_envelope(
            zero, BatchConfig(limits={"records": 10**20}), [("inputs", "zero")]
        )
    for field in (
        TensorField(("inputs", *["nested"] * 61), "float32"),
        TensorField(("inputs", "high_rank"), "float32", (0,) * 33),
    ):
        with pytest.raises(CoverageError, match="depth/rank"):
            input_envelope(
                InputSchema([field], zero.capacities),
                BatchConfig(limits={"records": 1}),
                [tuple(field.path)],
            )
    empty = input_envelope(
        zero, BatchConfig(limits={"records": 0}), [("inputs", "zero")]
    )
    assert empty.snapshot.tensors == empty.snapshot.largest_tensor == 0


def classifier_staging(*, metadata=0):
    model, objective, _ = definitions()
    schema = combined_schema(model.inputs, objective.inputs)
    batch = BatchConfig(limits={"max_examples": 4})
    placement = InputPlacement(
        GroupRef("cooperating", ((0, 1),)),
        fields=[InputFieldPlacement(tuple(f.path)) for f in schema.fields],
    )
    shape = BufferShape(2, {tuple(f.path): (2, *f.shape) for f in schema.fields})
    schedule = PackingSchedule("classifier", [[shape, shape]])
    bindings = {}
    for index in range(2):
        for rank in range(2):
            layout = InputLayout(schema, shape, placement, rank)
            if metadata:
                binding = InputBinding(
                    layout,
                    InputAdapter(
                        metadata=[
                            TensorField(
                                ("inputs", "sequence_metadata"), "int64", (metadata,)
                            )
                        ],
                        padding_safe=True,
                    ),
                )
            else:
                binding = model.bind_inputs(layout)
            bindings[index, rank] = binding
    plan = admit_input_staging(
        schema, batch, placement, schedule, bindings, Execution()
    )
    return schema, batch, placement, schedule, bindings, plan


def test_real_pipeline_stays_within_one_reservation_while_another_request_waits():
    schema, batch, placement, schedule, bindings, plan = classifier_staging(
        metadata=1000
    )
    budget = BufferBudget(plan.staging_bytes, max_requests=2)
    with plan.reserve(budget) as reservation:
        assert budget.used == (plan.staging_bytes, 1)
        started = Event()

        def pending():
            started.set()
            return plan.reserve(budget, timeout=2)

        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(pending)
            assert started.wait(1) and not future.done()
            # Wider float64 caller storage is allowed before conversion to float32.
            data = records()
            data[0] = replace(
                data[0], inputs={"pixels": data[0].inputs["pixels"].astype(np.float64)}
            )
            with (
                snapshot_batch(data, reservation.budget) as owned,
                reservation.budget.reserve(
                    plan.workspace_bytes + VALIDATION_SCRATCH_BYTES
                ) as workspace,
            ):
                validated = validate_batch(owned, schema, batch)
                workspace.shrink(plan.workspace_bytes)
                assignment = pack_batch(validated, schema, batch, placement, schedule)
                seen = []
                for index in range(2):
                    with materialize_buffer(
                        assignment, schema, index, reservation.budget
                    ) as packed:
                        for rank in range(2):
                            with bindings[index, rank].adapt(
                                route_buffer(packed, placement, rank),
                                reservation.budget,
                            ) as adapted:
                                assert adapted.fields[
                                    "inputs", "sequence_metadata"
                                ].shape == (1000,)
                                assert reservation.budget.used[0] <= plan.staging_bytes
                                if rank == 0:
                                    seen.extend(
                                        region.datum for region in packed.regions
                                    )
                assert sorted(seen) == list(range(len(data)))
            assert reservation.budget.used == (0, 0) and not future.done()
            reservation.close()
            with future.result(timeout=2):
                assert budget.used == (plan.staging_bytes, 1)
    assert budget.used == (0, 0)


def test_admission_checks_every_adapter_metadata_and_ownership_cleanup():
    schema, batch, placement, schedule, bindings, plan = classifier_staging(
        metadata=100000
    )
    smaller = classifier_staging()[-1]
    assert plan.staging_bytes > smaller.staging_bytes
    with pytest.raises(CoverageError, match="configured max_staging_bytes"):
        admit_input_staging(
            schema,
            batch,
            placement,
            schedule,
            bindings,
            Execution(max_staging_bytes=plan.staging_bytes - 1),
        )
    for bad in (
        {**bindings, (2, 0): bindings[0, 0]},
        {key: value for key, value in bindings.items() if key != (1, 1)},
    ):
        with pytest.raises(ValueError, match="exactly every"):
            admit_input_staging(schema, batch, placement, schedule, bad, Execution())
    with pytest.raises(ValueError, match="prepared ABI"):
        admit_input_staging(
            schema,
            batch,
            placement,
            schedule,
            {**bindings, (0, 1): bindings[0, 0]},
            Execution(),
        )
    parent = BufferBudget(plan.staging_bytes, max_requests=1)
    reservation = plan.reserve(parent)
    lease = reservation.budget.reserve(plan.staging_bytes)
    with pytest.raises(TimeoutError):
        reservation.budget.reserve(1)  # Never wait on the same call's own bytes.
    with pytest.raises(RuntimeError, match="live input owners"):
        reservation.close()
    assert parent.used == (plan.staging_bytes, 1)
    lease.close()
    reservation.close()
    assert parent.used == (0, 0)


def test_matrix_elements_use_aggregate_axes_and_bounded_inspection_imports():
    schema = InputSchema(
        [TensorField(("inputs", "matrix"), "float32", ("rows", "columns"))],
        [
            Capacity("records", "items", "count"),
            Capacity("rows", "rows", "sum", ("inputs", "matrix"), 0),
            Capacity("columns", "columns", "max", ("inputs", "matrix"), 1),
        ],
    )
    bound = input_envelope(
        schema,
        BatchConfig(limits={"records": 4, "rows": 5, "columns": 7}),
        [("inputs", "matrix")],
    )
    assert bound.snapshot.tensors == 5 * 7 * 8
    code = """
import sys
for name in ('torch', 'triton', 'shadowspill_workloads', 'mlops'):
    sys.modules[name] = None
from engine_tests.training.test_input_staging import classifier_staging
assert classifier_staging()[-1].staging_bytes > 0
"""
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr
