"""Actual packing respects partitions, original limits and scheduled padding."""

import subprocess
import sys

import numpy as np
import pytest

from shadowspill_client import BatchConfig, Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.inputs.packing import PackingError
from shadowspill_client.inputs.selection import PackingCandidate, select_packing
from shadowspill_client.inputs.shapes import BufferShape, PackingSchedule
from shadowspill_client.inputs.validation import validate_batch
from shadowspill_client.placement import GroupRef, InputFieldPlacement, InputPlacement
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.payload import snapshot_batch


def schema():
    path = ("inputs", "tokens")
    return InputSchema(
        [TensorField(path, "int64", ("length",), "pack", 0)],
        [
            Capacity("tokens", "tokens", "sum", path, 0, scheduled=True),
            Capacity("longest", "tokens", "max", path, 0),
            Capacity("records", "records", "count"),
        ],
    )


@pytest.mark.parametrize(
    "length,count,fit", [(4088, 512, True), (4088, 513, False), (2040, 1024, True)]
)
@pytest.mark.parametrize("cooperators", [1, 2])
def test_two_million_token_example_uses_actual_eight_partition_packing(
    length, count, fit, cooperators
):
    contract = schema()
    limits = BatchConfig(limits={"tokens": 2097152, "longest": 4096, "records": 8192})
    placement = InputPlacement(
        GroupRef(
            "inputs",
            tuple(
                tuple(range(i * cooperators, (i + 1) * cooperators)) for i in range(8)
            ),
        ),
        fields=[InputFieldPlacement(("inputs", "tokens"))],
    )
    physical = BufferShape(1024, {("inputs", "tokens"): (262144,)})
    schedule = PackingSchedule("eight", [(physical,)] * 8)
    candidates = [PackingCandidate("prepared", 0, schedule)]
    records = [
        Datum({"tokens": np.zeros(length, dtype=np.int64)}) for _ in range(count)
    ]
    budget = BufferBudget(128 << 20)
    with snapshot_batch(records, budget) as snapshot:
        batch = validate_batch(snapshot, contract, limits)
        assert batch.capacities["tokens"] <= 2097152
        if not fit:
            with pytest.raises(PackingError, match="no packing found"):
                select_packing(batch, [limits], placement, candidates, proofs={})
        else:
            candidate, assignment = select_packing(
                batch, [limits], placement, candidates, proofs={}
            )
            assert candidate.id == "prepared"
            assert sorted(i for row in assignment.records for i in row) == list(
                range(count)
            )
            assert all(len(row) * length <= 262144 for row in assignment.records)
            assert assignment.schedule.capacities(contract) == {"tokens": 2097152}
    assert budget.used == (0, 0)


def test_scheduled_padding_cannot_create_a_third_64k_microbatch_in_128k():
    contract = schema()
    limits = BatchConfig(limits={"tokens": 131072, "longest": 65536, "records": 1024})
    placement = InputPlacement.independent(
        1, fields=[InputFieldPlacement(("inputs", "tokens"))]
    )
    buffer = BufferShape(1024, {("inputs", "tokens"): (65536,)})
    schedule = PackingSchedule("too-many", [(buffer,) * 3])
    with snapshot_batch([Datum({"tokens": [1]})], BufferBudget(1 << 20)) as snapshot:
        batch = validate_batch(snapshot, contract, limits)
        with pytest.raises(ValueError, match="including padding"):
            select_packing(
                batch,
                [limits],
                placement,
                [PackingCandidate("bad", 0, schedule)],
                proofs={},
            )


def test_client_validation_imports_without_engine_or_gpu_frameworks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('torch','triton','mlops','shadowspill','shadowspill_workloads'):
    sys.modules[name] = None
from shadowspill_client import TrainingClient
from shadowspill_client.batch_validation import BatchValidator
from shadowspill_client.inputs.selection import select_packing
assert callable(TrainingClient.validate_batch)
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
