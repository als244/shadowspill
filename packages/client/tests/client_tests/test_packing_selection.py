"""Actual packing respects partitions, original limits and scheduled padding."""

import subprocess
import sys

import numpy as np
import pytest

from shadowspill_client import BatchConfig, Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.inputs.packing import PackingError, pack_batch
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


@pytest.mark.parametrize("partitions,cooperators", [(1, 1), (2, 2)])
def test_variable_conversations_balance_record_slots_without_extra_buffers(
    partitions, cooperators
):
    # The first real UltraChat demo batch: best-fit fills the second buffer's
    # 32 record slots too early, although all 58 conversations fit in two buffers.
    lengths = [
        1013,
        1046,
        915,
        913,
        770,
        1797,
        1102,
        1957,
        609,
        1674,
        1854,
        1947,
        3507,
        1450,
        870,
        1392,
        915,
        1702,
        871,
        589,
        442,
        682,
        1095,
        1834,
        1131,
        1353,
        492,
        704,
        899,
        1622,
        2432,
        672,
        1863,
        966,
        1262,
        932,
        1224,
        1108,
        345,
        627,
        1194,
        1351,
        509,
        1373,
        950,
        883,
        1050,
        1385,
        513,
        960,
        1013,
        616,
        485,
        2060,
        417,
        1102,
        497,
        397,
    ]
    contract = schema()
    limits = BatchConfig(limits={"tokens": 65536, "longest": 8192, "records": 64})
    placement = InputPlacement(
        GroupRef(
            "inputs",
            tuple(
                tuple(range(i * cooperators, (i + 1) * cooperators))
                for i in range(partitions)
            ),
        ),
        fields=[InputFieldPlacement(("inputs", "tokens"))],
    )
    physical = BufferShape(32, {("inputs", "tokens"): (32768,)})
    schedule = PackingSchedule("two", [(physical,) * (2 // partitions)] * partitions)
    records = [Datum({"tokens": np.arange(n)}) for n in lengths]
    with snapshot_batch(records, BufferBudget(4 << 20)) as snapshot:
        batch = validate_batch(snapshot, contract, limits)
        assignment = pack_batch(
            batch, contract, limits, placement, schedule, max_repair_nodes=0
        )
        assert assignment.strategy == "balanced" and assignment.repair_nodes == 0
        assert sorted(i for row in assignment.records for i in row) == list(range(58))
        assert all(len(row) <= 32 for row in assignment.records)
        assert all(sum(lengths[i] for i in row) <= 32768 for row in assignment.records)
        assert sum(lengths) == 65333
        assert assignment.schedule.capacities(contract) == {"tokens": 65536}


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
