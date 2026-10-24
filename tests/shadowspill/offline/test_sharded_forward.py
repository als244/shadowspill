"""Analytic sharded fetch accounting through the actual CPU planner/simulator."""

from dataclasses import replace

import pytest

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    Parallelism,
    Plan,
    apply_profiles,
    plan_program,
)
from shadowspill.reporting.rows import transfer_rows
from shadowspill.simulator import TransferDirection

from ._example import options, profiles, program


def forward():
    value = program()
    return replace(
        value,
        tasks=tuple(
            replace(task, mutations=(), phase="forward")
            if task.task_id == "update"
            else task
            for task in value.tasks
        ),
        setup={"gpu": "fixture"},
        parallelism=Parallelism.dp(ranks_per_replica=4),
        bindings={
            "kind": "forward",
            "weight_shards": {
                "weight_storage": {
                    "groups": ((0, 1, 2, 3),),
                    "shard_bytes": 16,
                    "logical_bytes": 64,
                    "padded_bytes": 64,
                    "dtype": "torch.uint8",
                },
            },
        },
    )


def test_planner_keeps_local_copies_and_full_gather_as_distinct_phases(tmp_path):
    profile = profiles()
    profile.setup = {"gpu": "fixture"}
    profiled = apply_profiles(forward(), profile)
    info = CommunicationInfo(
        64e6,
        64e6,
        topology={"node": (0, 1, 2, 3)},
        scale_up_bytes_per_second=48e6,
        scale_up_latency_seconds=1e-6,
    )
    result = plan_program(
        profiled,
        memory_budgets=MemoryBudgets(100, 16),
        communication_info=info,
        search_options=options(),
    )
    transfer = result._costed_program.alias_groups[0].transfer
    assert transfer.spill_size_bytes == 16
    assert transfer.fetch[0].kind == "copy"
    assert transfer.fetch[0].bytes == 16
    assert transfer.fetch[1].bytes == 48
    assert transfer.fetch[1].duration_ns == 4000  # 3 us startup + 48 bytes / 48 MB/s.
    assert transfer.evict[0].bytes == 16
    fetched = [
        item
        for item in result.timeline.transfer_intervals
        if item.direction == TransferDirection.FETCH
    ]
    # Native parent intervals count host bytes; the recipe records other legs.
    assert sum(item.bytes for item in fetched) == 16
    assert sum(item.end_ns - item.start_ns for item in fetched) == 4250
    assert result.summary["logical_ranks"] == 4
    assert result.summary["fetch_bytes"] == 64
    assert result.summary["fetch_host_bytes"] == 16
    assert result.summary["fetch_collective_bytes"] == 48
    rows = [row for row in transfer_rows(result) if row["direction"] == "fetch"]
    assert rows[0]["kind"] == "copy_and_collective"
    assert rows[0]["bytes"] == 64
    assert rows[0]["throughput_gb_per_second"] is None
    assert result.summary["spill_peak_bytes"] == 16
    assert result.summary["simulation_only"]
    assert any("distributed" in reason for reason in result.simulation_only_reasons)
    result.save(tmp_path / "sharded.ssp")
    loaded = Plan.load(tmp_path / "sharded.ssp")
    assert loaded.program.parallelism == result.program.parallelism
    assert loaded.timeline == result.timeline

    # A calibrated collective replaces only that phase, never its host copy.
    measured = replace(
        info,
        measurements=(
            {
                "operation": "all_gather",
                "ranks": (0, 1, 2, 3),
                "tensor_bytes": 64,
                "dtype": "torch.uint8",
                "duration_seconds": 2e-6,
                "setup": {"gpu": "fixture"},
                "topology": info.topology,
            },
        ),
    )
    faster = plan_program(
        profiled,
        memory_budgets=result.memory_budgets,
        communication_info=measured,
        search_options=options(),
    )
    assert faster.timeline.makespan_ns == result.timeline.makespan_ns - 2000
    assert faster._costed_program.alias_groups[0].transfer.fetch[0].bytes == 16


def test_unlowered_distributed_training_cannot_be_planned_as_local():
    value = replace(program(), parallelism=Parallelism.dp(ranks_per_replica=4))
    with pytest.raises(NotImplementedError, match="distributed training transfer"):
        plan_program(
            apply_profiles(value, profiles()),
            memory_budgets=MemoryBudgets(100, 64),
            communication_info=CommunicationInfo(64e6, 64e6),
        )
