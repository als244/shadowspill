"""Partial gradients spill as reduced shards before later backward contributions."""

from dataclasses import replace

from shadowspill import (
    MemoryBudgets,
    Parallelism,
    Profiles,
    Program,
    TaskProfile,
    apply_profiles,
    plan_program,
)
from shadowspill.artifacts import TaskDefinition
from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    MutationSpec,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskSpec,
)
from shadowspill.simulator import TransferDirection

from ._communication import estimated_communication
from ._example import options


def test_capacity_forces_early_reduction_and_shard_only_spill(tmp_path):
    lane = ResourceSpec("gpu", ResourceKind.COMPUTE)
    source = Program(
        devices=(DeviceSpec("gpu", "worker", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("gradient", "gpu", 128),
            AliasGroupSpec("other", "gpu", 128),
        ),
        objects=(
            ObjectSpec("gradient", "gradient", 0, 128),
            ObjectSpec("other", "other", 0, 128),
        ),
        tasks=(
            TaskSpec("first", lane, "local", outputs=("gradient",)),
            TaskSpec(
                "other", lane, "local", dependencies=("first",), outputs=("other",)
            ),
            TaskSpec(
                "consume", lane, "local", dependencies=("other",), inputs=("other",)
            ),
            TaskSpec(
                "add",
                lane,
                "local",
                dependencies=("first", "consume"),
                inputs=("gradient",),
                mutations=(MutationSpec("gradient"),),
            ),
        ),
        task_definitions=(TaskDefinition("local", "local", 0),),
        final_residency=(ResidencySpec("gradient", MemoryLocation.SPILL),),
        setup={"gpu": "fixture"},
        parallelism=Parallelism.dp(ranks_per_replica=2, num_replicas=2),
        bindings={
            "kind": "step",
            "distributed_training_lowered": "step",
            "weight_shards": {},
            "gradient_shards": {
                "gradient": {
                    "groups": ((0, 1), (2, 3)),
                    "replica_groups": ((0, 2), (1, 3)),
                    "logical_bytes": 128,
                    "padded_bytes": 128,
                    "shard_bytes": 64,
                    "dtype": "torch.float32",
                }
            },
        },
    )
    profiles = Profiles(
        tasks={"local": TaskProfile(1e-6, workspace_bytes=0)},
        fixed_device_bytes=0,
        setup=source.setup,
    )
    info = estimated_communication(source, topology={"node": (0, 1, 2, 3)})
    # Supply measured zero time; collective times intentionally use the ring estimate.
    info = replace(
        info,
        measurements=tuple(
            {**row, "zero_duration_seconds": 2e-6}
            if row["operation"] == "reduce_scatter"
            else row
            for row in info.measurements
        ),
    )
    result = plan_program(
        apply_profiles(source, profiles),
        memory_budgets=MemoryBudgets(128, 64),
        communication_info=info,
        search_options=options(),
    )
    transfer = result._costed_program.alias_groups[0].transfer
    assert transfer.spill_size_bytes == 64
    assert transfer.fetch[0].duration_ns == 2000
    assert transfer.fetch[0].timing_class == "zero_measurement"
    assert transfer.fetch[1].bytes == 64
    assert transfer.evict[0].bytes == 128  # RS 64 + replica AR 64, per rank.
    assert transfer.evict[1].bytes == 64
    copies = [
        i for i in result.timeline.transfer_intervals if i.alias_group_id == "gradient"
    ]
    evictions = [i for i in copies if i.direction == TransferDirection.EVICT]
    fetches = [i for i in copies if i.direction == TransferDirection.FETCH]
    assert len(evictions) == 2 and len(fetches) == 1
    tasks = {i.task_id: i for i in result.timeline.task_intervals}
    assert evictions[0].end_ns <= tasks["other"].start_ns
    assert fetches[0].end_ns <= tasks["add"].start_ns
    assert all(i.bytes == 64 for i in copies)
    assert result.summary["spill_peak_bytes"] == 64
    assert result.summary["execution_peak_bytes"] == 128
