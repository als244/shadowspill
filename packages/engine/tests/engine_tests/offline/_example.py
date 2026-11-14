"""A tiny update with analytically known copies, compute and peak memory."""

from shadowspill import CommunicationInfo, Profiles, TaskProfile, apply_profiles
from shadowspill.artifacts import Program, TaskDefinition
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
from shadowspill.planner import GenericPlanningOptions, SearchOptions


def program(*, workspace=None, overhead=None):
    return Program(
        devices=(DeviceSpec("rank0", "worker", "cuda", 0),),
        alias_groups=(AliasGroupSpec("weight_storage", "rank0", 64),),
        objects=(ObjectSpec("weight", "weight_storage", 0, 64),),
        tasks=(
            TaskSpec(
                "begin",
                ResourceSpec("rank0", ResourceKind.COMPUTE),
                "boundary",
                requires_entrypoint=False,
                phase="control",
            ),
            TaskSpec(
                "update",
                ResourceSpec("rank0", ResourceKind.COMPUTE),
                "update_impl",
                dependencies=("begin",),
                inputs=("weight",),
                mutations=(MutationSpec("weight"),),
                phase="optimizer",
            ),
            TaskSpec(
                "end",
                ResourceSpec("rank0", ResourceKind.COMPUTE),
                "boundary",
                dependencies=("update",),
                requires_entrypoint=False,
                phase="control",
            ),
        ),
        task_definitions=(
            TaskDefinition(
                "update_impl",
                "implementation-v1",
                workspace,
                "linear.weight update",
                ("linear",),
            ),
            TaskDefinition("boundary", "boundary-v1", 0),
        ),
        initial_residency=(ResidencySpec("weight_storage", MemoryLocation.SPILL),),
        final_residency=(ResidencySpec("weight_storage", MemoryLocation.SPILL),),
        name="Tiny update",
        sources={"weight_storage": ["linear.weight"]},
        fixed_device_bytes=overhead,
    )


def profiles():
    return Profiles(
        tasks={
            "update_impl": TaskProfile(duration_seconds=1e-6, workspace_bytes=16),
            "boundary": TaskProfile(0),
        },
        fixed_device_bytes=20,
    )


def profiled():
    return apply_profiles(program(), profiles())


def communication(latency=0):
    return CommunicationInfo(
        64e6, 64e6, fetch_latency_seconds=latency, evict_latency_seconds=latency
    )


def options():
    return SearchOptions(
        generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0),
        workers=1,
    )
