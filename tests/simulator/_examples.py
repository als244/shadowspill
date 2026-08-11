"""Programs that isolate simulator timing and capacity behavior."""

from __future__ import annotations

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectRole,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.simulator import SimulationConfig


def calibrated_config(
    *,
    device_capacity_bytes: int = 1024,
    host_capacity_bytes: int = 1024,
) -> SimulationConfig:
    return SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=device_capacity_bytes,
        host_capacity_bytes=host_capacity_bytes,
        h2d_bandwidth_bytes_per_second=1_000_000_000,
        d2h_bandwidth_bytes_per_second=1_000_000_000,
    )


def overlap_program() -> Program:
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    return Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("input_storage", "cuda_0", 64),
            AliasGroupSpec("activation_storage", "cuda_0", 128),
            AliasGroupSpec("output_storage", "cuda_0", 64),
        ),
        objects=(
            ObjectSpec("input", "input_storage", 0, 64, ObjectRole.INPUT),
            ObjectSpec(
                "activation",
                "activation_storage",
                0,
                128,
                ObjectRole.ACTIVATION,
            ),
            ObjectSpec("output", "output_storage", 0, 64, ObjectRole.OUTPUT),
        ),
        profiles=(
            TaskProfile("produce_profile", 100, 16, "produce_abi"),
            TaskProfile("middle_profile", 400, 8, "middle_abi"),
            TaskProfile("spacer_profile", 300, 8, "spacer_abi"),
            TaskProfile("consume_profile", 100, 16, "consume_abi"),
        ),
        tasks=(
            TaskSpec(
                "produce",
                compute,
                "produce_profile",
                inputs=("input",),
                outputs=("activation",),
            ),
            TaskSpec(
                "middle",
                compute,
                "middle_profile",
                dependencies=("produce",),
                inputs=("input",),
            ),
            TaskSpec(
                "spacer",
                compute,
                "spacer_profile",
                dependencies=("middle",),
                inputs=("input",),
            ),
            TaskSpec(
                "consume",
                compute,
                "consume_profile",
                dependencies=("produce", "spacer"),
                inputs=("activation",),
                outputs=("output",),
            ),
        ),
    )


def overlap_schedule() -> MemorySchedule:
    return MemorySchedule(
        initial_residency=(ResidencySpec("input_storage", MemoryLocation.DEVICE),),
        actions=(
            MemoryAction(
                "produce",
                "activation_storage",
                MemoryActionKind.OFFLOAD,
            ),
            MemoryAction(
                "middle",
                "activation_storage",
                MemoryActionKind.PREFETCH,
            ),
            MemoryAction(
                "consume",
                "activation_storage",
                MemoryActionKind.RELEASE,
            ),
        ),
        final_residency=(ResidencySpec("output_storage", MemoryLocation.DEVICE),),
    )


def concurrent_lane_program() -> Program:
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    communication = ResourceSpec("cuda_0", ResourceKind.COMMUNICATION)
    return Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(AliasGroupSpec("input_storage", "cuda_0", 64),),
        objects=(ObjectSpec("input", "input_storage", 0, 64, ObjectRole.INPUT),),
        profiles=(
            TaskProfile("compute_profile", 100, 8, "compute_abi"),
            TaskProfile("communication_profile", 200, 4, "communication_abi"),
        ),
        tasks=(
            TaskSpec(
                "compute",
                compute,
                "compute_profile",
                inputs=("input",),
            ),
            TaskSpec(
                "communication",
                communication,
                "communication_profile",
                inputs=("input",),
            ),
        ),
    )


def initial_only_schedule() -> MemorySchedule:
    return MemorySchedule(
        initial_residency=(ResidencySpec("input_storage", MemoryLocation.DEVICE),),
        actions=(),
    )


def ordered_action_program() -> Program:
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    communication = ResourceSpec("cuda_0", ResourceKind.COMMUNICATION)
    return Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("long_storage", "cuda_0", 64),
            AliasGroupSpec("short_storage", "cuda_0", 32),
        ),
        objects=(
            ObjectSpec("long_output", "long_storage", 0, 64),
            ObjectSpec("short_output", "short_storage", 0, 32),
        ),
        profiles=(
            TaskProfile("long_profile", 200, 0, "long_abi"),
            TaskProfile("short_profile", 10, 0, "short_abi"),
        ),
        tasks=(
            TaskSpec(
                "long_task",
                compute,
                "long_profile",
                outputs=("long_output",),
            ),
            TaskSpec(
                "short_task",
                communication,
                "short_profile",
                outputs=("short_output",),
            ),
        ),
    )


def ordered_action_schedule() -> MemorySchedule:
    return MemorySchedule(
        initial_residency=(),
        actions=(
            MemoryAction("long_task", "long_storage", MemoryActionKind.OFFLOAD),
            MemoryAction("short_task", "short_storage", MemoryActionKind.OFFLOAD),
        ),
        final_residency=(
            ResidencySpec("long_storage", MemoryLocation.HOST),
            ResidencySpec("short_storage", MemoryLocation.HOST),
        ),
    )
