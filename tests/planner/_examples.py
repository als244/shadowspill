"""Small operation-neutral planner fixtures."""

from __future__ import annotations

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    MutationSpec,
    ObjectSpec,
    Program,
    RecomputationGroup,
    RecomputationOption,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.simulator import SimulationConfig

DEVICE = DeviceSpec("cuda_0", "process_0", "cuda", 0)
COMPUTE = ResourceSpec("cuda_0", ResourceKind.COMPUTE)


def exact_capacity_program() -> Program:
    return Program(
        devices=(DEVICE,),
        alias_groups=(
            AliasGroupSpec("retained", "cuda_0", 61),
            AliasGroupSpec("later", "cuda_0", 61),
            AliasGroupSpec("temporary", "cuda_0", 61),
        ),
        objects=(
            ObjectSpec("retained_object", "retained", 0, 61),
            ObjectSpec("later_object", "later", 0, 61),
            ObjectSpec("temporary_object", "temporary", 0, 61),
        ),
        profiles=(TaskProfile("task_profile", 1_000, 0, "task_abi"),),
        tasks=(
            TaskSpec(
                "task0",
                COMPUTE,
                "task_profile",
                inputs=("retained_object",),
            ),
            TaskSpec(
                "task1",
                COMPUTE,
                "task_profile",
                outputs=("temporary_object",),
            ),
            TaskSpec(
                "task2",
                COMPUTE,
                "task_profile",
                inputs=("later_object",),
            ),
        ),
    )


def exact_capacity_residency() -> tuple[
    tuple[ResidencySpec, ...], tuple[ResidencySpec, ...]
]:
    return (
        (
            ResidencySpec("retained", MemoryLocation.DEVICE),
            ResidencySpec("later", MemoryLocation.DEVICE),
        ),
        (ResidencySpec("retained", MemoryLocation.DEVICE),),
    )


def config(capacity: int = 122) -> SimulationConfig:
    return SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=capacity,
        host_capacity_bytes=1_000,
        fetch_bandwidth_bytes_per_second=61_000_000,
        evict_bandwidth_bytes_per_second=61_000_000,
    )


def mutation_program() -> Program:
    return Program(
        devices=(DEVICE,),
        alias_groups=(
            AliasGroupSpec(
                "weight_storage",
                "cuda_0",
                61,
                retain_spill_copy=True,
            ),
            AliasGroupSpec("temporary_storage", "cuda_0", 61),
        ),
        objects=(
            ObjectSpec("weight", "weight_storage", 0, 61),
            ObjectSpec("temporary", "temporary_storage", 0, 61),
        ),
        profiles=(TaskProfile("task_profile", 1_000, 0, "task_abi"),),
        tasks=(
            TaskSpec(
                "update",
                COMPUTE,
                "task_profile",
                inputs=("weight",),
                mutations=(MutationSpec("weight"),),
            ),
            TaskSpec(
                "middle",
                COMPUTE,
                "task_profile",
                outputs=("temporary",),
            ),
            TaskSpec(
                "consume",
                COMPUTE,
                "task_profile",
                inputs=("weight",),
            ),
        ),
    )


def recomputation_program() -> Program:
    return Program(
        devices=(DEVICE,),
        alias_groups=(
            AliasGroupSpec("input_storage", "cuda_0", 10),
            AliasGroupSpec("activation_storage", "cuda_0", 100),
            AliasGroupSpec("temporary_storage", "cuda_0", 100),
        ),
        objects=(
            ObjectSpec("input", "input_storage", 0, 10),
            ObjectSpec("activation", "activation_storage", 0, 100),
            ObjectSpec("temporary", "temporary_storage", 0, 100),
        ),
        profiles=(
            TaskProfile("forward_profile", 100, 0, "forward_abi"),
            TaskProfile("middle_profile", 1_000, 0, "middle_abi"),
            TaskProfile("consume_profile", 100, 0, "consume_abi"),
        ),
        tasks=(
            TaskSpec(
                "forward_save",
                COMPUTE,
                "forward_profile",
                inputs=("input",),
                outputs=("activation",),
            ),
            TaskSpec(
                "middle",
                COMPUTE,
                "middle_profile",
                dependencies=("forward_save",),
                outputs=("temporary",),
            ),
            TaskSpec(
                "forward_recompute",
                COMPUTE,
                "forward_profile",
                dependencies=("middle",),
                inputs=("input",),
                outputs=("activation",),
            ),
            TaskSpec(
                "consume",
                COMPUTE,
                "consume_profile",
                dependencies=(
                    "forward_save",
                    "middle",
                    "forward_recompute",
                ),
                inputs=("activation",),
            ),
        ),
        recomputation_groups=(
            RecomputationGroup(
                "activation_tradeoff",
                (
                    RecomputationOption(
                        "save",
                        ("forward_save",),
                        ("activation_storage",),
                    ),
                    RecomputationOption(
                        "recompute",
                        ("forward_recompute",),
                    ),
                ),
            ),
        ),
    )


def training_chain_program(layers: int) -> Program:
    """Pure IR equivalent of the retained generic training-chain canary."""

    if layers < 1:
        raise ValueError("layers must be positive")
    alias_groups = [AliasGroupSpec("input", "cuda_0", 16)]
    objects = [ObjectSpec("input", "input", 0, 16)]
    initial = [ResidencySpec("input", MemoryLocation.DEVICE)]
    for layer in range(layers):
        for prefix in ("W", "dW"):
            name = f"{prefix}_{layer}"
            alias_groups.append(
                AliasGroupSpec(name, "cuda_0", 64, retain_spill_copy=True)
            )
            objects.append(ObjectSpec(name, name, 0, 64))
            initial.append(ResidencySpec(name, MemoryLocation.HOST))
    for name in ("W_head", "dW_head"):
        alias_groups.append(AliasGroupSpec(name, "cuda_0", 64, retain_spill_copy=True))
        objects.append(ObjectSpec(name, name, 0, 64))
        initial.append(ResidencySpec(name, MemoryLocation.HOST))

    tasks: list[TaskSpec] = []
    previous: str | None = None
    for layer in range(layers):
        activation = f"A_{layer}"
        output = f"y_{layer}"
        for name in (activation, output):
            alias_groups.append(AliasGroupSpec(name, "cuda_0", 32))
            objects.append(ObjectSpec(name, name, 0, 32))
        tasks.append(
            TaskSpec(
                f"f_{layer}",
                COMPUTE,
                "forward_profile",
                dependencies=(() if previous is None else (previous,)),
                inputs=(("input" if layer == 0 else f"y_{layer - 1}"), f"W_{layer}"),
                outputs=(activation, output),
            )
        )
        previous = f"f_{layer}"
    alias_groups.append(AliasGroupSpec("dy_head", "cuda_0", 32))
    objects.append(ObjectSpec("dy_head", "dy_head", 0, 32))
    tasks.append(
        TaskSpec(
            "head",
            COMPUTE,
            "head_profile",
            dependencies=((previous,) if previous else ()),
            inputs=(f"y_{layers - 1}", "W_head", "dW_head"),
            outputs=("dy_head",),
            mutations=(MutationSpec("dW_head"),),
        )
    )
    previous = "head"
    for layer in range(layers - 1, -1, -1):
        tasks.append(
            TaskSpec(
                f"r_{layer}",
                COMPUTE,
                "marker_profile",
                dependencies=tuple(dict.fromkeys((previous, f"f_{layer}"))),
                inputs=(f"A_{layer}", f"W_{layer}"),
            )
        )
        upstream = "dy_head" if layer == layers - 1 else f"dy_{layer + 1}"
        upstream_producer = "head" if layer == layers - 1 else f"b_{layer + 1}"
        output = f"dy_{layer}"
        alias_groups.append(AliasGroupSpec(output, "cuda_0", 32))
        objects.append(ObjectSpec(output, output, 0, 32))
        tasks.append(
            TaskSpec(
                f"b_{layer}",
                COMPUTE,
                "backward_profile",
                dependencies=tuple(
                    dict.fromkeys((f"r_{layer}", upstream_producer, f"f_{layer}"))
                ),
                inputs=(upstream, f"A_{layer}", f"W_{layer}", f"dW_{layer}"),
                outputs=(output,),
                mutations=(MutationSpec(f"dW_{layer}"),),
            )
        )
        previous = f"b_{layer}"
    return Program(
        devices=(DEVICE,),
        alias_groups=tuple(alias_groups),
        objects=tuple(objects),
        profiles=(
            TaskProfile("forward_profile", 10_000, 0, "forward_abi"),
            TaskProfile("head_profile", 2_000, 0, "head_abi"),
            TaskProfile("marker_profile", 0, 0, "marker_abi"),
            TaskProfile("backward_profile", 20_000, 0, "backward_abi"),
        ),
        tasks=tuple(tasks),
    )


def training_chain_initial(layers: int) -> tuple[ResidencySpec, ...]:
    values = [ResidencySpec("input", MemoryLocation.DEVICE)]
    for layer in range(layers):
        values.extend(
            (
                ResidencySpec(f"W_{layer}", MemoryLocation.HOST),
                ResidencySpec(f"dW_{layer}", MemoryLocation.HOST),
            )
        )
    values.extend(
        (
            ResidencySpec("W_head", MemoryLocation.HOST),
            ResidencySpec("dW_head", MemoryLocation.HOST),
        )
    )
    return tuple(values)


def training_chain_config(capacity: int) -> SimulationConfig:
    return SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=capacity,
        host_capacity_bytes=10_000,
        fetch_bandwidth_bytes_per_second=8_000_000,
        evict_bandwidth_bytes_per_second=8_000_000,
    )
