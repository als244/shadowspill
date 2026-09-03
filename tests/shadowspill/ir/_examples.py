"""Small, fully resolved programs shared by IR contract tests."""

from __future__ import annotations

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    EntrypointSpec,
    ExecutionPlan,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectRole,
    ObjectSpec,
    Persistence,
    PhysicalAdmission,
    PlanPrediction,
    Program,
    RecomputationGroup,
    RecomputationOption,
    RecomputationSelection,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)

SAVE_SELECTION = (RecomputationSelection("activation_tradeoff", "save"),)


def representative_program() -> Program:
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    control = ResourceSpec("cuda_0", ResourceKind.CONTROL)
    return Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("input_storage", "cuda_0", 64),
            AliasGroupSpec(
                "weight_storage",
                "cuda_0",
                256,
                retain_spill_copy=True,
            ),
            AliasGroupSpec("activation_storage", "cuda_0", 128),
            AliasGroupSpec("output_storage", "cuda_0", 64),
        ),
        objects=(
            ObjectSpec("input", "input_storage", 0, 64, ObjectRole.INPUT),
            ObjectSpec(
                "weight",
                "weight_storage",
                0,
                256,
                ObjectRole.PARAMETER,
                Persistence.CHECKPOINT,
            ),
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
            TaskProfile("forward_profile", 10, 8, "forward_abi"),
            TaskProfile("marker_profile", 0, 0, "marker_abi"),
            TaskProfile("consume_profile", 12, 16, "consume_abi"),
        ),
        tasks=(
            TaskSpec(
                "forward_save",
                compute,
                "forward_profile",
                inputs=("input", "weight"),
                outputs=("activation",),
                phase="forward",
            ),
            TaskSpec(
                "backward_marker",
                control,
                "marker_profile",
                dependencies=("forward_save",),
                phase="control",
                requires_entrypoint=False,
            ),
            TaskSpec(
                "forward_recompute",
                compute,
                "forward_profile",
                dependencies=("backward_marker",),
                inputs=("input", "weight"),
                outputs=("activation",),
                phase="recomputation",
            ),
            TaskSpec(
                "consume",
                compute,
                "consume_profile",
                dependencies=(
                    "forward_save",
                    "backward_marker",
                    "forward_recompute",
                ),
                inputs=("activation",),
                outputs=("output",),
                phase="backward",
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
                    RecomputationOption("recompute", ("forward_recompute",)),
                ),
            ),
        ),
    )


def representative_schedule() -> MemorySchedule:
    return MemorySchedule(
        initial_residency=(
            ResidencySpec("input_storage", MemoryLocation.DEVICE),
            ResidencySpec("weight_storage", MemoryLocation.DEVICE),
        ),
        actions=(
            MemoryAction(
                "forward_save",
                "activation_storage",
                MemoryActionKind.EVICT,
            ),
            MemoryAction(
                "backward_marker",
                "activation_storage",
                MemoryActionKind.FETCH,
            ),
            MemoryAction(
                "consume",
                "activation_storage",
                MemoryActionKind.RELEASE,
            ),
        ),
        final_residency=(ResidencySpec("output_storage", MemoryLocation.DEVICE),),
    )


def representative_plan() -> ExecutionPlan:
    return ExecutionPlan(
        program=representative_program(),
        schedule=representative_schedule(),
        selections=SAVE_SELECTION,
        entrypoints=(
            EntrypointSpec(
                "forward_save",
                "forward_entrypoint",
                "pytorch",
                "forward_abi",
            ),
            EntrypointSpec(
                "consume",
                "consume_entrypoint",
                "pytorch",
                "consume_abi",
            ),
        ),
        admission=PhysicalAdmission(
            device_budget_bytes=1024,
            spill_budget_bytes=1024,
            baseline_bytes=64,
            provider_headroom_bytes=64,
            slab_bytes=896,
            workspace_reserve_bytes=128,
            spill_reservation_bytes=256,
            predicted_fragmentation_bytes=32,
        ),
        prediction=PlanPrediction(
            device_peak_bytes=900,
            spill_peak_bytes=128,
            makespan_ns=38,
        ),
    )
