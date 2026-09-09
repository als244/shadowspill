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
    MutationSpec,
    ObjectRole,
    ObjectSpec,
    Persistence,
    PhysicalAdmission,
    PlanPrediction,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    TaskAlternativeChoice,
    TaskAlternativeGroup,
    TaskAlternativeOption,
    TaskProfile,
    TaskSpec,
)

SAVE_SELECTION = (TaskAlternativeChoice("activation_tradeoff", "save"),)


def representative_program() -> ShadowSpillProgram:
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    control = ResourceSpec("cuda_0", ResourceKind.CONTROL)
    return ShadowSpillProgram(
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
        task_alternative_groups=(
            TaskAlternativeGroup(
                "activation_tradeoff",
                (
                    TaskAlternativeOption(
                        "save",
                        ("forward_save",),
                        ("activation_storage",),
                    ),
                    TaskAlternativeOption("recompute", ("forward_recompute",)),
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


def write_back_program() -> ShadowSpillProgram:
    """One retained state alias: `update` writes it in place, `consume` reads
    it after a spacer long enough for a copy to land in between."""

    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    return ShadowSpillProgram(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("state_storage", "cuda_0", 128, retain_spill_copy=True),
        ),
        objects=(
            ObjectSpec("state", "state_storage", 0, 128, ObjectRole.OPTIMIZER_STATE),
        ),
        profiles=(
            TaskProfile("update_profile", 100, 16, "update_abi"),
            TaskProfile("spacer_profile", 300, 8, "spacer_abi"),
            TaskProfile("consume_profile", 100, 16, "consume_abi"),
        ),
        tasks=(
            TaskSpec(
                "update",
                compute,
                "update_profile",
                inputs=("state",),
                mutations=(MutationSpec("state"),),
            ),
            TaskSpec(
                "spacer",
                compute,
                "spacer_profile",
                dependencies=("update",),
            ),
            TaskSpec(
                "consume",
                compute,
                "consume_profile",
                dependencies=("spacer",),
                inputs=("state",),
            ),
        ),
    )


def write_back_schedule() -> MemorySchedule:
    """Write the state back after the update, so releasing it during the
    spacer costs nothing, and fetch it again for the consumer."""

    return MemorySchedule(
        initial_residency=(ResidencySpec("state_storage", MemoryLocation.DEVICE),),
        actions=(
            MemoryAction("update", "state_storage", MemoryActionKind.WRITE_BACK),
            MemoryAction("spacer", "state_storage", MemoryActionKind.RELEASE),
            MemoryAction("spacer", "state_storage", MemoryActionKind.FETCH),
        ),
        final_residency=(ResidencySpec("state_storage", MemoryLocation.DEVICE),),
    )


def release_behind_write_back_schedule() -> MemorySchedule:
    """Release and fetch at the write-back's own trigger: the release waits
    for the copy to land, and the fetch behind it waits for the release."""

    return MemorySchedule(
        initial_residency=(ResidencySpec("state_storage", MemoryLocation.DEVICE),),
        actions=(
            MemoryAction("update", "state_storage", MemoryActionKind.WRITE_BACK),
            MemoryAction("update", "state_storage", MemoryActionKind.RELEASE),
            MemoryAction("update", "state_storage", MemoryActionKind.FETCH),
        ),
        final_residency=(ResidencySpec("state_storage", MemoryLocation.DEVICE),),
    )
