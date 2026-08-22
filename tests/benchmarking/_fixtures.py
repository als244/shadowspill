"""Small reusable Program fixtures for benchmarking tests."""

from __future__ import annotations

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import (
    AdmissionTopology,
    PressureFitOptions,
    TaskAdmissionSpec,
)
from shadowspill.pytorch import (
    PressureFitProgram,
    StepProgram,
)
from shadowspill.simulator import SimulationConfig


def _fixture() -> StepProgram:
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("state", "cuda_0", 64, retain_spill_copy=True),
        ),
        objects=(ObjectSpec("state_object", "state", 0, 64),),
        profiles=(TaskProfile("profile", 10, 0, "abi"),),
        tasks=(
            TaskSpec(
                "task",
                ResourceSpec("cuda_0", ResourceKind.COMPUTE),
                "profile",
                inputs=("state_object",),
            ),
        ),
    )
    pre_pressurefit = PressureFitProgram(
        role="recurrent",
        program=program,
        initial_residency=(ResidencySpec("state", MemoryLocation.DEVICE),),
        final_residency=(ResidencySpec("state", MemoryLocation.SPILL),),
        simulation_config=SimulationConfig.single_device(
            "cuda_0",
            device_capacity_bytes=96,
            spill_capacity_bytes=1_024,
            fetch_bandwidth_bytes_per_second=1_000_000,
            evict_bandwidth_bytes_per_second=2_000_000,
        ),
        admission_topology=AdmissionTopology(
            "cuda_0",
            192,
            96,
            1,
            (TaskAdmissionSpec("task"),),
        ),
        source_execution_budget_bytes=224,
        maximum_execution_budget_bytes=512,
        maximum_spill_budget_bytes=2_048,
        fixed_execution_bytes=32,
        object_reserve_bytes=96,
        dynamic_scratch_reserve_bytes=0,
        options=PressureFitOptions(
            residency_strategies=("tight-stall",),
            prefetch_rules=("demand",),
            evaluate_coalesced=False,
            workers=1,
        ),
    )
    return StepProgram(
        recurrent=pre_pressurefit,
        initial=None,
        optimizer_ordering="stage_interleaved",
        signature_digests=("0" * 64,),
        profiling_metadata=(),
        phase_timings_ns=(("fixture", 1), ("total", 1)),
        cache_directories=(),
        cache_artifacts=(),
        transfer_capabilities_json="{}",
        unique_profile_count=1,
        captured_stage_count=1,
    )
