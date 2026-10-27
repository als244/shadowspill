"""Small reusable ShadowSpillProgram fixtures for benchmarking tests."""

from __future__ import annotations

from shadowspill import apply_profiles
from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import (
    AdmissionFacts,
    TaskAdmissionSpec,
)
from shadowspill.planner.program import (
    ShadowSpillPlanningProblem,
)
from shadowspill.simulator import SimulationConfig
from tests.benchmarking._profiled import measured
from tests.shadowspill.offline._costed_fixture import split_costed_program


def problem() -> ShadowSpillPlanningProblem:
    program = ShadowSpillProgram(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(AliasGroupSpec("state", "cuda_0", 64, retain_spill_copy=True),),
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
    return ShadowSpillPlanningProblem(
        role="step",
        program=program,
        initial_residency=(ResidencySpec("state", MemoryLocation.DEVICE),),
        final_residency=(ResidencySpec("state", MemoryLocation.SPILL),),
        simulation_config=SimulationConfig.single_device(
            "cuda_0",
            device_capacity_bytes=96,
            spill_capacity_bytes=1_024,
            fetch_solo_bandwidth_bytes_per_second=1_000_000,
            fetch_concurrent_bandwidth_bytes_per_second=(1_000_000),
            evict_solo_bandwidth_bytes_per_second=2_000_000,
            evict_concurrent_bandwidth_bytes_per_second=(2_000_000),
        ),
        admission_facts=AdmissionFacts(
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
    )


def _fixture():
    source = problem()
    program, profiles = split_costed_program(
        source.program,
        initial_residency=source.initial_residency,
        final_residency=source.final_residency,
        fixed_device_bytes=source.fixed_execution_bytes,
    )
    return measured(apply_profiles(program, profiles))
