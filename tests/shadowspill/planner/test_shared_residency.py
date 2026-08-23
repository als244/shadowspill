from __future__ import annotations

import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    SharedResidencyPolicy,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import PressureFitOptions, pressurefit
from shadowspill.pytorch.planning.admission import build_admission_facts
from shadowspill.pytorch.planning.admission.bindings import TaskOutputBinding
from shadowspill.pytorch.profiling import (
    TaskAllocationEvent,
    TaskAllocationOperation,
)
from shadowspill.simulator import SimulationConfig


def _program() -> Program:
    return Program(
        devices=(DeviceSpec("device_0", "process_0", "device", 0),),
        alias_groups=(
            AliasGroupSpec(
                "shared_storage",
                "device_0",
                64,
                shared_residency=SharedResidencyPolicy.SHARED_READ_ONLY,
            ),
            AliasGroupSpec("output_storage", "device_0", 32),
        ),
        objects=(
            ObjectSpec("shared", "shared_storage", 0, 64),
            ObjectSpec("output", "output_storage", 0, 32),
        ),
        profiles=(TaskProfile("profile", 10, 0, "task_contract"),),
        tasks=(
            TaskSpec(
                "task",
                ResourceSpec("device_0", ResourceKind.COMPUTE),
                "profile",
                inputs=("shared",),
                outputs=("output",),
            ),
        ),
    )


def _config() -> SimulationConfig:
    return SimulationConfig.single_device(
        "device_0",
        device_capacity_bytes=96,
        spill_capacity_bytes=64,
        fetch_bandwidth_bytes_per_second=1_000_000,
        evict_bandwidth_bytes_per_second=1_000_000,
    )


def test_pressurefit_never_creates_actions_for_shared_aliases() -> None:
    result = pressurefit(
        _program(),
        initial_residency=(),
        final_residency=(ResidencySpec("output_storage", MemoryLocation.DEVICE),),
        config=_config(),
        options=PressureFitOptions(
            residency_strategies=("relaxed-stall",),
            prefetch_rules=("latest-safe",),
            evaluate_coalesced=False,
        ),
    )

    assert result.schedule.actions == ()
    assert result.schedule.initial_residency == ()
    assert result.simulation.device_peak("device_0").total_bytes == 96


def test_admission_facts_exclude_shared_execution_footprint() -> None:
    topology = build_admission_facts(
        _program(),
        execution_pool_bytes=128,
        object_capacity_bytes=96,
        output_bindings={"task": (TaskOutputBinding(0, "output_storage"),)},
        allocation_traces_by_compatibility={
            "task_contract": (
                TaskAllocationEvent(
                    0,
                    TaskAllocationOperation.ALLOCATE,
                    32,
                    32,
                    output_leaf_indices=(0,),
                    output_view_offsets=(0,),
                    alignment_bytes=1,
                ),
            )
        },
        alignment=1,
    )

    assert topology.pool_capacity_bytes == 64
    assert topology.object_capacity_bytes == 32


def test_admission_facts_require_explicit_physical_evidence() -> None:
    with pytest.raises(ValueError, match="lacks explicit physical allocation"):
        build_admission_facts(
            _program(),
            execution_pool_bytes=128,
            object_capacity_bytes=96,
            alignment=1,
        )
