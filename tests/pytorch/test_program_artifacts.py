from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qualification.planner.corpus import (
    ProgramCaseIdentity,
    load_annotated_plan,
    load_step_program,
    save_annotated_plan,
    save_step_program,
)
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
    AnnotatedProgramPlan,
    MemoryBudgets,
    PressureFitProgram,
    StepProgram,
    TransferBandwidths,
    pressurefit_program,
)
from shadowspill.simulator import SimulationConfig


def _pressurefit_program() -> PressureFitProgram:
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
    return PressureFitProgram(
        role="recurrent",
        program=program,
        initial_residency=(ResidencySpec("state", MemoryLocation.DEVICE),),
        final_residency=(ResidencySpec("state", MemoryLocation.HOST),),
        simulation_config=SimulationConfig.single_device(
            "cuda_0",
            device_capacity_bytes=96,
            host_capacity_bytes=1_024,
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


def test_annotated_program_plan_separates_budgets_and_bandwidths(
    tmp_path: Path,
) -> None:
    source = _pressurefit_program()
    assert PressureFitProgram.from_json(source.to_json()).digest == source.digest
    transfer_bandwidths = TransferBandwidths(
        1_000_000,
        2_000_000,
        provenance="test calibration",
    )

    selected = pressurefit_program(
        source,
        transfer_bandwidths=transfer_bandwidths,
        planning_cachedir=tmp_path,
        verbose=False,
    )
    cached = pressurefit_program(
        _pressurefit_program(),
        transfer_bandwidths=transfer_bandwidths,
        planning_cachedir=tmp_path,
        verbose=False,
    )
    encoded = json.loads(selected.to_json())
    restored = AnnotatedProgramPlan.from_json(selected.to_json())

    assert "benchmark_point" not in encoded
    assert encoded["memory_budgets"] == {
        "execution_bytes": 224,
        "spill_bytes": 1_024,
    }
    assert encoded["transfer_bandwidths"] == transfer_bandwidths.to_dict()
    assert restored.memory_budgets == MemoryBudgets(224, 1_024)
    assert restored.transfer_bandwidths == transfer_bandwidths
    assert restored.digest == selected.digest
    assert selected.pressurefit_wall_time_ns > 0
    assert selected.physical_admission_wall_time_ns > 0
    assert (
        selected.pressurefit_wall_time_ns
        + selected.physical_admission_wall_time_ns
        + selected.orchestration_wall_time_ns
        == selected.wall_time_ns
    )
    assert restored.pressurefit_wall_time_ns == selected.pressurefit_wall_time_ns
    assert (
        restored.physical_admission_wall_time_ns
        == selected.physical_admission_wall_time_ns
    )
    assert encoded["timing"]["total_wall_time_ns"] == selected.wall_time_ns
    assert len(encoded["timing"]["refinement_attempts"]) == len(selected.attempts)
    assert not selected.pressurefit_cache_hit
    assert cached.pressurefit_cache_hit
    assert cached.digest == selected.digest

    legacy = dict(encoded)
    legacy["timing"] = {
        "pressurefit_and_admission_wall_time_ns": selected.wall_time_ns
    }
    restored_legacy = AnnotatedProgramPlan.from_dict(legacy)
    assert restored_legacy.digest == selected.digest
    assert restored_legacy.wall_time_ns == selected.wall_time_ns
    assert restored_legacy.pressurefit_wall_time_ns == 0
    assert restored_legacy.physical_admission_wall_time_ns == 0


def test_corpus_round_trip_keeps_plan_axes_separate(tmp_path: Path) -> None:
    pressurefit_input = _pressurefit_program()
    step_program = StepProgram(
        recurrent=pressurefit_input,
        initial=None,
        optimizer_ordering="stage_interleaved",
        signature_digests=("a" * 64,),
        profiling_metadata=(),
        phase_timings_ns=(("capture", 1), ("total", 1)),
        cache_directories=(("root", str(tmp_path / "planning-cache")),),
        cache_artifacts=(),
        transfer_capabilities_json="{}",
        unique_profile_count=1,
        captured_stage_count=1,
    )
    different_evidence = replace(
        step_program,
        phase_timings_ns=(("capture", 2), ("total", 3)),
        cache_directories=(("root", "/different/cache"),),
    )
    assert different_evidence.digest == step_program.digest
    identity = ProgramCaseIdentity("llama3", "mlops", 4_096, 1_024, 2)
    saved = save_step_program(
        tmp_path / "corpus",
        identity=identity,
        program=step_program,
        metadata={"purpose": "smoke"},
    )
    assert (
        save_step_program(
            tmp_path / "corpus",
            identity=identity,
            program=step_program,
            metadata={"purpose": "smoke"},
        )
        == saved
    )
    with pytest.raises(FileExistsError, match="different evidence"):
        save_step_program(
            tmp_path / "corpus",
            identity=identity,
            program=different_evidence,
            metadata={"purpose": "smoke"},
        )
    loaded_case, loaded_program = load_step_program(saved.directory)
    selected = pressurefit_program(
        loaded_program.recurrent,
        transfer_bandwidths=TransferBandwidths(1_000_000, 2_000_000),
        planning_cachedir=tmp_path / "planning-cache",
        verbose=False,
    )
    selection_directory = save_annotated_plan(loaded_case, selected)
    restored = load_annotated_plan(selection_directory)
    repeated_directory = save_annotated_plan(
        loaded_case,
        replace(selected, wall_time_ns=selected.wall_time_ns + 1),
        step_program=loaded_program,
    )
    repeated = load_annotated_plan(repeated_directory)

    assert loaded_program.digest == step_program.digest
    assert restored.digest == selected.digest
    assert repeated.digest == selected.digest
    assert repeated.wall_time_ns == selected.wall_time_ns + 1
    assert repeated_directory != selection_directory
    assert selection_directory.parent == repeated_directory.parent
    assert "execution-224_spill-1024" in str(selection_directory)
    assert "fetch-1000000_evict-2000000" in str(selection_directory)
