"""Fresh-process smoke for reusable Program and annotated-plan artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarking.planning_eval.plan_artifacts import (
    load_annotated_plan,
    save_annotated_plan,
)
from benchmarking.program_collection.corpus import (
    ProgramCaseIdentity,
    load_step_program,
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
    PressureFitProgram,
    StepProgram,
    TransferBandwidths,
    pressurefit_program,
)
from shadowspill.simulator import SimulationConfig

_SMOKE_SCHEMA = "shadowspill.planning_corpus.smoke/v1"


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


def write_smoke(root: Path, planning_cache: Path) -> dict[str, object]:
    """Produce both artifact levels and return their fresh-process manifest."""

    step_program = _fixture()
    case = save_step_program(
        root,
        identity=ProgramCaseIdentity("fixture", "native", 1_024, 1_024, 1),
        program=step_program,
        metadata={"purpose": "fresh-process serialization smoke"},
    )
    selected = pressurefit_program(
        step_program.recurrent,
        transfer_bandwidths=TransferBandwidths(
            1_000_000,
            2_000_000,
            provenance="deterministic smoke fixture",
        ),
        planning_cachedir=planning_cache,
        verbose=False,
    )
    selection_directory = save_annotated_plan(
        case,
        selected,
        output_root=root / "evaluations",
    )
    manifest: dict[str, object] = {
        "schema": _SMOKE_SCHEMA,
        "case_directory": str(case.directory),
        "step_program_digest": step_program.digest,
        "selection_directory": str(selection_directory),
        "annotated_program_plan_digest": selected.digest,
    }
    (root / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def verify_smoke(root: Path) -> dict[str, object]:
    """Reload and verify artifacts without using any producer-process values."""

    manifest = json.loads((root / "smoke_manifest.json").read_text())
    if manifest.get("schema") != _SMOKE_SCHEMA:
        raise ValueError("smoke manifest has an unsupported schema")
    _case, step_program = load_step_program(Path(manifest["case_directory"]))
    selected = load_annotated_plan(Path(manifest["selection_directory"]))
    if step_program.digest != manifest.get("step_program_digest"):
        raise ValueError("fresh-process StepProgram digest mismatch")
    if selected.digest != manifest.get("annotated_program_plan_digest"):
        raise ValueError("fresh-process annotated-plan digest mismatch")
    return {
        "passed": True,
        "step_program_digest": step_program.digest,
        "annotated_program_plan_digest": selected.digest,
        "memory_budgets": selected.memory_budgets.to_dict(),
        "transfer_bandwidths": selected.transfer_bandwidths.to_dict(),
        "makespan_ns": selected.simulation.makespan_ns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--planning-cache", type=Path)
    parser.add_argument("--phase", choices=("write", "verify"), required=True)
    arguments = parser.parse_args()
    if arguments.phase == "write":
        if arguments.planning_cache is None:
            parser.error("--planning-cache is required for --phase=write")
        result = write_smoke(arguments.root, arguments.planning_cache)
    else:
        result = verify_smoke(arguments.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
