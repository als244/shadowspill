"""Isolated worker that captures exactly one reusable planning Program."""

from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import replace
from pathlib import Path

from benchmarking.program_collection.corpus import (
    ProgramCaseIdentity,
    save_step_program,
)
from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    Runtime,
    StepProgram,
    export_model_state,
    import_model_state,
    make_step_program,
)

from .config import load_collection_config
from .matrix import ProgramRequest, select_program_request
from .state import atomic_json, utc_now
from .workload import build_program_case, profiling_metadata

_RESULT_SCHEMA = "shadowspill.program_corpus_collection.worker_result/v1"


def collect_program(
    request: ProgramRequest,
    *,
    output_root: Path,
    planning_cache: Path,
    verbose: bool,
    force_fresh: bool,
) -> dict[str, object]:
    """Build, cleanly release, and atomically persist one StepProgram."""

    started_at = utc_now()
    started = time.perf_counter()
    print(
        f"PROGRAM START case={request.case_id} utc={started_at} "
        f"data_geometry=({request.data_geometry.describe()})",
        flush=True,
    )
    print("PROGRAM PHASE build_model start", flush=True)
    case = build_program_case(request)
    print("PROGRAM PHASE build_model complete", flush=True)
    runtime: Runtime | None = None
    imported = False
    program: StepProgram | None = None
    primary_error: BaseException | None = None
    device_ordinal = request.runtime.execution_device or 0
    try:
        with case.implementations():
            print("PROGRAM PHASE runtime_initialize start", flush=True)
            runtime = Runtime(
                pools={
                    "execution": device(
                        physical_capacity=(
                            request.runtime.execution_pool_capacity_bytes
                        ),
                        device=device_ordinal,
                    ),
                    "spill": pinned_host(
                        capacity=request.runtime.spill_pool_capacity_bytes
                    ),
                }
            )
            print("PROGRAM PHASE runtime_initialize complete", flush=True)
            print("PROGRAM PHASE import_model start", flush=True)
            case = replace(
                case,
                model=import_model_state(
                    case.model,
                    runtime=runtime,
                    pool="spill",
                    release_source=True,
                ),
            )
            imported = True
            print("PROGRAM PHASE import_model complete", flush=True)
            print("PROGRAM PHASE capture_compile_profile_lower start", flush=True)
            program = make_step_program(
                case.model,
                objective=case.objective,
                opt=case.optimizer,
                example_inputs=case.microbatches,
                runtime=runtime,
                execution="execution",
                spill="spill",
                execution_budget=request.runtime.execution_budget_bytes,
                spill_budget=request.runtime.spill_budget_bytes,
                dynamic_scratch_reserve_bytes=(
                    request.runtime.dynamic_scratch_reserve_bytes
                ),
                execution_device=request.runtime.execution_device,
                optimizer_ordering=request.planning.optimizer_ordering,
                verbose=verbose,
                planning_cachedir=planning_cache,
                profiling_metadata=profiling_metadata(case),
                allocation_probe_seeds=request.planning.allocation_probe_seeds,
                allocation_probe_repetitions=(
                    request.planning.allocation_probe_repetitions
                ),
                save_plan=request.planning.save_plan,
                force_fresh=force_fresh or request.planning.force_fresh,
                overwrite_plan=force_fresh or request.planning.overwrite_plan,
                implementation_revision=request.planning.implementation_revision,
            )
            print("PROGRAM PHASE capture_compile_profile_lower complete", flush=True)
    except BaseException as error:
        primary_error = error
    finally:
        cleanup_errors: list[BaseException] = []
        if imported and runtime is not None:
            try:
                print("PROGRAM PHASE export_model start", flush=True)
                export_model_state(
                    case.model,
                    runtime=runtime,
                    release_runtime=True,
                )
                print("PROGRAM PHASE export_model complete", flush=True)
            except BaseException as cleanup_failure:
                cleanup_errors.append(cleanup_failure)
        if runtime is not None:
            try:
                print("PROGRAM PHASE runtime_close start", flush=True)
                runtime.close()
                print("PROGRAM PHASE runtime_close complete", flush=True)
            except BaseException as cleanup_failure:
                cleanup_errors.append(cleanup_failure)
        if primary_error is not None:
            for cleanup_detail in cleanup_errors:
                primary_error.add_note(f"worker cleanup also failed: {cleanup_detail}")
            raise primary_error
        if cleanup_errors:
            cleanup_error = cleanup_errors[0]
            for cleanup_detail in cleanup_errors[1:]:
                cleanup_error.add_note(f"additional cleanup failure: {cleanup_detail}")
            raise cleanup_error
    if program is None:
        raise AssertionError("Program collection completed without a StepProgram")
    print("PROGRAM PHASE save_artifact start", flush=True)
    saved = save_step_program(
        output_root,
        identity=ProgramCaseIdentity(
            request.model.family,
            request.model.implementation,
            request.tokens_per_microbatch,
            request.sequence_length,
            request.accumulation_rounds,
        ),
        program=program,
        metadata={
            "collection": {
                "schema": "shadowspill.program_corpus_collection.metadata/v1",
                "name": request.collection_name,
                "config_digest": request.config_digest,
                "request_digest": request.digest,
                "case_id": request.case_id,
            },
            "model": request.model.to_dict(),
            "geometry": request.to_dict()["geometry"],
            "seed": request.seed,
        },
    )
    print("PROGRAM PHASE save_artifact complete", flush=True)
    elapsed = time.perf_counter() - started
    completed_at = utc_now()
    print(
        f"PROGRAM COMPLETE case={request.case_id} utc={completed_at} "
        f"elapsed_seconds={elapsed:.3f} digest={program.digest}",
        flush=True,
    )
    return {
        "schema": _RESULT_SCHEMA,
        "passed": True,
        "case_id": request.case_id,
        "request_digest": request.digest,
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": elapsed,
        "artifact": {
            "directory": str(saved.directory),
            "program_path": str(saved.program_path),
            "program_digest": saved.program_digest,
            "artifact_digest": saved.artifact_digest,
        },
        "program_summary": {
            "captured_stage_count": program.captured_stage_count,
            "unique_profile_count": program.unique_profile_count,
            "phase_timings_ns": [list(item) for item in program.phase_timings_ns],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--planning-cache", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--quiet-plan", action="store_true")
    parser.add_argument("--force-fresh", action="store_true")
    arguments = parser.parse_args()
    try:
        config = load_collection_config(arguments.config)
        request = select_program_request(config, arguments.case_id)
        result = collect_program(
            request,
            output_root=arguments.output_dir.expanduser().resolve(),
            planning_cache=arguments.planning_cache.expanduser().resolve(),
            verbose=not arguments.quiet_plan,
            force_fresh=arguments.force_fresh,
        )
    except BaseException as error:
        result = {
            "schema": _RESULT_SCHEMA,
            "passed": False,
            "case_id": arguments.case_id,
            "completed_at": utc_now(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "notes": list(getattr(error, "__notes__", ())),
                "traceback": "".join(traceback.format_exception(error)),
            },
        }
        atomic_json(arguments.result, result)
        print(result["error"]["traceback"], flush=True)  # type: ignore[index]
        return 1
    atomic_json(arguments.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
