"""Build and profile one case using the ordinary allocator, then save its archive."""

import argparse
import time
import traceback
from pathlib import Path

from benchmarking._serialization import atomic_json
from shadowspill import apply_profiles, measure_program

from .artifacts import save_profiled_program
from .config import load_collection_config
from .corpus import ProgramCaseIdentity
from .matrix import ProgramRequest, select_program_request
from .state import utc_now
from .workload import build_program


def collect_program(request: ProgramRequest, *, output_root: Path) -> dict[str, object]:
    started_at, started = utc_now(), time.perf_counter()
    print(f"PROGRAM START case={request.case_id} utc={started_at}", flush=True)
    print("PROGRAM PHASE build start", flush=True)
    program = build_program(request)
    print("PROGRAM PHASE build complete; profile start", flush=True)
    profiles = measure_program(program, **request.profiling.to_dict())
    profiled = apply_profiles(program, profiles)
    print("PROGRAM PHASE profile complete; save start", flush=True)
    saved = save_profiled_program(
        output_root,
        identity=ProgramCaseIdentity(
            request.model.family,
            request.model.implementation,
            request.tokens_per_microbatch,
            request.sequence_length,
            request.accumulation_rounds,
        ),
        program=profiled,
        metadata=request.to_dict(),
    )
    elapsed = time.perf_counter() - started
    print(
        f"PROGRAM COMPLETE case={request.case_id} elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    return {
        "schema": "shadowspill.program_corpus_collection.worker_result/v2",
        "passed": True,
        "case_id": request.case_id,
        "request_digest": request.digest,
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed,
        "artifact": {
            "directory": str(saved.directory),
            "program_path": str(saved.program_path),
            "program_digest": saved.program_digest,
            "artifact_digest": saved.artifact_digest,
        },
        "program_summary": {
            "unique_profile_count": len(profiled.task_profiles),
            "fixed_device_bytes": profiled.fixed_device_bytes,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        request = select_program_request(
            load_collection_config(arguments.config), arguments.case_id
        )
        result = collect_program(
            request, output_root=arguments.output_dir.expanduser().resolve()
        )
    except BaseException as error:
        result = {
            "passed": False,
            "case_id": arguments.case_id,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        traceback.print_exc()
    atomic_json(arguments.result, result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
