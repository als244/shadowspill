"""One-process evaluation of every frontier point for one saved Program."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmarking.program_collection.corpus import (
    SavedProgramCase,
    load_step_program,
)
from shadowspill.errors import (
    PlanInfeasibleError,
    PlanSearchExhaustedError,
)
from shadowspill.planner import (
    PressureFitInfeasibleError,
    PressureFitOptions,
    PressureFitSearchExhaustedError,
)
from shadowspill.pytorch import (
    PressureFitProgram,
    StepProgram,
    pressurefit_program,
)
from shadowspill.simulator import SimulationInfeasibleError

from .config import FrontierConfig, load_frontier_config
from .evidence import failed_point_evidence, successful_point_evidence
from .matrix import FrontierPointRequest, expand_frontier_points
from .plan_artifacts import save_annotated_plan
from .source import CorpusProgramCase
from .storage import (
    BaselinePaths,
    atomic_json,
    begin_point_attempt,
    finish_point_attempt,
    initialize_point,
    point_attempt_count,
    point_complete,
    utc_now,
    write_active_point,
)

_EXPECTED_INFEASIBLE = (
    PressureFitInfeasibleError,
    PlanInfeasibleError,
    SimulationInfeasibleError,
)
_EXPECTED_EXHAUSTED = (
    PressureFitSearchExhaustedError,
    PlanSearchExhaustedError,
)


def _planner_options(config: FrontierConfig) -> PressureFitOptions | None:
    """The planner settings this config names, or None for the defaults."""

    named = {
        name: value
        for name, value in (
            ("capacity_refinement_bytes", config.capacity_refinement_bytes),
            ("max_repair_attempts", config.max_repair_attempts),
        )
        if value is not None
    }
    return PressureFitOptions(**named) if named else None


def evaluate_case(
    *,
    config: FrontierConfig,
    baseline_directory: Path,
    case_directory: Path,
    planning_cache: Path,
    verbose_pressurefit: bool,
    global_point_base: int,
    global_point_count: int,
    revision: str,
) -> dict[str, object]:
    """Load one Program once, then independently persist every point."""

    saved_case, step_program = load_step_program(case_directory)
    program = _select_program(step_program, config.program_role)
    case = CorpusProgramCase(
        saved_case.directory,
        saved_case.identity,
        saved_case.program_digest,
        saved_case.artifact_digest,
    )
    paths = _baseline_paths(baseline_directory)
    requests = expand_frontier_points(
        program,
        config.grids,
        transfer_baseline=config.transfer_bandwidths,
    )
    if len(requests) != config.expected_points_per_program:
        raise ValueError("worker frontier point count differs from config")
    if global_point_base < 0:
        raise ValueError("global point base must be nonnegative")
    if global_point_base + len(requests) > global_point_count:
        raise ValueError("Program points exceed the global frontier size")
    case_run_directory = paths.case_directory(case)
    case_run_directory.mkdir(parents=True, exist_ok=True)
    atomic_json(
        case_run_directory / "case.json",
        {
            "schema": "shadowspill.pressurefit_frontier_case/v1",
            "case": case.to_dict(),
            "program_role": program.role,
            "pressurefit_program_digest": program.digest,
            "points": [item.to_dict() for item in requests],
        },
    )
    write_active_point(case_run_directory, None)
    counts: dict[str, int] = {}
    for ordinal, request in enumerate(requests, 1):
        directory = initialize_point(paths, case, request)
        while not point_complete(directory, request):
            if point_attempt_count(directory, request) >= config.max_point_attempts:
                raise RuntimeError(
                    f"non-final point exhausted attempts: {request.point_id}"
                )
            _evaluate_point(
                config=config,
                paths=paths,
                case=case,
                saved_case=saved_case,
                step_program=step_program,
                program=program,
                request=request,
                directory=directory,
                planning_cache=planning_cache,
                verbose_pressurefit=verbose_pressurefit,
                ordinal=ordinal,
                point_count=len(requests),
                global_ordinal=global_point_base + ordinal,
                global_point_count=global_point_count,
                revision=revision,
            )
        status = _point_status(directory)
        counts[status] = counts.get(status, 0) + 1
    write_active_point(case_run_directory, None)
    result = {
        "schema": "shadowspill.pressurefit_frontier_worker_result/v1",
        "passed": counts.get("error", 0) == 0,
        "case_id": case.case_id,
        "case": case.to_dict(),
        "counts": counts,
        "completed_at": utc_now(),
    }
    atomic_json(case_run_directory / "worker-result.json", result)
    return result


def _evaluate_point(
    *,
    config: FrontierConfig,
    paths: BaselinePaths,
    case: CorpusProgramCase,
    saved_case: SavedProgramCase,
    step_program: StepProgram,
    program: PressureFitProgram,
    request: FrontierPointRequest,
    directory: Path,
    planning_cache: Path,
    verbose_pressurefit: bool,
    ordinal: int,
    point_count: int,
    global_ordinal: int,
    global_point_count: int,
    revision: str,
) -> None:
    attempt = begin_point_attempt(directory, request, revision=revision)
    write_active_point(paths.case_directory(case), request)
    started_at = utc_now()
    started = time.perf_counter()
    _print_point_start(
        case=case,
        request=request,
        attempt=attempt,
        started_at=started_at,
        ordinal=ordinal,
        point_count=point_count,
        global_ordinal=global_ordinal,
        global_point_count=global_point_count,
        revision=revision,
    )
    try:
        plan = pressurefit_program(
            program,
            execution_budget=request.axes.execution_budget_bytes,
            spill_budget=request.axes.spill_budget_bytes,
            transfer_bandwidths=request.transfer_bandwidths,
            options=_planner_options(config),
            planning_cachedir=planning_cache,
            verbose=verbose_pressurefit,
            save_plan=config.pressurefit_cache_mode == "warm",
            force_fresh=config.pressurefit_cache_mode == "cold",
            overwrite_plan=False,
        )
        annotated_directory = save_annotated_plan(
            saved_case,
            plan,
            metadata={
                "purpose": "pressurefit-frontier",
                "program_role": config.program_role,
            },
            step_program=step_program,
            output_root=paths.case_directory(case) / "annotated-plans",
        )
        elapsed = time.perf_counter() - started
        completed_at = utc_now()
        evidence = successful_point_evidence(
            case=saved_case,
            request=request,
            plan=plan,
            annotated_plan_directory=annotated_directory,
            started_at=started_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed,
        )
        finish_point_attempt(
            directory,
            request,
            attempt=attempt,
            status_name="succeeded",
            elapsed_seconds=elapsed,
            evidence=evidence,
            error=None,
            final=True,
        )
        _print_point_stop(
            case=case,
            request=request,
            status="SUCCESS",
            started_at=started_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed,
            global_ordinal=global_ordinal,
            global_point_count=global_point_count,
            details=(
                f"MAKESPAN: {plan.simulation.makespan_ns / 1e9:.6f} seconds",
                f"PLAN DIGEST: {plan.digest}",
            ),
        )
    except _EXPECTED_INFEASIBLE as error:
        _finish_error(
            directory,
            case,
            request,
            attempt=attempt,
            status="infeasible",
            error=error,
            started_at=started_at,
            started=started,
            final=True,
            global_ordinal=global_ordinal,
            global_point_count=global_point_count,
        )
    except _EXPECTED_EXHAUSTED as error:
        _finish_error(
            directory,
            case,
            request,
            attempt=attempt,
            status="search_exhausted",
            error=error,
            started_at=started_at,
            started=started,
            final=True,
            global_ordinal=global_ordinal,
            global_point_count=global_point_count,
        )
    except BaseException as error:
        attempts_exhausted = attempt >= config.max_point_attempts
        _finish_error(
            directory,
            case,
            request,
            attempt=attempt,
            status="error",
            error=error,
            started_at=started_at,
            started=started,
            final=attempts_exhausted,
            global_ordinal=global_ordinal,
            global_point_count=global_point_count,
        )
    finally:
        write_active_point(paths.case_directory(case), None)


def _finish_error(
    directory: Path,
    case: CorpusProgramCase,
    request: FrontierPointRequest,
    *,
    attempt: int,
    status: str,
    error: BaseException,
    started_at: str,
    started: float,
    final: bool,
    global_ordinal: int,
    global_point_count: int,
) -> None:
    elapsed = time.perf_counter() - started
    completed_at = utc_now()
    evidence = failed_point_evidence(
        case=case,
        request=request,
        error=error,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed,
    )
    raw_error = evidence["error"]
    assert isinstance(raw_error, dict)
    finish_point_attempt(
        directory,
        request,
        attempt=attempt,
        status_name=status,
        elapsed_seconds=elapsed,
        evidence=evidence if final else None,
        error=raw_error,
        final=final,
    )
    disposition = "FINAL" if final else "RETRY"
    _print_point_stop(
        case=case,
        request=request,
        status=f"{status.upper()} {disposition}",
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed,
        global_ordinal=global_ordinal,
        global_point_count=global_point_count,
        details=(
            f"ERROR TYPE: {type(error).__name__}",
            f"ERROR: {error}",
        ),
    )


def _print_point_start(
    *,
    case: CorpusProgramCase,
    request: FrontierPointRequest,
    attempt: int,
    started_at: str,
    ordinal: int,
    point_count: int,
    global_ordinal: int,
    global_point_count: int,
    revision: str,
) -> None:
    identity = case.identity
    execution_budget = request.axes.execution_budget_bytes
    spill_budget = request.axes.spill_budget_bytes
    transfers = request.transfer_bandwidths
    geometry = identity.data_geometry
    print()
    print(f"POINT [{global_ordinal}/{global_point_count}] START")
    print(f"  PROGRAM POINT: [{ordinal}/{point_count}]")
    print(f"  MODEL: {identity.provider}/{identity.family}")
    print("  DATA GEOMETRY:")
    print(f"    SEQUENCE LENGTH: {geometry.sequence_length} tokens")
    print(f"    TOKENS PER MICROBATCH: {geometry.tokens_per_microbatch}")
    print(f"    SEQUENCES PER MICROBATCH: {geometry.sequences_per_microbatch}")
    print(
        "    GRADIENT ACCUMULATION ROUNDS: "
        f"{geometry.accumulation_rounds}"
    )
    print(
        "    TOKENS PER OPTIMIZER STEP: "
        f"{geometry.tokens_per_optimizer_step}"
    )
    print(
        "  EXECUTION BUDGET: "
        f"{execution_budget / (1 << 30):.3f} GiB ({execution_budget} bytes)"
    )
    print(
        f"  SPILL BUDGET: {spill_budget / (1 << 30):.3f} GiB "
        f"({spill_budget} bytes)"
    )
    print(
        "  FETCH BANDWIDTH: "
        f"{transfers.fetch_bytes_per_second / 1e9:.6f} GB/s "
        f"({transfers.fetch_bytes_per_second} B/s)"
    )
    print(
        "  EVICT BANDWIDTH: "
        f"{transfers.evict_bytes_per_second / 1e9:.6f} GB/s "
        f"({transfers.evict_bytes_per_second} B/s)"
    )
    print(f"  PROGRAM ROLE: {request.role}")
    print(f"  PROGRAM DIGEST: {request.program_digest}")
    print(f"  ATTEMPT: {attempt}")
    print(f"  START: {started_at}", flush=True)


def _print_point_stop(
    *,
    case: CorpusProgramCase,
    request: FrontierPointRequest,
    status: str,
    started_at: str,
    completed_at: str,
    elapsed_seconds: float,
    global_ordinal: int,
    global_point_count: int,
    details: tuple[str, ...],
) -> None:
    print(f"POINT [{global_ordinal}/{global_point_count}] {status}")
    print(f"  MODEL: {case.identity.provider}/{case.identity.family}")
    print(f"  REQUEST: {request.point_id}")
    for detail in details:
        print(f"  {detail}")
    print(f"  START: {started_at}")
    print(f"  STOP: {completed_at}")
    print(f"  DURATION: {elapsed_seconds:.3f} seconds")
    print(flush=True)


def _select_program(step: StepProgram, role: str) -> PressureFitProgram:
    if role == "recurrent":
        return step.recurrent
    if role == "initial":
        if step.initial is None:
            raise ValueError("saved StepProgram has no initial variant")
        return step.initial
    if role == "forward" and step.recurrent.role == "forward":
        return step.recurrent
    raise ValueError(f"saved StepProgram does not provide role {role!r}")


def _baseline_paths(directory: Path) -> BaselinePaths:
    root = directory.expanduser().resolve()
    return BaselinePaths(
        root,
        root / "cases",
        root / "config.json",
        root / "manifest.json",
        root / "summary.json",
        root / "frontier.csv",
        root / "frontier.jsonl",
        root / "collection.log",
        root / "planner.patch",
    )


def _point_status(directory: Path) -> str:
    import json

    value = json.loads((directory / "point.json").read_text())
    status = value.get("status")
    if not isinstance(status, str):
        raise ValueError(f"point has invalid final status at {directory}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--planning-cache", type=Path, required=True)
    parser.add_argument("--global-point-base", type=int, required=True)
    parser.add_argument("--global-point-count", type=int, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--verbose-pressurefit", action="store_true")
    arguments = parser.parse_args()
    config = load_frontier_config(arguments.config)
    result = evaluate_case(
        config=config,
        baseline_directory=arguments.baseline_dir,
        case_directory=arguments.case_dir,
        planning_cache=arguments.planning_cache.expanduser().resolve(),
        verbose_pressurefit=arguments.verbose_pressurefit,
        global_point_base=arguments.global_point_base,
        global_point_count=arguments.global_point_count,
        revision=arguments.revision,
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
