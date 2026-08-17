"""Compiled, simulator-verified PressureFit orchestration."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from shadowspill.ir import Program, RecomputationSelection, ResidencySpec
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator._capi import load_simulator_library
from shadowspill.simulator._compiled import (
    CompiledSimulationTemplate,
    compile_simulation_template,
    simulate_compiled_template,
)

from ._admission import (
    CompiledAdmissionTopology,
    compile_admission_topology,
    evaluate_schedule_admission,
)
from ._capi import load_planner_library
from ._native_portfolio import (
    NativeContextResult,
    NativePreflightResult,
    decode_candidate_diagnostic,
    decode_schedule,
    evaluate_program_context_compiled,
    validate_program_context_compiled,
)
from ._recomputation import build_recomputation_portfolio
from .admission import AdmissionTopology
from .diagnostics import (
    PressureFitWorkDiagnostics,
    RecomputationChoiceDiagnostic,
    RecomputationContextDiagnostics,
)
from .model import (
    AdmissionRefinement,
    PressureFitDiagnostics,
    PressureFitInfeasibleError,
    PressureFitOptions,
    PressureFitResult,
    PressureFitSearchExhaustedError,
)

_ADMISSION_RESERVE_GRANULARITY_BYTES = 2 << 20
_ADMISSION_INITIAL_REFINEMENT_BYTES = 128 << 20
_ADMISSION_DOUBLING_LIMIT_BYTES = 1 << 30
_ADMISSION_LINEAR_REFINEMENT_BYTES = 512 << 20


def _scheduled_admission_refinement(attempt: int) -> int:
    """Return the deterministic reserve increment for one failed admission."""

    doubled = _ADMISSION_INITIAL_REFINEMENT_BYTES << attempt
    if doubled <= _ADMISSION_DOUBLING_LIMIT_BYTES:
        return doubled
    attempts_after_limit = attempt - 3
    return (
        _ADMISSION_DOUBLING_LIMIT_BYTES
        + attempts_after_limit * _ADMISSION_LINEAR_REFINEMENT_BYTES
    )


@dataclass(frozen=True, slots=True)
class _NativeSelectionContext:
    """One recomputation selection projected into the compiled planner ABI."""

    selections: tuple[RecomputationSelection, ...]
    selection_id: str
    compiled_template: CompiledSimulationTemplate
    compiled_admission: CompiledAdmissionTopology | None


def _selection_id(selections: tuple[RecomputationSelection, ...]) -> str:
    if not selections:
        return "none"
    return ",".join(f"{item.group_id}={item.option_id}" for item in selections)


def _validate_pressurefit_inputs(
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    admission: AdmissionTopology | None,
) -> None:
    """Validate public types and the logical/physical capacity relationship."""

    if not isinstance(program, Program):
        raise TypeError("program must be a Program")
    if not isinstance(initial_residency, tuple):
        raise TypeError("initial_residency must be a tuple")
    if not isinstance(final_residency, tuple):
        raise TypeError("final_residency must be a tuple")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if admission is None:
        return
    if not isinstance(admission, AdmissionTopology):
        raise TypeError("admission must be an AdmissionTopology")
    admission.validate(program)
    configured = {item.device_id: item for item in config.devices}
    if (
        admission.device_id not in configured
        or configured[admission.device_id].capacity_bytes
        != admission.object_capacity_bytes
    ):
        raise ValueError(
            "feasibility capacity must equal AdmissionTopology object capacity"
        )


def validate_schedule_feasibility(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    admission: AdmissionTopology | None = None,
) -> None:
    """Reject irreducible capacity failures using the compiled planner."""

    _validate_pressurefit_inputs(
        program,
        initial_residency,
        final_residency,
        config,
        admission,
    )
    load_simulator_library()
    load_planner_library()
    contexts = _build_native_contexts(
        program,
        initial_residency,
        final_residency,
        config,
        admission,
        portfolio=build_recomputation_portfolio(program),
        progress=None,
    )
    _preflight_native_contexts(contexts)


def _build_native_contexts(
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    admission: AdmissionTopology | None,
    *,
    portfolio: tuple[tuple[RecomputationSelection, ...], ...],
    progress: Callable[[str], None] | None,
) -> tuple[_NativeSelectionContext, ...]:
    """Project each recomputation selection without Python residency matrices."""

    contexts: list[_NativeSelectionContext] = []
    started = time.perf_counter_ns()
    for selection_index, selections in enumerate(portfolio, start=1):
        tasks = program.selected_tasks(selections)
        compiled_template = compile_simulation_template(
            program,
            selections,
            config,
            selected_tasks=tasks,
            initial_residency=initial_residency,
            final_residency=final_residency,
        )
        contexts.append(
            _NativeSelectionContext(
                selections=selections,
                selection_id=_selection_id(selections),
                compiled_template=compiled_template,
                compiled_admission=(
                    compile_admission_topology(admission, compiled_template)
                    if admission is not None
                    else None
                ),
            )
        )
        if progress is not None:
            progress(
                "PressureFit compiled context "
                f"{selection_index}/{len(portfolio)}: "
                f"tasks={len(tasks)}, aliases={len(program.alias_groups)}, "
                f"elapsed={(time.perf_counter_ns() - started) / 1e9:.3f}s"
            )
    return tuple(contexts)


def _preflight_error(
    context: _NativeSelectionContext,
    result: NativePreflightResult,
) -> ValueError:
    """Decode one compiled preflight failure into the public exception model."""

    if result.failure_kind == "missing_initial_residency":
        if result.error_alias is None:
            raise RuntimeError("compiled preflight omitted its failing alias")
        alias_id = context.compiled_template.alias_ids[result.error_alias]
        return ValueError(f"input alias {alias_id!r} has no initial residency")

    device_id = (
        None
        if result.error_device is None
        else context.compiled_template.device_ids[result.error_device]
    )
    boundary_task_id = (
        None
        if result.error_boundary is None
        or result.error_boundary < 0
        or result.error_boundary >= len(context.compiled_template.task_ids)
        else context.compiled_template.task_ids[result.error_boundary]
    )
    required = result.required_bytes
    capacity = result.capacity_bytes
    if result.failure_kind == "workspace_capacity":
        return PressureFitInfeasibleError(
            f"task workspace {required} exceeds capacity {capacity} on {device_id!r}",
            kind="workspace_capacity",
            device_id=device_id,
            boundary_task_id=boundary_task_id,
            required_bytes=required,
            capacity_bytes=capacity,
        )
    if result.failure_kind == "required_capacity":
        return PressureFitInfeasibleError(
            f"required inputs and outputs need {required} bytes at "
            f"{boundary_task_id or 'initialization'} on {device_id!r}, "
            f"exceeding object capacity {capacity}",
            kind="required_capacity",
            device_id=device_id,
            boundary_task_id=boundary_task_id,
            required_bytes=required,
            capacity_bytes=capacity,
        )
    raise RuntimeError(
        f"compiled preflight returned unknown failure {result.failure_kind!r}"
    )


def _preflight_native_contexts(
    contexts: tuple[_NativeSelectionContext, ...],
) -> tuple[_NativeSelectionContext, ...]:
    """Keep selections that satisfy the compiled semantic-capacity preflight."""

    valid: list[_NativeSelectionContext] = []
    failures: list[ValueError] = []
    for context in contexts:
        result = validate_program_context_compiled(
            context.compiled_template,
            admission=context.compiled_admission,
        )
        if result.valid:
            valid.append(context)
        else:
            failures.append(_preflight_error(context, result))
    if valid:
        return tuple(valid)
    if failures:
        raise failures[0]
    raise PressureFitInfeasibleError(
        "no recomputation selection could be constructed",
        kind="recomputation_selection",
    )


def _native_worker_count(options: PressureFitOptions, count: int) -> int:
    if count <= 1 or options.workers == 1:
        return 1
    if options.workers > 1:
        return min(options.workers, count)
    return min(max(os.cpu_count() or 1, 1), count)


def _run_native_contexts(
    contexts: tuple[_NativeSelectionContext, ...],
    options: PressureFitOptions,
) -> tuple[NativeContextResult | None, ...]:
    """Evaluate every recomputation selection in the compiled planner."""

    def evaluate(context: _NativeSelectionContext) -> NativeContextResult | None:
        return evaluate_program_context_compiled(
            context.compiled_template,
            options,
            admission=context.compiled_admission,
        )

    workers = _native_worker_count(options, len(contexts))
    if workers == 1:
        return tuple(evaluate(context) for context in contexts)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return tuple(executor.map(evaluate, contexts))


def _finish_native_pressurefit(
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    options: PressureFitOptions,
    contexts: tuple[_NativeSelectionContext, ...],
    results: tuple[NativeContextResult, ...],
    admission: AdmissionTopology | None,
) -> PressureFitResult:
    """Decode the globally best compiled result and its diagnostics."""

    recomputation_contexts: list[RecomputationContextDiagnostics] = []
    selected: tuple[int, int, NativeContextResult] | None = None
    for context_index, (context, result) in enumerate(
        zip(contexts, results, strict=True)
    ):
        candidates = tuple(
            decode_candidate_diagnostic(
                candidate,
                selection_id=context.selection_id,
                simulation=context.compiled_template,
            )
            for candidate in result.candidates
        )
        selected_candidate = (
            None
            if result.selected_candidate_index is None
            else candidates[result.selected_candidate_index]
        )
        recomputation_contexts.append(
            RecomputationContextDiagnostics(
                selection_id=context.selection_id,
                choices=tuple(
                    RecomputationChoiceDiagnostic(item.group_id, item.option_id)
                    for item in context.selections
                ),
                selected_candidate_id=(
                    None
                    if selected_candidate is None
                    else selected_candidate.candidate_id
                ),
                selected_makespan_ns=(
                    None
                    if selected_candidate is None
                    else selected_candidate.makespan_ns
                ),
                candidate_evaluations=candidates,
                work=result.work,
            )
        )
        if result.selected_candidate_index is None:
            continue
        assert result.selected_makespan_ns is not None
        candidate_ordinal = (
            context_index * len(result.candidates) + result.selected_candidate_index
        )
        key = (result.selected_makespan_ns, candidate_ordinal)
        if selected is None or key < (selected[0], selected[1]):
            selected = (key[0], key[1], result)

    if selected is None:
        frozen = tuple(
            candidate
            for context in recomputation_contexts
            for candidate in context.candidate_evaluations
        )
        if any(item.status == "exhausted" for item in frozen):
            raise PressureFitSearchExhaustedError(
                "PressureFit exhausted its bounded candidate-repair budget "
                "before proving a feasible schedule",
                diagnostics=frozen,
            )
        first = frozen[0] if frozen else None
        physical_slack = tuple(
            candidate.error_required_bytes
            for result in results
            for candidate in result.candidates
            if candidate.status == 3 and candidate.error_required_bytes > 0
        )
        raise PressureFitInfeasibleError(
            "no simulator-valid PressureFit candidate satisfied the declared "
            "capacity and residency constraints",
            kind=(
                first.failure_kind
                if first is not None and first.failure_kind
                else "no_candidate"
            ),
            required_bytes=min(physical_slack) if physical_slack else None,
            capacity_bytes=(
                config.devices[0].capacity_bytes if len(config.devices) == 1 else None
            ),
            diagnostics=frozen,
        )

    _makespan, ordinal, result = selected
    per_context = len(result.candidates)
    context_index, candidate_index = divmod(ordinal, per_context)
    context = contexts[context_index]
    indexed_schedule = result.selected_schedule
    assert indexed_schedule is not None
    schedule = decode_schedule(indexed_schedule, context.compiled_template)
    result_admission_calls = 0
    result_admission_time_ns = 0
    simulation_admission = None
    if context.compiled_admission is not None:
        admission_started = time.perf_counter_ns()
        simulation_admission = evaluate_schedule_admission(
            context.compiled_template,
            context.compiled_admission,
            indexed_schedule,
        ).simulation_admission
        result_admission_time_ns = time.perf_counter_ns() - admission_started
        result_admission_calls = 1
    simulation_started = time.perf_counter_ns()
    simulation = simulate_compiled_template(
        context.compiled_template,
        schedule,
        admission=simulation_admission,
    )
    result_simulation_time_ns = time.perf_counter_ns() - simulation_started
    selected_diagnostic = result.candidates[candidate_index]
    aggregate_work = PressureFitWorkDiagnostics()
    for context_result in results:
        aggregate_work += context_result.work
    aggregate_work += PressureFitWorkDiagnostics(
        result_simulation_calls=1,
        result_admission_calls=result_admission_calls,
        result_simulation_time_ns=result_simulation_time_ns,
        result_admission_time_ns=result_admission_time_ns,
    )
    diagnostics = PressureFitDiagnostics(
        selected_candidate_id=selected_diagnostic.candidate_id,
        selected_selection_id=context.selection_id,
        selected_makespan_ns=simulation.makespan_ns,
        recomputation_contexts=tuple(recomputation_contexts),
        work=aggregate_work,
    )
    return PressureFitResult(
        program=program,
        options=options,
        initial_residency=initial_residency,
        final_residency=final_residency,
        simulation_config=config,
        schedule=schedule,
        selections=context.selections,
        simulation=simulation,
        diagnostics=diagnostics,
        admission_topology=admission,
    )


def _pressurefit_once(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    options: PressureFitOptions | None = None,
    admission: AdmissionTopology | None = None,
    progress: Callable[[str], None] | None = None,
) -> PressureFitResult:
    """Evaluate every selection through the required compiled planner."""

    _validate_pressurefit_inputs(
        program,
        initial_residency,
        final_residency,
        config,
        admission,
    )
    load_simulator_library()
    load_planner_library()

    selected_options = options or PressureFitOptions()
    portfolio = build_recomputation_portfolio(program)
    if progress is not None:
        progress(
            "PressureFit portfolio: "
            f"groups={len(program.recomputation_groups)}, "
            f"selections={len(portfolio)}"
        )
    started = time.perf_counter_ns()
    contexts = _preflight_native_contexts(
        _build_native_contexts(
            program,
            initial_residency,
            final_residency,
            config,
            admission,
            portfolio=portfolio,
            progress=progress,
        )
    )
    native_results = _run_native_contexts(contexts, selected_options)
    valid_pairs = tuple(
        (context, result)
        for context, result in zip(contexts, native_results, strict=True)
        if result is not None
    )
    if progress is not None:
        progress(
            "PressureFit compiled contexts and candidates finished: "
            f"valid={len(valid_pairs)}/{len(contexts)}, "
            "candidates="
            f"{sum(len(result.candidates) for _context, result in valid_pairs)}, "
            f"workers={_native_worker_count(selected_options, len(contexts))}, "
            f"elapsed={(time.perf_counter_ns() - started) / 1e9:.3f}s"
        )
    if not valid_pairs:
        raise RuntimeError(
            "compiled PressureFit rejected every selection after semantic "
            "feasibility validation succeeded"
        )
    return _finish_native_pressurefit(
        program,
        initial_residency,
        final_residency,
        config,
        selected_options,
        tuple(context for context, _result in valid_pairs),
        tuple(result for _context, result in valid_pairs),
        admission,
    )


def _round_up_admission_reserve(value: int) -> int:
    granularity = _ADMISSION_RESERVE_GRANULARITY_BYTES
    return ((value + granularity - 1) // granularity) * granularity


def _with_object_capacity(
    config: SimulationConfig,
    admission: AdmissionTopology,
    capacity_bytes: int,
) -> tuple[SimulationConfig, AdmissionTopology]:
    devices = tuple(
        replace(device, capacity_bytes=capacity_bytes)
        if device.device_id == admission.device_id
        else device
        for device in config.devices
    )
    if devices == config.devices:
        raise ValueError(
            f"admission device {admission.device_id!r} is absent from simulation"
        )
    return (
        replace(config, devices=devices),
        replace(admission, object_capacity_bytes=capacity_bytes),
    )


def pressurefit(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    options: PressureFitOptions | None = None,
    admission: AdmissionTopology | None = None,
    progress: Callable[[str], None] | None = None,
) -> PressureFitResult:
    """Select a compiled schedule and refine dynamic-slab headroom."""

    original_config = config
    current_config = config
    current_admission = admission
    refinements: list[AdmissionRefinement] = []
    while True:
        try:
            result = _pressurefit_once(
                program,
                initial_residency=initial_residency,
                final_residency=final_residency,
                config=current_config,
                options=options,
                admission=current_admission,
                progress=progress,
            )
            effective_capacity = (
                None
                if current_admission is None
                else current_admission.object_capacity_bytes
            )
            return replace(
                result,
                simulation_config=original_config,
                diagnostics=replace(
                    result.diagnostics,
                    admission_refinements=tuple(refinements),
                    effective_object_capacity_bytes=effective_capacity,
                ),
                admission_topology=admission,
            )
        except PressureFitInfeasibleError as error:
            if (
                current_admission is None
                or error.kind != "physical_admission"
                or error.required_bytes is None
                or error.required_bytes <= 0
            ):
                raise
            previous = current_admission.object_capacity_bytes
            scheduled_increment = _scheduled_admission_refinement(len(refinements))
            increment = max(
                _round_up_admission_reserve(error.required_bytes),
                scheduled_increment,
            )
            capacity = previous - increment
            if capacity <= 0:
                raise
            refinements.append(
                AdmissionRefinement(
                    attempt=len(refinements) + 1,
                    previous_object_capacity_bytes=previous,
                    required_additional_slack_bytes=error.required_bytes,
                    reserve_increment_bytes=increment,
                    object_capacity_bytes=capacity,
                )
            )
            if progress is not None:
                progress(
                    "PressureFit physical admission refinement "
                    f"{len(refinements)}: required_slack={error.required_bytes}, "
                    f"reserve_increment={increment}, object_capacity={capacity}"
                )
            current_config, current_admission = _with_object_capacity(
                current_config,
                current_admission,
                capacity,
            )


__all__ = ["pressurefit", "validate_schedule_feasibility"]
