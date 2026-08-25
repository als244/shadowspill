from __future__ import annotations

from dataclasses import replace

from shadowspill.ir import MemorySchedule, Program, TaskProfile, TaskSpec
from shadowspill.planner import (
    AdmissionFacts,
    CandidateDiagnostic,
    PressureFitDiagnostics,
    PressureFitOptions,
    PressureFitResult,
    RecomputationProblemDiagnostics,
    TaskAdmissionSpec,
)
from shadowspill.planner.cache import CachedPressureFitResult
from shadowspill.pytorch.planning.admission import (
    resolve_fixed_layout_selection,
)
from shadowspill.simulator import SimulationConfig, simulate
from tests.shadowspill.planner._examples import COMPUTE, DEVICE


def _selection(
    config: SimulationConfig,
    *,
    effective_capacity: int | None = None,
) -> CachedPressureFitResult:
    program = Program(
        devices=(DEVICE,),
        alias_groups=(),
        objects=(),
        profiles=(TaskProfile("profile", 10, 0, "abi"),),
        tasks=(TaskSpec("task", COMPUTE, "profile"),),
    )
    schedule = MemorySchedule((), (), ())
    simulation = simulate(program, schedule, config=config)
    return CachedPressureFitResult(
        PressureFitResult(
            program,
            PressureFitOptions(workers=1),
            (),
            (),
            config,
            schedule,
            (),
            simulation,
            PressureFitDiagnostics(
                selected_candidate_id="candidate",
                selected_selection_id="selection",
                selected_makespan_ns=simulation.makespan_ns,
                recomputation_problems=(
                    RecomputationProblemDiagnostics(
                        selection_id="selection",
                        choices=(),
                        selected_candidate_id="candidate",
                        selected_makespan_ns=simulation.makespan_ns,
                        candidate_evaluations=(
                            CandidateDiagnostic(
                                candidate_id="candidate",
                                selection_id="selection",
                                status="valid",
                                makespan_ns=simulation.makespan_ns,
                            ),
                        ),
                    ),
                ),
                effective_object_capacity_bytes=effective_capacity,
            ),
        ),
        False,
    )


def _config(capacity: int) -> SimulationConfig:
    return SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=capacity,
        spill_capacity_bytes=capacity,
        fetch_bandwidth_bytes_per_second=1,
        evict_bandwidth_bytes_per_second=1,
    )


def _topology(capacity: int) -> AdmissionFacts:
    return AdmissionFacts(
        "cuda_0",
        capacity,
        capacity,
        1,
        (TaskAdmissionSpec("task"),),
    )


def test_refinement_uses_pressurefit_effective_capacity() -> None:
    capacity = 2 << 30
    effective = capacity - (384 << 20)

    selected = resolve_fixed_layout_selection(
        _config(capacity),
        _topology(capacity),
        lambda config: _selection(
            config,
            effective_capacity=effective,
        ),
    )

    assert selected.original_object_capacity_bytes == capacity
    assert selected.facts.object_capacity_bytes == effective
    assert selected.capacity_reduction_bytes == 384 << 20
    assert len(selected.attempts) == 1
    attempt = selected.attempts[0]
    assert (
        attempt.requested_object_capacity_bytes,
        attempt.effective_object_capacity_bytes,
        attempt.required_bytes,
        attempt.pool_capacity_bytes,
        attempt.accepted,
    ) == (capacity, effective, 0, capacity, True)
    assert attempt.pressurefit_wall_time_ns > 0
    assert attempt.physical_admission_wall_time_ns > 0


def test_refinement_plans_once_at_full_capacity() -> None:
    capacity = 2 << 30
    resolved: list[int] = []

    def resolve(config):  # type: ignore[no-untyped-def]
        resolved.append(config.devices[0].capacity_bytes)
        return _selection(config)

    selected = resolve_fixed_layout_selection(
        _config(capacity),
        _topology(capacity),
        resolve,
    )

    # The search answers with a plan it has already measured against this
    # pool, so this layer certifies one capacity and never walks down.
    assert resolved == [capacity]
    assert len(selected.attempts) == 1


def test_refinement_rejects_invalid_effective_capacity() -> None:
    capacity = 1 << 30
    invalid = replace(
        _selection(_config(capacity)).result.diagnostics,
        effective_object_capacity_bytes=capacity + 1,
    )

    def resolve(config):  # type: ignore[no-untyped-def]
        base = _selection(config)
        return replace(base, result=replace(base.result, diagnostics=invalid))

    try:
        resolve_fixed_layout_selection(
            _config(capacity),
            _topology(capacity),
            resolve,
        )
    except ValueError as error:
        assert "invalid effective object capacity" in str(error)
    else:
        raise AssertionError("invalid effective capacity was accepted")
