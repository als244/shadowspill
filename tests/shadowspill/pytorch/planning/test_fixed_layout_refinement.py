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
    FixedLayoutInfeasibleError,
    refinement,
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
        lambda config, _excluded: _selection(
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


def test_refinement_retries_in_256_mib_steps(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    capacity = 2 << 30
    calls: list[int] = []
    original = refinement.build_fixed_layout_admission  # type: ignore[attr-defined]

    def reject_full_capacity(selected, facts, **kwargs):  # type: ignore[no-untyped-def]
        # Rejecting every plan at the full capacity, so ruling candidates out
        # cannot rescue it and the ladder has to reduce capacity instead.
        calls.append(facts.object_capacity_bytes)
        if facts.object_capacity_bytes == capacity:
            raise FixedLayoutInfeasibleError(capacity + 1, capacity)
        return original(selected, facts, **kwargs)

    monkeypatch.setattr(
        refinement, "build_fixed_layout_admission", reject_full_capacity
    )

    selected = resolve_fixed_layout_selection(
        _config(capacity),
        _topology(capacity),
        lambda config, _excluded: _selection(config),
    )

    # The full capacity is tried twice: once as planned, once more with the
    # candidate that could not be placed ruled out. Only when that fails too
    # does the ladder reduce capacity -- the coarse rung accepts at
    # capacity - 256 MiB, and the fine approach bisects the rejected interval
    # back down to a 64-MiB reduction.
    assert calls == [
        capacity,
        capacity,
        capacity - (256 << 20),
        capacity - (128 << 20),
        capacity - (64 << 20),
    ]
    assert tuple(item.accepted for item in selected.attempts) == (
        False, False, True, True, True,
    )
    assert selected.capacity_reduction_bytes == 64 << 20


def test_refinement_switches_to_512_mib_steps_after_one_gib() -> None:
    capacity = 3 << 30

    reductions = refinement._capacity_reductions(capacity)

    assert reductions[:5] == tuple(index * (256 << 20) for index in range(5))
    assert reductions[5:8] == (1536 << 20, 2048 << 20, 2560 << 20)


def test_refinement_first_rung_runs_without_speculation() -> None:
    capacity = 2 << 30
    resolved: list[int] = []

    def resolve(config, _excluded=()):  # type: ignore[no-untyped-def]
        resolved.append(config.devices[0].capacity_bytes)
        return _selection(config)

    selected = resolve_fixed_layout_selection(
        _config(capacity),
        _topology(capacity),
        resolve,
    )

    # A point whose full-capacity plan admits must plan exactly once:
    # speculative rungs open only after the first rejection.
    assert resolved == [capacity]
    assert len(selected.attempts) == 1


def test_refinement_consumes_speculative_rungs_in_ladder_order(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    capacity = 2 << 30
    admitted: list[int] = []
    resolved: list[int] = []
    original = refinement.build_fixed_layout_admission  # type: ignore[attr-defined]

    def resolve(config, _excluded=()):  # type: ignore[no-untyped-def]
        resolved.append(config.devices[0].capacity_bytes)
        return _selection(config)

    step = 256 << 20

    def reject_top_three_rungs(selected, facts, **kwargs):  # type: ignore[no-untyped-def]
        # Rejecting by capacity rather than by call count, so that ruling a
        # candidate out at one capacity does not shift which rungs are
        # refused and the ladder order stays the thing under test.
        admitted.append(facts.object_capacity_bytes)
        if facts.object_capacity_bytes > capacity - 3 * step:
            raise FixedLayoutInfeasibleError(capacity + 1, capacity)
        return original(selected, facts, **kwargs)

    monkeypatch.setattr(
        refinement, "build_fixed_layout_admission", reject_top_three_rungs
    )

    selected = resolve_fixed_layout_selection(
        _config(capacity),
        _topology(capacity),
        resolve,
    )

    # Admission consumes rungs in strict ladder order regardless of the
    # speculative planning that runs them concurrently; the fine final
    # approach then bisects back from the accepted rung.
    # Each refused rung is tried twice -- once as planned, once more with the
    # candidate that could not be placed ruled out -- and only then does the
    # ladder step down. The fourth rung admits, and the fine approach probes
    # back toward it; those probes sit above the rejection threshold here, so
    # the coarse acceptance stands.
    fine = 64 << 20
    assert [(capacity - value) >> 20 for value in admitted] == [
        0, 0, 256, 256, 512, 512, 768, 640, 704
    ]
    assert step >> 20 == 256 and fine >> 20 == 64
    assert tuple(item.accepted for item in selected.attempts) == (
        False, False, False, False, False, False, True, False, False,
    )
    # Both fine probes sit above the rejection threshold, so the coarse
    # acceptance stands rather than being recovered toward.
    assert selected.capacity_reduction_bytes == 3 * step
    # Speculation planned ahead of the accepted rung.
    assert len(set(resolved)) > 4


def test_refinement_rejects_invalid_effective_capacity() -> None:
    capacity = 1 << 30
    invalid = replace(
        _selection(_config(capacity)).result.diagnostics,
        effective_object_capacity_bytes=capacity + 1,
    )

    def resolve(config, _excluded=()):  # type: ignore[no-untyped-def]
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
