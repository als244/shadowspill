from __future__ import annotations

from dataclasses import replace

from shadowspill.ir import MemorySchedule, Program, TaskProfile, TaskSpec
from shadowspill.planner import (
    AdmissionTopology,
    PressureFitDiagnostics,
    PressureFitOptions,
    PressureFitResult,
    TaskAdmissionSpec,
)
from shadowspill.planner._cache import CachedPressureFitResult
from shadowspill.pytorch.planning.admission import (
    FixedLayoutInfeasibleError,
    refinement,
    resolve_fixed_layout_selection,
)
from shadowspill.simulator import SimulationConfig, simulate
from tests.planner._examples import COMPUTE, DEVICE


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
                "candidate",
                "selection",
                1,
                1,
                simulation.makespan_ns,
                (),
                effective_object_capacity_bytes=effective_capacity,
            ),
        ),
        False,
    )


def _config(capacity: int) -> SimulationConfig:
    return SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=capacity,
        host_capacity_bytes=capacity,
        fetch_bandwidth_bytes_per_second=1,
        evict_bandwidth_bytes_per_second=1,
    )


def _topology(capacity: int) -> AdmissionTopology:
    return AdmissionTopology(
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
        lambda config, _topology: _selection(
            config,
            effective_capacity=effective,
        ),
    )

    assert selected.original_object_capacity_bytes == capacity
    assert selected.topology.object_capacity_bytes == effective
    assert selected.capacity_reduction_bytes == 384 << 20
    assert selected.attempts == (
        refinement.FixedLayoutAttempt(
            capacity,
            effective,
            0,
            capacity,
            True,
        ),
    )


def test_refinement_retries_in_128_mib_steps(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    capacity = 2 << 30
    calls: list[int] = []
    original = refinement.build_fixed_layout_admission  # type: ignore[attr-defined]

    def reject_first(selected, topology, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(topology.object_capacity_bytes)
        if len(calls) == 1:
            raise FixedLayoutInfeasibleError(capacity + 1, capacity)
        return original(selected, topology, **kwargs)

    monkeypatch.setattr(refinement, "build_fixed_layout_admission", reject_first)

    selected = resolve_fixed_layout_selection(
        _config(capacity),
        _topology(capacity),
        lambda config, _topology: _selection(config),
    )

    assert calls == [capacity, capacity - (128 << 20)]
    assert tuple(item.accepted for item in selected.attempts) == (False, True)
    assert selected.capacity_reduction_bytes == 128 << 20


def test_refinement_switches_to_512_mib_steps_after_one_gib() -> None:
    capacity = 3 << 30

    reductions = refinement._capacity_reductions(capacity)

    assert reductions[:9] == tuple(index * (128 << 20) for index in range(9))
    assert reductions[9:12] == (1536 << 20, 2048 << 20, 2560 << 20)


def test_refinement_rejects_invalid_effective_capacity() -> None:
    capacity = 1 << 30
    invalid = replace(
        _selection(_config(capacity)).result.diagnostics,
        effective_object_capacity_bytes=capacity + 1,
    )

    def resolve(config, _topology):  # type: ignore[no-untyped-def]
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
