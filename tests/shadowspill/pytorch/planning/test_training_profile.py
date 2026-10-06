from __future__ import annotations

from types import SimpleNamespace

import pytest

from shadowspill.pytorch.planning.training import profile as training_profile


def test_a_failed_profile_recovers_the_runtime_before_releasing_what_it_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device that ran out of memory while profiling latches the runtime,
    which then refuses to unregister anything, so the saved values profiling
    kept in the spill pool are released only after the recovery; released
    first, they outlive the plan and the runtime cannot close."""

    calls: list[str] = []

    class Profiler:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def release_host_memory(self) -> None:
            calls.append("release")

    def fail(*args: object, **kwargs: object) -> None:
        raise MemoryError("the device ran out while profiling")

    def recover(runtime: object, error: BaseException, **kwargs: object) -> None:
        assert isinstance(error, MemoryError)
        calls.append("recover")

    monkeypatch.setattr(training_profile, "TaskProfiler", Profiler)
    monkeypatch.setattr(training_profile, "existing_execution_reserve", lambda _: 0)
    monkeypatch.setattr(training_profile, "SavedValuePool", lambda *args: None)
    monkeypatch.setattr(training_profile, "_profile_training_tasks", fail)
    monkeypatch.setattr(training_profile, "prepare_failure_cleanup", recover)
    runtime = SimpleNamespace(pools={"spill": SimpleNamespace(pool_id=1)})
    materialized = SimpleNamespace(
        state=SimpleNamespace(
            runtime=runtime,
            bridge=SimpleNamespace(spill_pool_id=1, plan_handle=7),
        )
    )
    captured = SimpleNamespace(
        installed=SimpleNamespace(library=None, runtime_handle=None),
        device_ordinal=0,
    )

    with pytest.raises(MemoryError, match="ran out while profiling"):
        training_profile.profile_training_tasks(
            captured,  # type: ignore[arg-type]
            materialized,  # type: ignore[arg-type]
            plan_id=7,
            stores=None,  # type: ignore[arg-type]
            timer=None,  # type: ignore[arg-type]
        )
    assert calls == ["recover", "release"]
