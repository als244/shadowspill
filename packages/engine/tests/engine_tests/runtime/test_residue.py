"""A closing plan reads its residue from the device pool, whatever it is called."""

from types import SimpleNamespace

import pytest

from shadowspill.runtime import residue


def test_residue_is_read_from_the_device_pool_under_the_callers_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The README names its device pool "device"; nothing may assume "execution".
    runtime = SimpleNamespace(
        pools={
            "spill": SimpleNamespace(kind="pinned_host"),
            "device": SimpleNamespace(kind="device"),
        }
    )
    asked: list[str] = []

    def live_allocations(runtime: object, pool: str) -> tuple[()]:
        asked.append(pool)
        return ()

    monkeypatch.setattr(residue, "live_allocations", live_allocations)
    monkeypatch.setattr(
        residue,
        "runtime_library",
        lambda: SimpleNamespace(shadowspill_plan_id=lambda handle: 7),
    )

    assert residue.plan_scoped_residue(runtime, plan_handle=1) == ()  # type: ignore[arg-type]
    assert asked == ["device"]
