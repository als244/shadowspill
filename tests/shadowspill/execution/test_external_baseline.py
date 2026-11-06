"""Cold startup consumes the external profile component without in-pool costs."""

from types import SimpleNamespace

import pytest

from shadowspill import ProfiledProgram, apply_profiles
from shadowspill.execution.resources import worker_external_baseline
from shadowspill.memory import DevicePool, device
from tests.shadowspill.offline._example import profiles, program


def test_saved_profile_external_component_is_reserved_once_for_named_plans():
    data = profiles()
    data.fixed_device_bytes = 20
    data.metadata.update(
        external_device_bytes=8,
        retained_device_bytes=7,
        external_headroom_bytes=5,
    )
    original = apply_profiles(program(), data)
    restored = ProfiledProgram.from_dict(original.to_dict())
    first = SimpleNamespace(profiled_program=restored)
    second = SimpleNamespace(
        profiled_program=SimpleNamespace(
            metadata={
                "external_device_bytes": 11,
                "retained_device_bytes": 1000,
                "external_headroom_bytes": 20,
            }
        )
    )
    assert worker_external_baseline([first, first, second]) == 11
    assert worker_external_baseline([first]) == 8
    assert restored.fixed_device_bytes == 20
    assert (
        worker_external_baseline(
            [
                SimpleNamespace(profiled_program=SimpleNamespace(metadata={})),
                SimpleNamespace(
                    profiled_program=SimpleNamespace(
                        metadata={"external_device_bytes": None}
                    )
                ),
            ]
        )
        == 0
    )


@pytest.mark.parametrize("value", [-1, True, 1.5, 21])
def test_external_profile_component_must_be_valid_and_fit_overhead(value):
    data = profiles()
    data.metadata["external_device_bytes"] = value
    with pytest.raises(ValueError):
        apply_profiles(program(), data)


def test_external_profile_and_headroom_must_fit_together():
    data = profiles()
    data.metadata.update(external_device_bytes=16, external_headroom_bytes=5)
    with pytest.raises(ValueError, match="baseline and headroom"):
        apply_profiles(program(), data)


@pytest.mark.parametrize("value", [-1, True, 1.5, 901])
def test_pool_rejects_invalid_external_floor_before_bootstrap(value):
    with pytest.raises((TypeError, ValueError), match="external_baseline"):
        DevicePool(1000, external_headroom=100, external_baseline=value)


def test_pool_floor_does_not_change_headroom_or_enforcement_defaults():
    pool = device(physical_capacity=8 << 30, external_baseline=1 << 30)
    assert pool.external_baseline == 1 << 30
    assert pool.external_headroom == 512 << 20
    assert pool.reject_overbudget is False


def test_worker_passes_external_floor_to_pool_without_changing_plan(monkeypatch):
    import torch

    from shadowspill.execution import setup

    opened = []

    class Runtime:
        def __init__(self, **options):
            opened.append(options["pools"]["execution"])

    identity = {"torch": torch.__version__}
    monkeypatch.setattr(setup, "Runtime", Runtime)
    monkeypatch.setattr(setup, "PyTorchFrontend", lambda: None)
    monkeypatch.setattr(setup, "require_compiled_device", lambda *args: None)
    monkeypatch.setattr(setup, "build_setup", lambda *args: identity)
    plan = SimpleNamespace(
        pool_memory=None,
        simulation_only_reasons=(),
        program=SimpleNamespace(devices=("cuda_0",), setup=identity),
        memory_budgets=SimpleNamespace(execution_bytes=8 << 30, spill_bytes=64 << 20),
        communication_info=SimpleNamespace(metadata={}),
    )
    setup.open_worker(plan, 0, 512 << 20, external_baseline_bytes=900 << 20)
    assert opened[0].external_baseline == 900 << 20
    assert opened[0].physical_capacity == 8 << 30
