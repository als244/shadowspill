"""All four transfer rates survive planning, storage, and overrides."""

from dataclasses import replace

import pytest

from shadowspill.planner.plan_store import PlanStore
from shadowspill.planner.program_inputs import TransferBandwidths
from tests.shadowspill.planner.test_annotated_plan import _pressurefit_program
from tests.shadowspill.planner.test_plan_store import FEW_CANDIDATES

FIELDS = (
    "fetch_solo_bytes_per_second",
    "fetch_concurrent_bytes_per_second",
    "evict_solo_bytes_per_second",
    "evict_concurrent_bytes_per_second",
)


@pytest.mark.parametrize("field", FIELDS)
def test_every_rate_changes_identity_and_reaches_simulation(tmp_path, field):
    base = TransferBandwidths(
        40_000_000_000, 26_000_000_000, 56_000_000_000, 25_000_000_000
    )
    changed = replace(base, **{field: getattr(base, field) + 1000})
    assert changed.digest != base.digest
    assert TransferBandwidths.from_value(changed.to_dict()) == changed
    problem = _pressurefit_program()
    config, _ = problem.machine_inputs(transfer_bandwidths=base)
    updated, _ = problem.machine_inputs(transfer_bandwidths=changed)
    device_field = field.replace("_bytes_per_second", "_bandwidth_bytes_per_second")
    assert getattr(updated.devices[0], device_field) == getattr(changed, field)
    assert (
        replace(problem, simulation_config=config).digest
        != replace(problem, simulation_config=updated).digest
    )
    store = PlanStore(tmp_path)

    def resolve(machine):
        return store.resolve(
            problem.program,
            initial_residency=problem.initial_residency,
            final_residency=problem.final_residency,
            config=machine,
            search_options=FEW_CANDIDATES,
        )

    assert not resolve(config).from_store
    assert resolve(config).from_store
    assert not resolve(updated).from_store
    restored = resolve(updated)
    assert restored.from_store
    assert restored.result.simulation_config == updated
