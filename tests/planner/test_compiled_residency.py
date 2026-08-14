from __future__ import annotations

import pytest

from shadowspill.planner._capi import planner_library_path
from shadowspill.planner._dense_residency import (
    compile_residency_template,
    reduce_residency_compiled,
)
from shadowspill.planner._facts import build_facts
from shadowspill.planner._residency import reduce_pressure, seed_residency
from shadowspill.planner.model import (
    InitialPlacement,
    PressureFitInfeasibleError,
)

from ._examples import (
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)

pytestmark = pytest.mark.skipif(
    planner_library_path() is None,
    reason="compiled planner library is not installed",
)


@pytest.mark.parametrize("layers", [1, 2, 3, 10])
@pytest.mark.parametrize("capacity", [96, 160, 224, 320, 512, 1_024])
@pytest.mark.parametrize(
    "strategy",
    ["tight-stall", "tight-transfer", "headroom-stall", "headroom-transfer"],
)
@pytest.mark.parametrize("with_repair_pressure", [False, True])
def test_compiled_reducer_matches_python_reference(
    layers: int,
    capacity: int,
    strategy: str,
    *,
    with_repair_pressure: bool,
) -> None:
    program = training_chain_program(layers)
    selected_config = training_chain_config(capacity)
    try:
        facts = build_facts(
            program,
            (),
            training_chain_initial(layers),
            (),
            selected_config,
        )
    except PressureFitInfeasibleError:
        pytest.skip("workspace alone exceeds this fixture capacity")
    seed = seed_residency(
        facts,
        selected_config,
        InitialPlacement.GREEDY,
        initial_capacity_by_device=facts.object_capacity_by_device,
    )
    extra = {("cuda_0", max(0, layers - 1)): 33} if with_repair_pressure else None

    def reduce_python() -> object:
        return reduce_pressure(
            facts,
            selected_config,
            seed,
            strategy,
            extra_pressure=extra,
        )

    def reduce_compiled() -> object:
        return reduce_residency_compiled(
            compile_residency_template(facts, selected_config, seed),
            seed,
            strategy,
            extra_pressure=extra,
        )

    try:
        reference = reduce_python()
    except PressureFitInfeasibleError as reference_error:
        with pytest.raises(PressureFitInfeasibleError) as compiled:
            reduce_compiled()
        assert compiled.value.kind == reference_error.kind
        assert compiled.value.device_id == reference_error.device_id
        assert compiled.value.boundary_task_id == reference_error.boundary_task_id
        assert compiled.value.required_bytes == reference_error.required_bytes
        assert compiled.value.capacity_bytes == reference_error.capacity_bytes
    else:
        assert reduce_compiled() == reference
