"""Spawned CPU searches preserve serial schedules and family admission."""

from dataclasses import replace

import pytest

from engine_tests.training.test_local_planning import measured  # noqa: F401
from engine_tests.training.test_setup_resolution import installed
from shadowspill.errors import PlanInfeasibleError
from shadowspill.training.preparation import bundle_validation, family
from shadowspill.training.preparation.bundle import TrainingBundle
from shadowspill.training.preparation.family import plan_family
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.parallel_search import parallel_search
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.shared_planning import representative_sources
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client import Memory


@pytest.mark.parametrize("measured", [(1, 8), (2, 16)], indirect=True)
def test_parallel_search_matches_serial_plans_and_joint_admission(measured):  # noqa: F811
    root, _, inputs, inventory = measured
    expected = plan_family(inventory, inputs.sources, inputs.communication)
    searched = parallel_search(root, inventory, max_processes=2)
    actual = plan_family(
        inventory, inputs.sources, inputs.communication, search_results=searched
    )
    assert actual.families.keys() == expected.families.keys()
    assert actual.rejected == expected.rejected
    assert actual.coverage == expected.coverage
    for rank in expected.plans:
        assert actual.plans[rank].keys() == expected.plans[rank].keys()
        for key, plan in expected.plans[rank].items():
            left, right = actual.plans[rank][key].to_dict(), plan.to_dict()
            # Timing, worker counts and equal-cost heuristic names may differ.
            # The executable schedule, physical layout and predicted time agree.
            left.pop("diagnostics")
            right.pop("diagnostics")
            assert left == right
    # Two counts, searched once each, instead of once per physical rank.
    assert len(list((root / "searched").glob("*/*.ssp"))) == 2
    for name, plan in actual.plans[0].items():
        assert all(plans[name] is plan for plans in actual.plans.values())
    with pytest.raises(ValueError, match="exactly the Plan requests"):
        plan_family(inventory, inputs.sources, inputs.communication, search_results={})


@pytest.mark.parametrize("measured", [(2, 16)], indirect=True)
def test_shared_plans_decode_and_validate_once_without_skipping_changed_rank(
    measured,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    _, _, inputs, inventory = measured
    planned, checked = [], []
    search, validate = family.plan_program, bundle_validation.validate_plan

    def counted_search(*args, **kwargs):
        planned.append(args[0].program)
        return search(*args, **kwargs)

    def counted_validation(*args):
        checked.append(args[1])
        return validate(*args)

    monkeypatch.setattr(family, "plan_program", counted_search)
    monkeypatch.setattr(bundle_validation, "validate_plan", counted_validation)
    prepared = plan_family(inventory, inputs.sources, inputs.communication)
    assert len(planned) == 2
    bundle = TrainingBundle.from_prepared(prepared, inputs.sources)
    assert checked == []  # Assembly does not repeat admission.
    path = tmp_path / "shared.ssp"
    bundle.save(path)
    loaded = TrainingBundle.load(path)
    for name, plan in loaded.plans[0].items():
        assert loaded.plans[1][name] is plan
    loaded.recheck(installed())
    assert len(checked) == 2
    name, plan = next(iter(loaded.plans[1].items()))
    changed = replace(
        plan,
        profiled_program=replace(
            plan.profiled_program, program=replace(plan.program, name="different graph")
        ),
    )
    damaged = replace(
        loaded, plans={0: loaded.plans[0], 1: {**loaded.plans[1], name: changed}}
    )
    with pytest.raises(ValueError, match="source operation"):
        damaged.recheck(installed())


@pytest.mark.parametrize("measured", [2], indirect=True)
@pytest.mark.parametrize("change", ["hardware", "loss", "computation", "geometry"])
def test_distinct_worker_requirements_are_not_shared(measured, change):  # noqa: F811
    _, _, inputs, _ = measured
    sources = dict(inputs.sources)
    assert len(set(representative_sources(sources).values())) == 1
    key = next(reversed(sources))
    source = sources[key]
    program = source.program
    from shadowspill.artifacts._values import plain

    if change == "hardware":
        program = replace(program, setup={**program.setup, "gpu": "different"})
    elif change in ("loss", "geometry"):
        bindings = plain(program.bindings)
        binding = bindings["preparation_template"]["binding"]
        if change == "loss":
            binding["contributes_loss"] = not binding["contributes_loss"]
        else:
            binding["inputs"]["extra_extent"] = 1
        program = replace(program, bindings=bindings)
    else:
        program = replace(program, executables={"input": b"different computation"})
    sources[key] = replace(source, program=program)
    assert len(set(representative_sources(sources).values())) == 2


@pytest.mark.parametrize("measured", [(2, 16)], indirect=True)
def test_parallel_search_includes_variants_of_distinct_hardware(measured):  # noqa: F811
    from engine_tests.training.test_allocation_coverage import certify

    root, _, inputs, inventory = measured
    sources = dict(inputs.sources)
    key = next(reversed(sources))
    source = sources[key]
    setup = {**source.program.setup, "gpu": "different worker hardware"}
    program = certify(inventory.templates[key], replace(source.program, setup=setup))
    sources[key] = replace(source, program=program, setup=setup)
    path = root / "calibrated" / f"{key}.ssp"
    path.unlink()
    sources[key].save(path)
    searched = parallel_search(root, inventory, sources=sources, max_processes=2)
    assert len(list((root / "searched").glob("*/*.ssp"))) == 4
    assert len({id(plan) for plan in searched.values()}) == 4
    actual = plan_family(
        inventory, sources, inputs.communication, search_results=searched
    )
    for name, plan in actual.plans[0].items():
        assert actual.plans[1][name] is not plan


@pytest.mark.parametrize("measured", [2], indirect=True)
def test_parallel_search_preserves_infeasible_outcomes(measured):  # noqa: F811
    root, _, inputs, inventory = measured
    config = replace(inventory.layouts.setup.config, memory=Memory(1 / (1 << 30), 1))
    inventory = capture_inventory(
        prepare_layouts(resolve_training(installed(), config=config))
    )
    searched = parallel_search(root, inventory, max_processes=2)
    assert all(isinstance(result, PlanInfeasibleError) for result in searched.values())
    with pytest.raises(PlanInfeasibleError):
        plan_family(
            inventory, inputs.sources, inputs.communication, search_results=searched
        )
