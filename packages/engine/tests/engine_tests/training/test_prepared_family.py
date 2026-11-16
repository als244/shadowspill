"""Coverage cannot be lost to independently fast but jointly infeasible Plans."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from engine_tests.training.test_preparation_templates import inventory
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.parallelism import InputFieldPlacement, InputPlacement, Parallelism
from shadowspill.planning.shared import SharedMemory
from shadowspill.training.preparation import family_selection
from shadowspill.training.preparation.family import plan_family
from shadowspill.training.preparation.family_checks import _collectives
from shadowspill.training.preparation.family_identity import (
    execution_names,
    plan_identity,
)
from shadowspill.training.preparation.family_records import (
    PlannedCandidate,
    PlanRejection,
    WorkerPlans,
)
from shadowspill.training.preparation.family_selection import select_family
from shadowspill_client.protocol.operations import OperationKey


def test_prepared_workers_share_execution_names_without_merging_local_plans(
    monkeypatch,
):
    fields = inventory().layouts.setup.context.schema.fields
    prepared = inventory(
        parallelism=Parallelism.dp(
            num_replicas=2,
            inputs=InputPlacement.independent(
                2, fields=[InputFieldPlacement(tuple(f.path)) for f in fields]
            ),
        )
    )
    rows, plans = [], {rank: {} for rank in (0, 1)}
    for candidates in prepared.families.values():
        for source in candidates:
            workers = {}
            for rank, worker in source.workers.items():
                names = []
                for use in (worker.main, worker.masked, worker.optimizer):
                    name = None if use is None else plan_identity(use, None)
                    names.append(name)
                    if name is not None:
                        plans[rank].setdefault(
                            name, SimpleNamespace(rank=rank, use=use)
                        )
                workers[rank] = WorkerPlans(*names)
            rows.append(PlannedCandidate(str(len(rows)), source, None, workers, 1.0))
    assert not plans[0].keys() & plans[1].keys()
    monkeypatch.setattr(
        family_selection,
        "shared_memory",
        lambda *args: SharedMemory({}, {}, 0, 0, 0),
    )
    actual = select_family(prepared, rows, plans, (), max_family_attempts=10)
    assert actual.plans[0].keys() == actual.plans[1].keys()
    assert min(actual.plans[0]) == min(actual.plans[1])
    for candidates in actual.families.values():
        for row in candidates:
            assert row.workers[0] == row.workers[1]
            for rank, worker in row.workers.items():
                source = row.source.workers[rank]
                for name, use in zip(
                    (worker.main, worker.masked, worker.optimizer),
                    (source.main, source.masked, source.optimizer),
                    strict=True,
                ):
                    if name is not None:
                        assert (
                            actual.plans[rank][name]
                            is plans[rank][plan_identity(use, None)]
                        )
                        assert actual.plans[rank][name].rank == rank


def choices():
    prepared = inventory(operations={"step": ["metrics"], "forward": ["predictions"]})
    step = OperationKey("step", "metrics")
    forward = OperationKey("forward", "predictions", "evaluation")
    rows, plans = [], {0: {}}
    # Two independent coverage witnesses for step. The fastest is too expensive
    # to coexist with the required evaluation capability.
    for name, key, index, cost, retained in (
        ("fast", step, 3, 1.0, 9),
        ("small", step, 4, 2.0, 2),
        ("eval", forward, 4, 1.0, 4),
    ):
        source = prepared.families[key][index]
        plans[0][name] = SimpleNamespace(retained=retained)
        mask = None
        if source.workers[0].masked is not None:
            mask = name + ".mask"
            plans[0][mask] = SimpleNamespace(retained=0)
        rows.append(
            PlannedCandidate(
                name, source, None, {0: WorkerPlans(name, mask, None)}, cost
            )
        )
    return prepared, rows, plans


def bounded_shared(primary, additional):
    demand = primary.retained + sum(plan.retained for plan in additional.values())
    if demand > 10:
        raise PlanInfeasibleError(
            "joint retained bytes exceed 10", kind="shared_spill_memory"
        )
    return SharedMemory({}, {}, 0, demand, demand)


def test_joint_selection_keeps_every_capability_and_rejects_only_optional_alternatives(
    monkeypatch,
):
    prepared, rows, plans = choices()
    monkeypatch.setattr(family_selection, "shared_memory", bounded_shared)
    actual = select_family(prepared, rows, plans, [], max_family_attempts=10)
    assert actual.family_attempts == 2
    assert {row.id for family in actual.families.values() for row in family} == {
        "small",
        "eval",
    }
    expected = {
        name
        for row in rows[1:]
        for name in execution_names(row.source, row.ordering).members()
    }
    assert actual.plans[0].keys() == expected
    assert actual.memory[0].spill_bytes == 6
    assert len(actual.coverage) == 2
    for key, proof in actual.coverage.items():
        assert proof.schedule in [
            row.source.layout.candidate.schedule for row in actual.families[key[0]]
        ]
    assert actual.rejected[0].candidate == "fast"
    assert actual.rejected[0].stage == "joint_memory"
    with pytest.raises(TypeError):
        actual.plans[0]["unadmitted"] = plans[0]["fast"]


def test_family_search_limits_and_unknown_candidate_results_are_not_infeasibility(
    monkeypatch,
):
    prepared, rows, plans = choices()
    monkeypatch.setattr(family_selection, "shared_memory", bounded_shared)
    with pytest.raises(PlanSearchExhaustedError, match="1 complete families"):
        select_family(prepared, rows, plans, [], max_family_attempts=1)
    with pytest.raises(PlanInfeasibleError, match="no complete coverage witness"):
        select_family(prepared, [rows[0], rows[2]], plans, [], max_family_attempts=10)
    unknown = PlanRejection(
        "small", rows[1].operation, 0, 2, 1, None, "planning", "bounded stop", True
    )
    with pytest.raises(PlanSearchExhaustedError, match="some searches exhausted"):
        select_family(
            prepared, [rows[0], rows[2]], plans, [unknown], max_family_attempts=10
        )


def test_missing_capability_and_unexpected_native_errors_are_never_silently_pruned(
    monkeypatch,
):
    prepared, rows, plans = choices()
    with pytest.raises(
        PlanInfeasibleError, match="no admitted coverage witness for forward"
    ):
        select_family(prepared, rows[:2], plans, [], max_family_attempts=10)
    failure = ValueError("state geometry changed")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(family_selection, "shared_memory", fail)
    with pytest.raises(ValueError) as caught:
        select_family(prepared, rows, plans, [], max_family_attempts=10)
    assert caught.value is failure


def test_exact_capture_inventory_is_required_before_any_planning():
    prepared, _, _ = choices()
    with pytest.raises(ValueError, match="exactly the capture inventory"):
        plan_family(prepared, {}, {})
    with pytest.raises(ValueError, match="max_family_attempts"):
        plan_family(prepared, {}, {}, max_family_attempts=0)


def test_family_planner_import_has_no_framework_or_catalog_dependency():
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads'):
    sys.modules[name] = None
from shadowspill.training.preparation.family import plan_family
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.templates import capture_inventory
setup = resolve_training(installed(), **arguments())
inventory = capture_inventory(prepare_layouts(setup))
try:
    plan_family(inventory, {}, {})
except ValueError as error:
    assert 'exactly the capture inventory' in str(error)
else:
    raise AssertionError('missing profiles accepted')
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_family_collective_order_uses_the_same_saved_worker_comparison():
    from engine_tests.execution.test_collective_sequence import plan

    a, b = plan(["embed", "head"]), plan(["head", "embed"])
    a._accumulation_plan = b._accumulation_plan = None
    workers = {rank: WorkerPlans("main", None, None) for rank in (0, 1)}
    _collectives(workers, {0: {"main": a}, 1: {"main": a}})
    with pytest.raises(PlanInfeasibleError, match="collective 1") as caught:
        _collectives(workers, {0: {"main": a}, 1: {"main": b}})
    assert caught.value.kind == "prepared_collective_order"
    # The additive path must agree too, even when fresh paths match.
    a._accumulation_plan = b
    c = plan(["embed", "head"])
    c._accumulation_plan = plan(["embed", "head"])
    with pytest.raises(PlanInfeasibleError, match="collective 1"):
        _collectives(workers, {0: {"main": a}, 1: {"main": c}})


def test_opaque_provider_collectives_require_matching_finalized_graphs():
    from shadowspill.parallelism import GroupRef
    from shadowspill.resources.description import Resource

    resource = Resource(
        "external.provider:create", groups={"workers": GroupRef("workers", ((0, 1),))}
    )

    def opaque(code):
        return SimpleNamespace(
            program=SimpleNamespace(
                bindings={},
                objects=[],
                resources={"provider": resource},
                task_definitions=[
                    SimpleNamespace(profile_key="opaque", compatibility_digest=code)
                ],
                tasks=[SimpleNamespace(task_id="call", profile_id="opaque")],
            ),
            _costed_program=SimpleNamespace(
                alias_groups=[],
                selected_tasks=lambda _: [
                    SimpleNamespace(
                        task_id="call",
                        profile_id="opaque.workspace_16",
                        phase="forward",
                    )
                ],
            ),
            schedule=SimpleNamespace(actions=[]),
            selections=(),
            _accumulation_plan=None,
        )

    workers = {rank: WorkerPlans("main", None, None) for rank in (0, 1)}
    _collectives(workers, {0: {"main": opaque("same")}, 1: {"main": opaque("same")}})
    with pytest.raises(ValueError, match="opaque provider collective graphs differ"):
        _collectives(workers, {0: {"main": opaque("a")}, 1: {"main": opaque("b")}})
