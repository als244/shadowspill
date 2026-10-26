"""Backend composition preserves candidates, state ownership and search policy."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

import shadowspill.training.backends.shadowspill as module
from shadowspill.planner import StepDataOrdering
from shadowspill.training import Forward
from shadowspill.training._legacy_trainer import Trainer


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("variants", [None, ("recompute",), ("save",)])
def test_search_and_forward_share_imported_state_and_build_store(
    monkeypatch, tmp_path, fixed, variants
):
    imported, released, builds, searches, plans, forwards = [], [], [], [], [], []

    class RuntimeStub:
        def close(self):
            pass

    runtime = RuntimeStub()
    monkeypatch.setattr(module, "resolve_device", lambda _: torch.device("cuda:0"))
    monkeypatch.setattr(module, "Runtime", lambda **_: runtime)

    def import_model(model, **kwargs):
        result = copy.deepcopy(model)
        if initialize := kwargs.get("initialize"):
            initialize(result)
        imported.append(result)
        return result

    monkeypatch.setattr(module, "import_model_state", import_model)
    monkeypatch.setattr(
        module, "release_model_state", lambda model, **_: released.append(model)
    )

    prepared_step = object()

    def prepare(model, **kwargs):
        builds.append(kwargs)
        return prepared_step

    monkeypatch.setattr(module, "prepare_step", prepare)

    def search(candidates, **kwargs):
        kwargs["candidates"] = candidates
        searches.append(kwargs)
        winner = SimpleNamespace(
            candidate="split", ordering=StepDataOrdering.depth_first(2)
        )
        return SimpleNamespace(
            winner=lambda *_: winner,
            winner_plans={kwargs["budgets"][0]: "chosen"},
            planned_lanes=None,
        )

    monkeypatch.setattr(module, "search_prepared_steps", search)

    def plan(model, prepared, selected, **kwargs):
        assert prepared is prepared_step
        assert selected == "chosen"
        plans.append(kwargs)
        return SimpleNamespace(plan_report=None, close=lambda: None)

    monkeypatch.setattr(module, "admit_step", plan)
    monkeypatch.setattr(
        module,
        "plan_step",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("Trainer must admit the selected prepared plan")
        ),
    )

    def forward(model, **kwargs):
        forwards.append(kwargs)
        return SimpleNamespace(plan_report=None, close=lambda: None)

    monkeypatch.setattr(module, "plan_forward", forward)

    def split(x):
        yield x[:2], 0.5
        yield x[2:], 0.5

    model = torch.nn.Linear(3, 1)
    policy = object()
    with (
        module.ShadowSpill(
            execution_gib=2,
            spill_gib=3,
            device="cuda:0",
            search_options=policy,
            **({} if variants is None else {"graph_pair_variants": variants}),
            orderings=[StepDataOrdering.depth_first(2)] if fixed else None,
        ) as backend,
        Trainer(
            model,
            objective=lambda m, x: m(x).square().mean(),
            optimizer=torch.optim.SGD,
            optimizer_args={"lr": 0.1},
            microbatch_candidates={"split": split},
            backend=backend,
        ) as trainer,
    ):
        trainer.prepare(torch.ones(4, 3))
        assert len(imported) == 1
        assert trainer.model is imported[0]
        assert trainer.model is not model
        assert tuple(searches[0]["candidates"]) == ("split",)
        # Searched or not, a step builds and plans in the backend's store, the
        # library's own when it names none.
        assert searches[0]["artifact_store"] is None
        assert len(plans[0]["example_inputs"]) == 2
        assert len(builds) == 1
        expected_variants = variants or ("save", "recompute")
        assert builds[0]["graph_pair_variants"] == expected_variants
        assert plans[0]["graph_pair_variants"] == expected_variants
        if fixed:
            assert builds[0]["orderings"] == (StepDataOrdering.depth_first(2),)
            with pytest.raises(ValueError, match="every microbatch"):
                backend.orderings(3)
        assert builds[0]["example_inputs"] == plans[0]["example_inputs"]
        assert "objective" not in plans[0]
        assert plans[0]["search_options"] is searches[0]["search_options"] is policy
        with Forward(trainer.model, backend=backend) as runner:
            runner.prepare(torch.ones(2, 3))
            assert len(imported) == 1
            assert forwards[0]["share_slab_with"] is trainer._execution.call
            assert "execution_budget" not in forwards[0]
            assert "graph_pair_variants" not in forwards[0]
            assert forwards[0]["artifact_store"] == plans[0]["artifact_store"]
    assert len(released) == 1 and released[0] is imported[0]


@pytest.mark.parametrize("microbatches", [1, 2])
def test_distributed_build_and_search_use_requested_variants(monkeypatch, microbatches):
    monkeypatch.setattr(module, "resolve_device", lambda _: torch.device("cuda:0"))
    backend = module.ShadowSpill(
        execution_gib=2, spill_gib=3, graph_pair_variants=("recompute",)
    )
    backend._import = lambda model, distributed: model
    model = torch.nn.Linear(3, 1)
    spec = SimpleNamespace(
        distributed=object(),
        shard_optimizer=True,
        objective=object(),
        optimizer=lambda model: object(),
        hyperparams=(),
        master_dtype=None,
        grad_dtype=None,
        parameter_metrics=None,
    )
    searches, plans = [], []

    def search(model, **kwargs):
        searches.append(kwargs)
        winner = SimpleNamespace(
            candidate="batch", ordering=StepDataOrdering.depth_first(microbatches)
        )
        return SimpleNamespace(
            winner=lambda *_: winner,
            winner_plans={backend.budget: "selected"},
            planned_lanes=None,
        )

    def plan(model, **kwargs):
        plans.append(kwargs)
        return SimpleNamespace(plan_report=None)

    monkeypatch.setattr(module, "plan_step_search", search)
    monkeypatch.setattr(module, "plan_step", plan)
    _, chosen = backend.prepare_step(
        model, spec, {"batch": [(torch.ones(1, 3),)] * microbatches}
    )
    assert chosen == "batch"
    assert len(searches) == int(microbatches > 1)
    assert len(plans) == 1
    for request in [*searches, *plans]:
        assert request["graph_pair_variants"] == ("recompute",)
