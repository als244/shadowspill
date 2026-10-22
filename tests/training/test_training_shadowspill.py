"""Backend composition preserves candidates, state ownership and search policy."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

import shadowspill.training.backends.shadowspill as module
from shadowspill.planner import StepDataOrdering
from shadowspill.training import Forward, Trainer


@pytest.mark.parametrize("fixed", [False, True])
def test_search_and_forward_share_imported_state_and_build_store(
    monkeypatch, tmp_path, fixed
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
            assert forwards[0]["artifact_store"] == plans[0]["artifact_store"]
    assert len(released) == 1 and released[0] is imported[0]


def test_preparation_continues_after_one_microbatch_candidate_exhausts_device(
    monkeypatch,
):
    from shadowspill.search.report import StepSearchReport

    class RuntimeStub:
        pass

    backend = object.__new__(module.ShadowSpill)
    backend.runtime = RuntimeStub()
    backend.budget = (100, 200)
    backend.orderings = lambda count: [StepDataOrdering.depth_first(count)]
    backend.artifact_store = backend.search_options = None
    backend.fuse_narrow_gradient_accumulation = True
    backend._import = lambda model: model
    backend._planning_args = lambda: {"search_options": None}
    spec = SimpleNamespace(
        distributed=None,
        objective=None,
        optimizer=lambda model: None,
        hyperparams=(),
        master_dtype=None,
        grad_dtype=None,
        parameter_metrics=None,
    )
    artifact = object()

    def prepare(model, *, example_inputs, **kwargs):
        if example_inputs[0][0] == "too large":
            raise torch.OutOfMemoryError("candidate cannot be profiled")
        return artifact

    def search(candidates, **kwargs):
        assert candidates == {"fits": artifact}
        return StepSearchReport(tuple(kwargs["budgets"]), (), ())

    monkeypatch.setattr(module, "prepare_step", prepare)
    monkeypatch.setattr(module, "search_prepared_steps", search)
    bundle = backend.prepare_artifacts(
        torch.nn.Linear(1, 1),
        spec,
        {"large": [("too large",)], "fits": [("small",)]},
        budgets=[(80, 200), (100, 200)],
    )
    assert len(bundle.search.points) == 2
    assert all(
        p.candidate == "large" and p.status == "infeasible"
        for p in bundle.search.points
    )
