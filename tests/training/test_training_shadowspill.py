"""Backend composition preserves candidates, state ownership and search policy."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

import shadowspill.training.backends.shadowspill as module
from shadowspill.planner import StepDataOrdering
from shadowspill.training import Forward, Trainer


def test_search_and_forward_share_imported_state_and_build_store(monkeypatch, tmp_path):
    imported, released, searches, plans, forwards = [], [], [], [], []

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

    def search(model, **kwargs):
        searches.append(kwargs)
        winner = SimpleNamespace(
            candidate="split", ordering=StepDataOrdering.depth_first(2)
        )
        return SimpleNamespace(
            winner=lambda *_: winner,
            winner_plans={kwargs["budgets"][0]: "chosen"},
            planned_lanes=None,
        )

    monkeypatch.setattr(module, "plan_step_search", search)

    def plan(model, **kwargs):
        plans.append(kwargs)
        return SimpleNamespace(plan_report=None, close=lambda: None)

    monkeypatch.setattr(module, "plan_step", plan)

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
        ) as backend,
        Trainer(
            model,
            objective=lambda m, x: m(x).square().mean(),
            optimizer=torch.optim.SGD,
            optimizer_args={"lr": 0.1},
            microbatches={"split": split},
            backend=backend,
        ) as trainer,
    ):
        trainer.prepare(torch.ones(4, 3))
        assert len(imported) == 1
        assert trainer.model is imported[0]
        assert trainer.model is not model
        assert tuple(searches[0]["candidates"]) == ("split",)
        assert len(plans[0]["example_inputs"]) == 2
        assert plans[0]["incumbent"] == "chosen"
        assert plans[0]["search_options"] is searches[0]["search_options"] is policy
        with Forward(trainer.model, backend=backend) as runner:
            runner.prepare(torch.ones(2, 3))
            assert len(imported) == 1
            assert forwards[0]["share_slab_with"] is trainer._execution.call
            assert "execution_budget" not in forwards[0]
            assert forwards[0]["artifact_store"] == plans[0]["artifact_store"]
    assert len(released) == 1 and released[0] is imported[0]
