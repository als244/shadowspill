"""The ShadowSpill backend's planning: searched on a run's first launch,
planned directly from the record on every later one.

ShadowSpill itself is stood in for here -- runtime, import, search and plan --
so the test sees only what the backend asks of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import training.backends.shadowspill as backend_module
from shadowspill.planner.program_inputs import TransferBandwidths
from tests.training._synthetic import TinyModel, write_dataset
from training.backends import Setup
from training.backends.shadowspill import PLANNING_RECORD, ShadowSpill
from training.data import PackedTokens
from training.objectives import Objective, model_loss

LANES = TransferBandwidths(
    fetch_bytes_per_second=13_000_000_000, evict_bytes_per_second=16_000_000_000
)


@dataclass
class Summary:
    simulated_step_seconds: float = 2.0
    unconstrained_step_seconds: float = 1.5

    def as_dict(self) -> dict[str, float]:
        return {"simulated_step_seconds": self.simulated_step_seconds}


class StandIns:
    """ShadowSpill's entry points, recording how the backend calls them."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.searches: list[dict[str, Any]] = []
        self.plans: list[dict[str, Any]] = []
        self.forwards: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.search_allowed = True
        runtime = SimpleNamespace(close=lambda: None)
        monkeypatch.setattr(backend_module, "_runtime", lambda *_: runtime)
        monkeypatch.setattr(backend_module, "import_model_state", self.import_model)
        monkeypatch.setattr(backend_module, "plan_step_search", self.search)
        monkeypatch.setattr(backend_module, "plan_step", self.plan)
        monkeypatch.setattr(backend_module, "plan_forward", self.plan_forward)
        monkeypatch.setattr(
            backend_module, "release_model_state", lambda *_, **__: None
        )

    @staticmethod
    def import_model(module: torch.nn.Module, **_: Any) -> torch.nn.Module:
        return module

    def search(self, module: torch.nn.Module, **arguments: Any) -> Any:
        assert self.search_allowed, "a launch with a planning record searched again"
        self.searches.append(arguments)
        examples = arguments["example_microbatches"](2, 3)
        assert len(examples) == 3 and examples[0][0].shape == (1, 1024)
        winner = SimpleNamespace(
            sequences_per_microbatch=2,
            accumulation_count=arguments["total_sequences_per_step"] // 2,
            ordering=SimpleNamespace(
                depth=3, breadth=2, reverse_breadth=False, pair_loss=True
            ),
        )
        return SimpleNamespace(
            winner=lambda *budget: winner,
            winner_plans={tuple(arguments["budgets"][0]): "the search's plan"},
            planned_lanes=LANES,
            save=lambda path: Path(path).write_text("{}") and path,
        )

    def plan(self, module: torch.nn.Module, **arguments: Any) -> Any:
        self.plans.append(arguments)
        self.calls.append("plan_step")
        return SimpleNamespace(
            plan_report=SimpleNamespace(summary=Summary()), close=lambda: None
        )

    def plan_forward(self, module: torch.nn.Module, **arguments: Any) -> Any:
        self.forwards.append(arguments)
        self.calls.append("plan_forward")
        return SimpleNamespace(close=lambda: None)


def _setup(tmp_path: Path) -> Setup:
    torch.set_default_device("meta")
    try:
        model = TinyModel()
    finally:
        torch.set_default_device(None)
    return Setup(
        module=Objective(model, model_loss, {}),
        optimizer=torch.optim.AdamW,
        optimizer_args={"lr": 0.01},
        data=PackedTokens(write_dataset(tmp_path / "tokens")),
        max_seq_len=512,
        max_tokens_per_step=4096,
        max_tokens_per_microbatch=None,
        hyperparams=("lr",),
        master_dtype=None,
        grad_dtype=None,
        seed=0,
        run_dir=tmp_path / "run",
        artifact_store=tmp_path / "run" / "artifact_store",
    )


def test_a_run_searches_once_and_later_launches_plan_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_ins = StandIns(monkeypatch)
    setup = _setup(tmp_path)
    setup.run_dir.mkdir()

    first = ShadowSpill(execution_gib=1, spill_gib=2)
    first.setup(setup)
    assert len(stand_ins.searches) == 1
    assert stand_ins.searches[0]["total_sequences_per_step"] == 8
    assert stand_ins.searches[0]["min_tokens_per_microbatch"] is None
    assert stand_ins.searches[0]["max_tokens_per_microbatch"] is None
    assert stand_ins.searches[0]["orderings"] is None
    assert first.geometry == (1024, 4)
    assert (setup.run_dir / "search.json").exists()
    assert first.plan.simulated_step_seconds == 2.0
    assert stand_ins.plans[0]["incumbent"] == "the search's plan"
    assert stand_ins.plans[0]["transfer_bandwidths"] == LANES
    assert (setup.run_dir / PLANNING_RECORD).exists()

    stand_ins.search_allowed = False
    later = ShadowSpill(execution_gib=1, spill_gib=2)
    later.setup(setup)
    assert later.geometry == first.geometry
    assert later.planning == first.planning
    replanned = stand_ins.plans[1]
    assert replanned["incumbent"] is None
    assert replanned["transfer_bandwidths"] == LANES
    assert (replanned["depth"], replanned["breadth"]) == (3, 2)
    assert replanned["example_inputs"][0][0].shape == (1, 1024)
    assert len(replanned["example_inputs"]) == 4


@pytest.mark.parametrize("eval_execution_gib, budget", [(None, None), (0.5, 1 << 29)])
def test_setup_plans_evaluation_right_after_the_step_into_its_slab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eval_execution_gib: float | None,
    budget: int | None,
) -> None:
    """A run that cannot evaluate stops at setup, before it trains. Without
    a budget of its own the forward plans within the step's: the whole slab."""

    stand_ins = StandIns(monkeypatch)
    setup = _setup(tmp_path)
    setup.run_dir.mkdir()
    backend = ShadowSpill(
        execution_gib=1, spill_gib=2, eval_execution_gib=eval_execution_gib
    )
    backend.setup(setup)

    assert stand_ins.calls == ["plan_step", "plan_forward"]
    (forward,) = stand_ins.forwards
    assert forward["share_slab_with"] is backend.train_step
    assert forward["execution_budget"] == budget
    assert forward["example_inputs"][0].shape == (1, 1024)
    assert backend.forward is not None


def test_a_run_refuses_other_budgets_than_it_was_planned_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    StandIns(monkeypatch)
    setup = _setup(tmp_path)
    setup.run_dir.mkdir()
    ShadowSpill(execution_gib=1, spill_gib=2).setup(setup)
    with pytest.raises(ValueError, match="planned at other budgets"):
        ShadowSpill(execution_gib=2, spill_gib=2).setup(setup)


@pytest.mark.parametrize("round_once", [False, True])
def test_a_run_plans_its_steps_with_the_accumulation_it_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, round_once: bool
) -> None:
    stand_ins = StandIns(monkeypatch)
    setup = _setup(tmp_path)
    setup.run_dir.mkdir()

    backend = ShadowSpill(execution_gib=1, spill_gib=2)
    if round_once:
        backend = ShadowSpill(
            execution_gib=1, spill_gib=2, round_accumulation_once=True
        )
    backend.setup(setup)

    assert stand_ins.searches[0]["round_accumulation_once"] is round_once
    assert stand_ins.plans[0]["round_accumulation_once"] is round_once


def _resolution(search_options: Any) -> tuple[str, ...]:
    return tuple(
        str(share) for share in search_options.algorithm.options.resolution_options
    )


def test_the_trainers_microbatch_pins_the_geometry_when_the_backend_names_no_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_ins = StandIns(monkeypatch)
    setup = _setup(tmp_path)
    setup.run_dir.mkdir()
    pinned = Setup(**{**setup.__dict__, "max_tokens_per_microbatch": 1024})
    ShadowSpill(execution_gib=1, spill_gib=2).setup(pinned)
    (search,) = stand_ins.searches
    assert search["min_tokens_per_microbatch"] == 1024
    assert search["max_tokens_per_microbatch"] == 1024


def test_planning_bounds_and_search_options_reach_both_phases_and_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backend's bounds are the search's, its options are the search's
    and the replan's alike, and the record keeps them so a later launch asks
    the store the same question."""

    stand_ins = StandIns(monkeypatch)
    setup = _setup(tmp_path)
    setup.run_dir.mkdir()
    backend = ShadowSpill(
        execution_gib=1,
        spill_gib=2,
        planning_min_tokens_per_microbatch=1024,
        planning_max_tokens_per_microbatch=2048,
        resolution_options="halves",
        orderings="depth-first",
    )
    backend.setup(setup)

    (search,) = stand_ins.searches
    assert search["min_tokens_per_microbatch"] == 1024
    assert search["max_tokens_per_microbatch"] == 2048
    assert _resolution(search["search_options"]) == ("0", "1/2", "1")
    walks = search["orderings"](4)
    assert [(walk.depth, walk.breadth) for walk in walks] == [(4, 1)]
    (plan,) = stand_ins.plans
    assert plan["search_options"] is search["search_options"]
    assert backend.planning.resolution_options == ["0", "1/2", "1"]
    assert backend.planning.orderings == "depth-first"
    assert backend.planning.planning_min_tokens_per_microbatch == 1024
    assert backend.planning.planning_max_tokens_per_microbatch == 2048

    stand_ins.search_allowed = False
    again = ShadowSpill(
        execution_gib=1,
        spill_gib=2,
        planning_min_tokens_per_microbatch=1024,
        planning_max_tokens_per_microbatch=2048,
        resolution_options=["0", "1/2", "1"],
        orderings="depth-first",
    )
    again.setup(setup)
    assert again.planning == backend.planning
    assert stand_ins.plans[1]["search_options"] is again.search_options
    with pytest.raises(ValueError, match="other search options"):
        ShadowSpill(execution_gib=1, spill_gib=2, resolution_options="quarters").setup(
            setup
        )


def test_a_record_from_before_the_options_were_kept_reads_as_the_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_ins = StandIns(monkeypatch)
    setup = _setup(tmp_path)
    setup.run_dir.mkdir()
    ShadowSpill(execution_gib=1, spill_gib=2).setup(setup)
    record = setup.run_dir / PLANNING_RECORD
    kept = json.loads(record.read_text())
    for name in (
        "resolution_options",
        "orderings",
        "planning_min_tokens_per_microbatch",
        "planning_max_tokens_per_microbatch",
    ):
        del kept[name]
    record.write_text(json.dumps(kept))

    stand_ins.search_allowed = False
    later = ShadowSpill(execution_gib=1, spill_gib=2)
    later.setup(setup)
    assert later.planning.resolution_options == ["0", "1/4", "1/2", "3/4", "1"]
    assert later.planning.orderings == "factors"


def test_the_backend_refuses_bounds_and_walks_it_cannot_plan() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        ShadowSpill(
            execution_gib=1,
            spill_gib=2,
            planning_min_tokens_per_microbatch=4096,
            planning_max_tokens_per_microbatch=2048,
        )
    with pytest.raises(ValueError, match="positive"):
        ShadowSpill(execution_gib=1, spill_gib=2, planning_min_tokens_per_microbatch=0)
    with pytest.raises(ValueError, match="orderings"):
        ShadowSpill(execution_gib=1, spill_gib=2, orderings="breadth-first")
    with pytest.raises(ValueError):
        ShadowSpill(execution_gib=1, spill_gib=2, resolution_options="x,1")
