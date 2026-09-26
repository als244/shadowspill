"""The ShadowSpill backend's planning: searched on a run's first launch,
planned directly from the record on every later one.

ShadowSpill itself is stood in for here -- runtime, import, search and plan --
so the test sees only what the backend asks of it.
"""

from __future__ import annotations

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
        self.search_allowed = True
        runtime = SimpleNamespace(close=lambda: None)
        monkeypatch.setattr(backend_module, "_runtime", lambda *_: runtime)
        monkeypatch.setattr(backend_module, "import_model_state", self.import_model)
        monkeypatch.setattr(backend_module, "plan_step_search", self.search)
        monkeypatch.setattr(backend_module, "plan_step", self.plan)
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
        )

    def plan(self, module: torch.nn.Module, **arguments: Any) -> Any:
        self.plans.append(arguments)
        return SimpleNamespace(
            plan_report=SimpleNamespace(summary=Summary()), close=lambda: None
        )


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
    assert first.geometry == (1024, 4)
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


def test_a_run_refuses_other_budgets_than_it_was_planned_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    StandIns(monkeypatch)
    setup = _setup(tmp_path)
    setup.run_dir.mkdir()
    ShadowSpill(execution_gib=1, spill_gib=2).setup(setup)
    with pytest.raises(ValueError, match="planned at other budgets"):
        ShadowSpill(execution_gib=2, spill_gib=2).setup(setup)
