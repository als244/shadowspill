"""The spill each Plan leaves for persistent state its run's other operations keep."""

from types import SimpleNamespace

from shadowspill.artifacts.state import StateId
from shadowspill.training.preparation.planning import reservations as module
from shadowspill.training.preparation.planning.reservations import spill_reservations

WEIGHT = StateId("parameter", "weight")
MASTER = StateId("master", "weight")
MOMENT = StateId("optimizer", "weight.exp_avg")
GRADIENT = StateId("gradient", "weight")


def test_each_plan_leaves_room_for_what_the_others_keep(monkeypatch):
    held = {
        "step": ({WEIGHT: 10, MASTER: 20, MOMENT: 20}, 0),
        "backward": ({WEIGHT: 10, GRADIENT: 5}, 0),
        "update": ({WEIGHT: 10, GRADIENT: 5, MASTER: 20, MOMENT: 20}, 0),
        "eval-small": ({WEIGHT: 10}, 3),
        "eval-large": ({WEIGHT: 10}, 7),
    }
    monkeypatch.setattr(
        module,
        "_persistent",
        lambda program, rank: module._Persistent(
            held[program][0],
            held[program][1],
            frozenset(key for key in held[program][0] if key.role == "gradient"),
        ),
    )
    kinds = {"step": "step", "backward": "forward_backward", "update": "optim_step"}
    requests = {
        key: SimpleNamespace(
            rank=0, use=SimpleNamespace(kind=kinds.get(key, "forward"))
        )
        for key in held
    }
    candidates = {
        "step": ("step",),
        "split": ("backward", "update"),
        "small": ("eval-small",),
        "large": ("eval-large",),
    }
    descriptions = {
        "step": {"operation": "step", "batch": 0},
        "split": {"operation": "forward_backward", "batch": 0},
        "small": {"operation": "forward", "batch": 1},
        "large": {"operation": "forward", "batch": 1},
    }
    reserved = spill_reservations(
        requests, candidates, descriptions, {key: key for key in held}
    )
    assert reserved == {
        # Gradients are never held while a step runs; the larger evaluation's
        # private 7 bytes are.
        "step": 7,
        # Masters and moments the update keeps, beside the evaluation.
        "backward": 20 + 20 + 7,
        "update": 7,
        # Everything but the weights it holds, gradients included.
        "eval-small": 20 + 20 + 5,
        "eval-large": 20 + 20 + 5,
    }


def test_ranks_reserve_only_for_their_own_plans(monkeypatch):
    monkeypatch.setattr(
        module,
        "_persistent",
        lambda program, rank: module._Persistent(
            {StateId("parameter", program): 8}, 0, frozenset()
        ),
    )
    requests = {
        key: SimpleNamespace(rank=rank, use=SimpleNamespace(kind="forward"))
        for key, rank in (("a0", 0), ("b0", 0), ("a1", 1))
    }
    reserved = spill_reservations(
        requests,
        {"a": ("a0", "a1"), "b": ("b0",)},
        {"a": {"operation": "a", "batch": 0}, "b": {"operation": "b", "batch": 0}},
        {key: key for key in requests},
    )
    assert reserved == {"a0": 8, "b0": 8, "a1": 0}
