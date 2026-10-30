"""Workers compare the order their Plans issue collectives in, by tensor name."""

from types import SimpleNamespace

import pytest

from shadowspill.execution.distributed.sequence import (
    check_collective_order,
    collective_sequence,
    first_difference,
)
from shadowspill.ir import MemoryActionKind

FETCH, EVICT = MemoryActionKind.FETCH, MemoryActionKind.EVICT


def phase(size):
    return SimpleNamespace(kind="timed", bytes=size)


def plan(fetch_order, *, communication=("embed", "head")):
    """A Plan fetching sharded weights in ``fetch_order``, then reducing
    gradients in ``communication`` order; ``optimizer`` state moves no bytes."""
    names = ("embed", "head", "optimizer")
    objects = [
        SimpleNamespace(object_id=f"o_{n}", alias_group_id=f"a_{n}") for n in names
    ]
    objects.append(
        SimpleNamespace(object_id="o_embed_grad", alias_group_id="a_embed_grad")
    )
    transfers = {
        "a_embed": SimpleNamespace(fetch=[phase(64)], evict=[phase(0)]),
        "a_head": SimpleNamespace(fetch=[phase(32)], evict=[phase(0)]),
        "a_optimizer": SimpleNamespace(fetch=[phase(0)], evict=[phase(0)]),
    }
    tasks = [SimpleNamespace(phase="forward", inputs=("o_embed",), outputs=())]
    tasks += [
        SimpleNamespace(phase="gradient_communication", inputs=(f"o_{n}",), outputs=())
        for n in communication
    ]
    actions = [
        SimpleNamespace(alias_group_id=f"a_{n}", kind=FETCH) for n in fetch_order
    ]
    actions.append(SimpleNamespace(alias_group_id="a_embed", kind=EVICT))
    actions.append(SimpleNamespace(alias_group_id="a_optimizer", kind=FETCH))
    return SimpleNamespace(
        program=SimpleNamespace(
            objects=objects,
            bindings={
                "registrations": [
                    {"name": n, "object_id": f"o_{n}"} for n in ("embed", "head")
                ],
                "gradients": [
                    {"gradient_object_id": "o_embed_grad", "parameter_name": "embed"}
                ],
                "weight_shards": {"a_embed": {}, "a_head": {}},
                "gradient_shards": {},
            },
        ),
        _costed_program=SimpleNamespace(
            alias_groups=[
                SimpleNamespace(alias_group_id=a, transfer=t)
                for a, t in transfers.items()
            ],
            selected_tasks=lambda selections: tasks,
        ),
        schedule=SimpleNamespace(actions=actions),
        selections=(),
    )


class Groups:
    """Every worker's value, as a gather returns it."""

    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    def gather(self, value):
        self.calls += 1
        if self.calls == 1:
            from shadowspill.artifacts._values import digest

            return [
                digest({k: [list(r) for r in v] for k, v in item.items()})
                for item in self.values
            ]
        return self.values


def test_a_plan_lists_the_collectives_it_issues_in_order():
    assert collective_sequence(plan(["embed", "head"])) == (
        ("transfer", "fetch", "embed"),
        ("transfer", "fetch", "head"),
        ("task", "gradient_communication", "embed"),
        ("task", "gradient_communication", "head"),
    )


def test_workers_whose_plans_order_collectives_differently_are_refused():
    ours = collective_sequence(plan(["embed", "head"]))
    theirs = collective_sequence(plan(["head", "embed"]))
    assert first_difference("training", [ours, ours]) is None
    assert first_difference("training", [ours, theirs]) == (
        "collective 1 of the training Plan is a fetch of embed on worker 0 "
        "but a fetch of head on worker 1"
    )
    assert first_difference("training", [ours, ours[:2]]) == (
        "collective 3 of the training Plan is a gradient communication task of "
        "embed on worker 0 but no collective on worker 1"
    )
    same = {"default": ours}
    check_collective_order(Groups(same, same), {"default": plan(["embed", "head"])})
    with pytest.raises(ValueError, match="collective 1 of the default Plan"):
        check_collective_order(
            Groups(same, {"default": theirs}), {"default": plan(["embed", "head"])}
        )
    with pytest.raises(ValueError, match="hold different Plans"):
        check_collective_order(
            Groups(same, {"default": ours, "evaluation": ours}),
            {"default": plan(["embed", "head"])},
        )
