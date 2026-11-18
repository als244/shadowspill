"""A step's token count comes from the schema's own unit, or there is none."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.training.test_worker_loop import call
from shadowspill.server.plan_summaries import _scheduled_tokens
from shadowspill.training.session.prepared import PreparedCall
from shadowspill.training.throughput import token_capacity
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField

TOKENS = TensorField(("tokens",), "int64", ("position",), "pack", 0)
SUMMED = Capacity("token_capacity", "tokens", "sum", ("tokens",), 0, scheduled=True)
LONGEST = Capacity("max_sequence_length", "tokens", "max", ("tokens",), 0)
COUNT = Capacity("max_sequences", "sequences", "count")


def schema(*capacities):
    return InputSchema([TOKENS], list(capacities))


def test_the_scheduled_capacity_counted_in_tokens_is_the_steps_token_count():
    assert token_capacity(schema(SUMMED, LONGEST, COUNT)) == "token_capacity"


@pytest.mark.parametrize(
    "capacities",
    [
        (),
        (LONGEST, COUNT),  # token-valued limits that no schedule sums
        (replace(SUMMED, scheduled=False),),
        (Capacity("max_examples", "examples", "count", scheduled=True),),
        (SUMMED, replace(SUMMED, name="second_capacity")),  # no single count
    ],
)
def test_a_schema_without_one_scheduled_token_capacity_has_no_token_count(capacities):
    assert token_capacity(schema(*capacities)) is None


def assignment(capacities, supplied, scheduled):
    described = schema(*capacities)
    return SimpleNamespace(
        batch=SimpleNamespace(
            schema=described,
            capacities={"token_capacity": supplied, "max_sequences": 3},
        ),
        schedule=SimpleNamespace(
            capacities=lambda given: (
                {"token_capacity": scheduled} if given is described else None
            )
        ),
    )


def test_a_request_reports_what_it_supplies_beside_what_is_scheduled():
    prepared = SimpleNamespace(assignment=assignment((SUMMED, COUNT), 900, 1024))
    assert PreparedCall.step_tokens(prepared) == (900, 1024)


def test_a_request_without_inputs_or_without_a_token_capacity_reports_none():
    assert PreparedCall.step_tokens(SimpleNamespace(assignment=None)) is None
    prepared = SimpleNamespace(assignment=assignment((COUNT,), 900, 1024))
    assert PreparedCall.step_tokens(prepared) is None


def test_a_worker_call_carries_both_counts_or_neither():
    command = replace(call(), batch_tokens=900, scheduled_tokens=1024)
    assert type(command).from_dict(command.to_dict()) == command
    assert call().batch_tokens is None and call().scheduled_tokens is None
    for supplied, scheduled in (
        (900, None),
        (None, 1024),
        (1025, 1024),
        (-1, 4),
        (1.5, 4),
    ):
        with pytest.raises(ValueError):
            replace(call(), batch_tokens=supplied, scheduled_tokens=scheduled)


def candidate(scheduled, workers):
    return SimpleNamespace(
        source=SimpleNamespace(
            layout=SimpleNamespace(
                candidate=SimpleNamespace(
                    schedule=SimpleNamespace(
                        capacities=lambda _: {"token_capacity": scheduled}
                    )
                )
            )
        ),
        workers={
            rank: SimpleNamespace(main=main, optimizer="optimizer")
            for rank, main in workers.items()
        },
    )


def prepared(described, *candidates):
    context = SimpleNamespace(schema=described)
    setup = SimpleNamespace(context=context)
    return SimpleNamespace(
        inventory=SimpleNamespace(layouts=SimpleNamespace(setup=setup)),
        families={"step": candidates},
    )


def test_each_compute_plan_is_priced_for_its_candidates_scheduled_tokens():
    plans = prepared(
        schema(SUMMED),
        candidate(1024, {0: "one", 1: "one"}),
        candidate(4096, {0: "four", 1: "shared"}),
        candidate(8192, {0: "eight", 1: "shared"}),
    )
    # A Plan two candidates schedule differently, and the optimizer Plan, have none.
    assert _scheduled_tokens(plans) == {
        (0, "one"): 1024,
        (1, "one"): 1024,
        (0, "four"): 4096,
        (0, "eight"): 8192,
    }
    assert _scheduled_tokens(prepared(schema(COUNT), candidate(1024, {0: "one"}))) == {}
