"""Reductions follow parameter copies, not only the within-replica shard group."""

import pytest

from shadowspill import Parallelism, ParameterStorage, Sharded
from shadowspill.parallelism.reductions import gradient_reduction
from shadowspill.parallelism.resolve import ParameterGeometry, resolve_parameters


def reduction_cases():
    ordinary = Parallelism.dp(ranks_per_replica=2, num_replicas=2)
    experts = Parallelism(
        ranks_per_replica=4,
        groups={"exchange": ((0, 1), (2, 3)), "owners": ((0, 2), (1, 3))},
        default_parameter_placement=Sharded("exchange", dim=0),
        default_parameter_storage=ParameterStorage(
            group="owners", weights="sharded", gradients="sharded", optimizer="sharded"
        ),
    )
    replicated = Parallelism.dp(ranks_per_replica=4, gradients="replicated")
    return [
        resolve_parameters(spec, (ParameterGeometry(("weight",), (5,), True),))[0]
        for spec in (ordinary, experts, replicated)
    ]


def test_gradient_completion_includes_matching_owners_across_replicas():
    ordinary, experts, replicated = reduction_cases()
    result = gradient_reduction(ordinary)
    assert (result.source_elements, result.padded_elements, result.result_elements) == (
        5,
        6,
        3,
    )
    assert [phase.operation for phase in result.phases] == [
        "reduce_scatter",
        "all_reduce",
    ]
    assert result.phases[0].groups == ((0, 1), (2, 3))
    assert result.phases[1].groups == ((0, 2), (1, 3))
    assert result.phases[1].input_elements == 3
    result = gradient_reduction(experts)
    assert len(result.phases) == 1
    assert result.phases[0].groups == ((0, 2), (1, 3))
    result = gradient_reduction(replicated)
    assert [phase.operation for phase in result.phases] == ["all_reduce"]
    assert result.result_elements == 5
    assert result.phases[0].groups == ((0, 1, 2, 3),)
    # Optimizer ownership may differ; destination geometry is explicit.
    update = gradient_reduction(replicated, destination="optimizer")
    assert update.result_elements == 2 and update.padded_elements == 8
    assert [phase.operation for phase in update.phases] == ["reduce_scatter"]


def test_no_reduction_for_unique_owned_parameter_and_no_frozen_gradient():
    spec = Parallelism.dp()
    parameter = resolve_parameters(spec, (ParameterGeometry(("weight",), (5,), True),))[
        0
    ]
    assert gradient_reduction(parameter).phases == ()
    frozen = resolve_parameters(spec, (ParameterGeometry(("weight",), (5,), False),))[0]
    with pytest.raises(ValueError, match="frozen parameters"):
        gradient_reduction(frozen)
