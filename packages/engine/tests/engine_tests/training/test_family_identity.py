"""Shared invocation names describe every rank, independent of preparation order."""

from dataclasses import replace

import pytest

from engine_tests.training.test_preparation_templates import inventory
from shadowspill.parallelism import InputFieldPlacement, InputPlacement, Parallelism
from shadowspill.step.ordering import StepDataOrdering
from shadowspill.training.preparation.planning.identity import execution_names
from shadowspill_client.protocol.operations import OperationKey


def test_execution_names_reuse_only_when_every_worker_uses_the_same_operation():
    fields = inventory().layouts.setup.context.schema.fields
    prepared = inventory(
        parallelism=Parallelism.dp(
            num_replicas=2,
            inputs=InputPlacement.independent(
                2, fields=[InputFieldPlacement(tuple(f.path)) for f in fields]
            ),
        )
    )
    step = prepared.families[OperationKey("step", "metrics")][0]
    backward = prepared.families[OperationKey("forward_backward", "metrics")][0]
    order = StepDataOrdering.depth_first(2)
    names = execution_names(backward, order)
    complete = execution_names(step, order)
    assert names.main != complete.main
    assert names.optimizer is not None and complete.optimizer is None
    assert (
        execution_names(
            replace(backward, workers=dict(reversed(list(backward.workers.items())))),
            order,
        )
        == names
    )
    other_order = execution_names(backward, StepDataOrdering(1, 2))
    assert other_order.main != names.main
    assert other_order.optimizer == names.optimizer
    # Rank 0's local Plan remains reusable; the invocation must still differ
    # when rank 1 needs different work. Equal communication digests alone cannot
    # decide this, since forward and optimizer-only Plans may exchange
    # the same tensors while computing different results.
    peer = backward.workers[1]
    for main in (
        replace(peer.main, count=peer.main.count + 1),
        replace(peer.main, template="another-capture"),
        replace(peer.main, kind="forward"),
    ):
        changed = replace(
            backward,
            workers={0: backward.workers[0], 1: replace(peer, main=main)},
        )
        assert execution_names(changed, order).main != names.main
    with pytest.raises(ValueError, match="operation dependencies"):
        execution_names(
            replace(
                backward,
                workers={0: backward.workers[0], 1: replace(peer, optimizer=None)},
            ),
            order,
        )
