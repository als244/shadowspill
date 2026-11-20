"""Resolved setup stays immutable and portable in a NumPy-only client."""

from dataclasses import replace

import pytest

from shadowspill_client import (
    BatchConfig,
    BuildCase,
    BuildOptions,
    Execution,
    Memory,
    Model,
    Objective,
    PlanningOptions,
    Policy,
    ProfilingOptions,
    Weights,
)
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.components import (
    ComponentSource,
    OutputSet,
    Participation,
)
from shadowspill_client.protocol.controls import ControlSchema
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_client.protocol.preparation import ResolvedPreparation
from shadowspill_client.protocol.training import ResolvedTraining


def resolved():
    source = ComponentSource("external", "1", "a" * 64)
    return ResolvedTraining(
        model=Model("external/model@1", initialize=Weights.random(seed=4)),
        objective=Objective("external/objective@1"),
        optimizer=None,
        batches=[BatchConfig(limits={"records": 2})],
        memory=Memory(2, 1),
        parallelism={"portable_core_record": True},
        execution=Execution(),
        operations=[OperationKey("forward", "prediction", "evaluation")],
        preparation=ResolvedPreparation(
            "external/preset@1",
            source,
            [
                BuildCase(
                    2,
                    BuildOptions(
                        partition=Policy("external/whole@1"),
                        chunking=None,
                        graph_pair_policy=None,
                    ),
                )
            ],
            ProfilingOptions(),
            PlanningOptions(microbatch_counts=[1], search={}, workers=0),
            {},
        ),
        component_sources={"model": source, "objective": source},
        policy_sources={"external/whole@1": source},
        engine_source=source,
        client_source=source,
        inputs=InputSchema(
            [TensorField(("inputs", "x"), "float32", (4,))],
            [Capacity("records", "items", "count")],
        ),
        loss_controls=ControlSchema({}),
        optimizer_controls=None,
        outputs={"prediction": OutputSet(model_inputs=[("x",)])},
        participation=Participation(),
        origins={"model": "explicit"},
    )


def test_full_setup_roundtrip_owns_nested_config_and_source_records():
    config = resolved()
    assert decode(encode(config)) == config
    assert ResolvedTraining.from_dict(config.to_dict()) == config
    with pytest.raises(TypeError):
        config.component_sources["model"] = config.engine_source
    with pytest.raises(TypeError):
        config.parallelism["portable_core_record"] = False


@pytest.mark.parametrize(
    "change",
    [
        lambda c: {"model": Model.from_pretrained("fixture://model")},
        lambda c: {"operations": []},
        lambda c: {"operations": [*c.operations, *c.operations]},
        lambda c: {"operations": [OperationKey("optim_step", None)]},
        lambda c: {"operations": [OperationKey("step", "prediction")]},
        lambda c: {"outputs": {}},
        lambda c: {"optimizer_controls": ControlSchema({})},
        lambda c: {"component_sources": {"model": c.engine_source}},
        lambda c: {"batches": [*c.batches, *c.batches]},
        lambda c: {"origins": {"model": "guessed"}},
    ],
)
def test_incomplete_or_inconsistent_snapshots_are_not_resolved_configs(change):
    config = resolved()
    with pytest.raises((ValueError, TypeError)):
        replace(config, **change(config))
