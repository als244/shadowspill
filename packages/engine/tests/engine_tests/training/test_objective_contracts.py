"""Structural and per-operation objective contracts through real CPU capture."""

from dataclasses import dataclass, replace

import pytest
import torch

from engine_tests.fixtures.definition_operations import (
    definitions,
    example,
    predictions,
)
from engine_tests.fixtures.external_definition import classification_loss
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.pytorch.contracts import ObjectiveResult
from shadowspill.training.definitions import Registry
from shadowspill.training.preparation.operations import OperationContract
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.controls import ControlSchema
from shadowspill_client.protocol.definitions import OutputSet, TensorOutput
from shadowspill_client.protocol.operations import OperationKey


def training_predictions(model, inputs, loss_inputs, loss_params):
    result = classification_loss(model, inputs, loss_inputs, loss_params)
    return ObjectiveResult(
        result.loss, {**result.metrics, "prediction": model(**inputs)}
    )


def operation_binding(config, key):
    return predictions if key.operation == "forward" else training_predictions


def contract_definition():
    _, objective, _ = definitions()
    metrics = objective.outputs["metrics"]
    tensor = {"prediction": TensorOutput("float32", (3,), "datum")}
    return replace(
        objective.definition,
        outputs={
            "metrics": replace(
                metrics,
                tensors=tensor,
                forward=OutputSet(model_inputs=metrics.model_inputs, tensors=tensor),
            )
        },
        bind=operation_binding,
    )


def test_same_output_name_has_distinct_saved_forward_and_training_requirements():
    registry = installed(objective=contract_definition())
    setup = resolve_training(
        registry,
        **arguments(),
        operations={"forward_backward": ["metrics"], "forward": ["metrics"]},
    )
    forward = OperationContract(
        OperationKey("forward", "metrics", "evaluation"), setup.objective
    )
    backward = OperationContract(
        OperationKey("forward_backward", "metrics"), setup.objective
    )
    assert forward.required_paths == (("inputs", "pixels"),)
    assert not forward.controls.fields and not forward.outputs.metrics
    assert ("loss_inputs", "class_id") in backward.required_paths
    assert backward.controls.fields.keys() == {"temperature"}
    assert backward.outputs.metrics == {"valid_examples": "sum"}
    assert backward.outputs.forward is None
    assert (
        OperationContract(
            OperationKey("forward", "metrics", "training"), setup.objective
        ).outputs
        == forward.outputs
    )
    assert decode(encode(setup.config)) == setup.config
    assert resolve_training(registry, setup.config).config == setup.config
    changed = replace(contract_definition(), outputs={"metrics": backward.outputs})
    with pytest.raises(ValueError, match="contract differs"):
        resolve_training(installed(objective=changed), setup.config)


def test_bound_forward_omits_targets_but_backward_captures_loss_and_predictions():
    model, objective, _ = definitions()
    registry = Registry()
    registry.add_objective(contract_definition(), source=objective.source)
    objective = registry.resolve_objective(objective.definition.id, model=model)
    created = model.create()
    from engine_tests.fixtures.definition_operations import records

    forward = OperationKey("forward", "metrics", "evaluation")
    with example(model, objective, forward, datums=records(targets=False)) as data:
        binding, adapted = data
        loss, values = binding.forward(created, *binding.arguments(adapted))
        assert loss is None and values["prediction"].shape == (4, 3)
    backward = OperationKey("forward_backward", "metrics")
    with (
        pytest.raises(ValueError, match="missing required field"),
        example(model, objective, backward, datums=records(targets=False)),
    ):
        pytest.fail("accepted missing training data")
    with example(model, objective, backward) as (binding, adapted):
        inputs = binding.arguments(adapted)
        captured = capture_training_objective(created, binding.objective, list(inputs))
        loss, prediction, valid = captured.exported.exported_program.module()(*inputs)
        expected = classification_loss(created, *inputs)
        torch.testing.assert_close(loss, expected.loss)
        assert prediction.shape == (4, 3) and valid.item() == 3
        assert all(
            torch.isfinite(g).all()
            for g in torch.autograd.grad(loss, tuple(created.parameters()))
        )


@dataclass(frozen=True)
class Configuration:
    num_classes: int
    scaled: bool = True


def test_model_dependent_configuration_resolves_conditional_contracts_on_cpu():
    model, objective, _ = definitions()

    def configure(values, resolved_model):
        size = resolved_model.config["num_classes"]
        if values.get("num_classes", size) != size:
            raise ValueError("class count differs from model")
        return {**values, "num_classes": size}

    definition = replace(
        objective.definition,
        config=Configuration,
        configure=configure,
        inputs=lambda config: objective.inputs,
        controls=lambda config: (
            objective.controls if config.scaled else ControlSchema({})
        ),
        outputs=lambda config: {
            "metrics": replace(
                objective.outputs["metrics"],
                controls=("temperature",) if config.scaled else (),
            )
        },
    )
    registry = Registry()
    registry.add_objective(definition, source=objective.source)
    for scaled in (True, False):
        resolved = registry.resolve_objective(
            definition.id, model=model, config={"scaled": scaled}
        )
        assert resolved.config == {"num_classes": 3, "scaled": scaled}
        assert resolved.defaults == ({"temperature": 1.0} if scaled else {})
        assert (
            registry.resolve_objective(
                definition.id, model=model, config=resolved.config
            ).outputs
            == resolved.outputs
        )
    with pytest.raises(ValueError, match="differs from model"):
        registry.resolve_objective(
            definition.id, model=model, config={"num_classes": 4}
        )
    with pytest.raises(ValueError, match="unknown"):
        registry.resolve_objective(
            definition.id,
            model=model,
            config={"scaled": False},
            defaults={"temperature": 2.0},
        )
    unstable = replace(
        definition,
        id="external/unstable_objective@1",
        configure=lambda values, model: {
            "num_classes": values.get("num_classes", 0) + 1
        },
    )
    registry.add_objective(unstable, source=objective.source)
    with pytest.raises(ValueError, match="not stable"):
        registry.resolve_objective(unstable.id, model=model)


def test_forward_contract_is_validated_even_if_not_requested():
    model, objective, _ = definitions()
    registry = Registry()
    definition = replace(
        objective.definition,
        outputs={"metrics": OutputSet(forward=OutputSet(loss_inputs=[("typo",)]))},
    )
    registry.add_objective(definition, source=objective.source)
    with pytest.raises(ValueError, match="undeclared input"):
        registry.resolve_objective(definition.id, model=model)
    with pytest.raises(ValueError, match="undeclared control"):
        replace(
            definition,
            outputs={"metrics": OutputSet(forward=OutputSet(controls=["typo"]))},
        )
    with pytest.raises(ValueError, match="cannot be nested"):
        OutputSet(forward=OutputSet(forward=OutputSet()))
