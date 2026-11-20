"""Independent non-text losses share one captured model evaluation."""

from copy import deepcopy
from dataclasses import replace
from functools import partial

import pytest
import torch

from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.pytorch.contracts import ObjectiveResult
from shadowspill.training.components import (
    ModelComponent,
    ObjectiveComponent,
    ObjectiveEvaluation,
    Registry,
)
from shadowspill.training.preparation.configuration.operations import OperationContract
from shadowspill_client import Objective
from shadowspill_client.protocol.components import (
    ComponentSource,
    OutputSet,
    Participation,
    TensorOutput,
)
from shadowspill_client.protocol.controls import ControlSchema, ScalarControl
from shadowspill_client.protocol.inputs import InputSchema, TensorField
from shadowspill_client.protocol.operations import OperationKey

SOURCE = ComponentSource("external-composition", "1", "a" * 64)


def evaluate(model, inputs, loss_inputs, requests):
    assert requests == {"prediction": True}
    return model(inputs["features"])


def evaluation(config, key):
    return ObjectiveEvaluation("external/regression@1", evaluate, {"prediction": True})


def reduce(prediction, inputs, fields, controls, *, absolute=False, outputs=False):
    valid = fields["valid"]
    residual = (
        prediction
        - torch.where(valid[:, None], fields["target"], 0)
        - controls["offset"]
    )
    values = residual.abs() if absolute else residual.square()
    loss = torch.where(valid[:, None], values, 0).sum()
    metrics = {"loss_sum": loss.detach(), "valid": valid.sum()}
    if outputs:
        metrics["prediction"] = prediction
    return ObjectiveResult(loss, metrics)


def forward(prediction, inputs, fields, controls):
    return {"prediction": prediction}


def bind(config, key, *, absolute=False):
    if key.operation == "forward" and key.outputs == "prediction":
        return forward
    return partial(reduce, absolute=absolute, outputs=key.outputs == "prediction")


def fixture(*, second=None):
    registry = Registry()
    registry.add_model(
        ModelComponent(
            "external/regression@1",
            dict,
            lambda _: torch.nn.Linear(2, 1),
            InputSchema([TensorField(("features",), "float32", (2,))]),
        ),
        source=SOURCE,
    )
    fields = InputSchema(
        [
            TensorField(("target",), "float32", (1,)),
            TensorField(("valid",), "bool", padding=False),
        ]
    )
    base = OutputSet(
        model_inputs=(("features",),),
        loss_inputs=(("target",), ("valid",)),
        controls=("offset",),
        metrics={"loss_sum": "sum", "valid": "sum"},
    )
    tensor = {"prediction": TensorOutput("float32", (1,), "datum")}
    for name, absolute in (("squared", False), ("absolute", True)):
        component = ObjectiveComponent(
            f"external/{name}@1",
            partial(reduce, absolute=absolute),
            fields,
            controls=ControlSchema({"offset": ScalarControl(0.0)}),
            outputs={
                "metrics": base,
                "prediction": replace(
                    base,
                    tensors=tensor,
                    forward=OutputSet(model_inputs=(("features",),), tensors=tensor),
                ),
            },
            bind=partial(bind, absolute=absolute),
            evaluate=evaluation,
            participation=Participation(("valid",)),
            padding_safe=True,
        )
        if absolute and second is not None:
            component = second(component)
        registry.add_objective(component, source=SOURCE)
    model = registry.resolve_model("external/regression@1")
    settings = Objective.sum(
        {
            "squared": Objective("external/squared@1", defaults={"offset": 0.2}),
            "absolute": Objective("external/absolute@1"),
        },
        weights={"absolute": 0.3},
    )
    resolved = registry.resolve_objective(
        settings.name,
        model=model,
        config=settings.config,
        defaults=settings.defaults,
    )
    return registry, model, settings, resolved


def test_composed_short_objective_names_pin_every_child_component():
    registry, model, settings, expected = fixture()
    short = Objective.sum(
        {
            "squared": Objective("squared", defaults={"offset": 0.2}),
            "absolute": Objective("absolute"),
        },
        weights={"absolute": 0.3},
    )
    actual = registry.resolve_objective(
        short.name, model=model, config=short.config, defaults=short.defaults
    )
    assert actual.config == expected.config
    assert actual.source == expected.source
    assert actual.defaults == expected.defaults
    assert actual.config["terms"]["squared"]["name"] == "external/squared@1"
    squared = registry.resolve_objective("squared", model=model)
    registry.add_objective(
        replace(squared.component, id="other/squared@1"), source=SOURCE
    )
    with pytest.raises(ValueError, match="ambiguous objective name"):
        registry.resolve_objective(short.name, model=model, config=short.config)
    pinned = registry.resolve_objective(
        settings.name,
        model=model,
        config=actual.config,
        defaults=actual.defaults,
        source=actual.source,
    )
    assert pinned.config == actual.config


@pytest.mark.parametrize("split", [False, True])
def test_weighted_sum_two_updates_and_each_gradient_use_one_evaluation(split):
    _, description, _, objective = fixture()
    torch.manual_seed(97)
    model = description.create()
    reference = deepcopy(model)
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    inputs = {"features": torch.randn(4, 2)}
    fields = {
        "target": torch.tensor([[0.2], [-1.5], [0.7], [float("nan")]]),
        "valid": torch.tensor([True, True, True, False]),
    }
    compute = objective.computation(
        OperationKey("forward_backward" if split else "step", "metrics")
    )
    for weight in (0.3, -0.5):
        controls = {
            name: torch.tensor(value)
            for name, value in objective.controls.resolve(
                defaults=objective.defaults, overrides={"absolute.weight": weight}
            ).items()
        }
        model.zero_grad()
        reference.zero_grad()
        for part in (slice(0, 2), slice(2, 4)) if split else (slice(None),):
            result = compute(
                model,
                {k: v[part] for k, v in inputs.items()},
                {k: v[part] for k, v in fields.items()},
                controls,
            )
            x, y, valid = (
                inputs["features"][part],
                fields["target"][part],
                fields["valid"][part],
            )
            prediction = torch.nn.functional.linear(x, reference.weight, reference.bias)
            difference = prediction[valid] - y[valid]
            expected = ((difference - 0.2).square() + weight * difference.abs()).sum()
            torch.testing.assert_close(result.loss, expected)
            torch.testing.assert_close(result.metrics["loss_sum"], expected)
            assert result.metrics["squared.valid"] == valid.sum()
            (result.loss * 0.4).backward()
            (expected * 0.4).backward()
        for actual, expected in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual.grad, expected.grad)
            with torch.no_grad():
                actual.add_(actual.grad, alpha=-0.01)
                expected.add_(expected.grad, alpha=-0.01)
            torch.testing.assert_close(actual, expected)
    assert len(calls) == (4 if split else 2)
    hook.remove()
    fields["valid"].zero_()
    result = compute(model, inputs, fields, controls)
    assert result.loss.item() == 0
    assert all(
        not g.any() for g in torch.autograd.grad(result.loss, tuple(model.parameters()))
    )


def test_composed_capture_keeps_coefficients_as_runtime_inputs():
    _, description, _, objective = fixture()
    model = description.create()
    inputs = {"features": torch.randn(3, 2)}
    fields = {"target": torch.randn(3, 1), "valid": torch.tensor([True, True, False])}
    controls = {name: torch.tensor(value) for name, value in objective.defaults.items()}
    captured = capture_training_objective(
        model, objective.computation(), [inputs, fields, controls]
    )
    graph = captured.exported.exported_program.module()
    for coefficient in (0.0, 0.8):
        controls["absolute.weight"] = torch.tensor(coefficient)
        actual, *metrics = graph(inputs, fields, controls)
        difference = (
            model(inputs["features"])[fields["valid"]]
            - fields["target"][fields["valid"]]
        )
        expected = ((difference - 0.2).square() + coefficient * difference.abs()).sum()
        torch.testing.assert_close(actual, expected)
        assert len(metrics) == 5
        for a, b in zip(
            torch.autograd.grad(actual, tuple(model.parameters())),
            torch.autograd.grad(expected, tuple(model.parameters())),
            strict=True,
        ):
            torch.testing.assert_close(a, b)


def test_output_only_forward_needs_no_targets_or_coefficients_and_runs_once():
    _, description, _, objective = fixture()
    model = description.create()
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    key = OperationKey("forward", "prediction", "evaluation")
    contract = objective.outputs["prediction"].select(forward=True)
    assert not contract.loss_inputs and not contract.controls
    inputs = {"features": torch.randn(2, 2)}
    values = objective.computation(key)(model, inputs, {}, {})
    assert set(values) == {"squared.prediction", "absolute.prediction"}
    assert len(calls) == 1
    hook.remove()


def test_forward_metrics_preserves_weighted_loss_and_call_local_coefficients():
    _, description, _, objective = fixture()
    model = description.create()
    key = OperationKey("forward", "metrics", "evaluation")
    contract = OperationContract(key, objective)
    compute = objective.computation(key)
    inputs = {"features": torch.randn(3, 2)}
    fields = {
        "target": torch.tensor([[0.3], [-1.2], [float("nan")]]),
        "valid": torch.tensor([True, True, False]),
    }
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        difference = (
            torch.nn.functional.linear(inputs["features"][:2], model.weight, model.bias)
            - fields["target"][:2]
        )
        for overrides, weight in (
            (None, 0.3),
            ({"absolute.weight": -0.5}, -0.5),
            (None, 0.3),
        ):
            controls = {
                name: torch.tensor(value)
                for name, value in contract.loss_params(overrides).items()
            }
            result = compute(model, inputs, fields, controls)
            assert isinstance(result, ObjectiveResult)
            expected = ((difference - 0.2).square() + weight * difference.abs()).sum()
            torch.testing.assert_close(result.loss, expected)
            assert set(result.metrics) == set(contract.outputs.metrics)
    assert len(calls) == 3
    assert all(parameter.grad is None for parameter in model.parameters())
    hook.remove()


def test_forward_metrics_does_not_report_a_partial_loss_for_output_only_child():
    def prediction_metric(prediction, inputs, fields, controls):
        return {"mean": prediction.mean()}

    def second(component):
        original = component.bind
        return replace(
            component,
            outputs={
                **component.outputs,
                "metrics": replace(
                    component.outputs["metrics"],
                    forward=OutputSet(
                        model_inputs=(("features",),), metrics={"mean": "sum"}
                    ),
                ),
            },
            bind=lambda config, key: (
                prediction_metric
                if key.operation == "forward" and key.outputs == "metrics"
                else original(config, key)
            ),
        )

    _, description, _, objective = fixture(second=second)
    key = OperationKey("forward", "metrics", "evaluation")
    contract = OperationContract(key, objective)
    result = objective.computation(key)(
        description.create(),
        {"features": torch.ones(2, 2)},
        {"target": torch.zeros(2, 1), "valid": torch.ones(2, dtype=torch.bool)},
        {name: torch.tensor(value) for name, value in contract.loss_params().items()},
    )
    assert isinstance(result, dict) and set(result) == set(contract.outputs.metrics)


def test_normalized_setup_roundtrip_and_child_source_change_are_checked():
    registry, model, _, objective = fixture()
    normalized = Objective.from_dict(
        Objective("shadowspill/sum@1", objective.config, objective.defaults).to_dict()
    )
    replay = registry.resolve_objective(
        normalized.name,
        model=model,
        config=normalized.config,
        defaults=normalized.defaults,
        source=objective.source,
    )
    assert replay.config == objective.config and replay.defaults == objective.defaults
    component, _ = registry._objectives["external/absolute@1"]
    registry._objectives[component.id] = (component, replace(SOURCE, digest="b" * 64))
    with pytest.raises(ValueError, match="source changed"):
        registry.resolve_objective(
            normalized.name,
            model=model,
            config=normalized.config,
            source=objective.source,
        )


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda d: replace(d, evaluate=None), "shared evaluation hook"),
        (
            lambda d: replace(
                d,
                evaluate=lambda c, k: ObjectiveEvaluation("external/other@1", evaluate),
            ),
            "incompatible model evaluations",
        ),
        (
            lambda d: replace(
                d,
                evaluate=lambda c, k: ObjectiveEvaluation(
                    "external/regression@1", evaluate, {"prediction": False}
                ),
            ),
            "evaluation requests",
        ),
        (lambda d: replace(d, participation=Participation()), "participation field"),
        (
            lambda d: replace(
                d,
                inputs=InputSchema(
                    [
                        replace(f, default=True, required=False)
                        if f.path == ("valid",)
                        else f
                        for f in d.inputs.fields
                    ]
                ),
            ),
            "input schemas",
        ),
        (lambda d: replace(d, padding_safe=False), "padding safety"),
    ],
)
def test_incompatible_terms_fail_during_cpu_resolution(change, match):
    with pytest.raises(ValueError, match=match):
        fixture(second=change)


def test_evaluation_receives_original_mode():
    modes = []

    def recorded(config, key):
        modes.append(key.mode)
        return evaluation(config, key)

    _, _, _, objective = fixture(second=lambda d: replace(d, evaluate=recorded))
    modes.clear()
    objective.computation(OperationKey("forward", "metrics", "evaluation"))
    assert modes == ["evaluation"]
