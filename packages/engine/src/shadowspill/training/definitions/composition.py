"""CPU resolution and one-evaluation execution of named objective sums."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from shadowspill_client import Objective
from shadowspill_client._records import Value, mapping, plain
from shadowspill_client.protocol.controls import ControlSchema, Scalar, ScalarControl
from shadowspill_client.protocol.definitions import DefinitionSource, OutputSet
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.operations import OperationKey

from .contracts import Computation, ObjectiveDefinition, ObjectiveEvaluation
from .registry import Registry, ResolvedModel, ResolvedObjective


def _merge(target: dict[Any, Any], values: Mapping[Any, Any], kind: str) -> None:
    for name, value in values.items():
        if name in target and target[name] != value:
            raise ValueError(f"objective terms disagree on {kind}: {name}")
        target[name] = value


def _selection(
    terms: Mapping[str, ResolvedObjective], key: OperationKey
) -> dict[str, tuple[ResolvedObjective, OperationKey]]:
    return {
        name: (
            term,
            OperationKey(
                key.operation,
                key.outputs
                if key.outputs != "metrics" and key.outputs in term.outputs
                else term.definition.default_outputs,
                key.mode,
            ),
        )
        for name, term in terms.items()
        if key.operation != "forward"
        or key.outputs == "metrics"
        or key.outputs in term.outputs
    }


def _evaluation(
    selected: Mapping[str, tuple[ResolvedObjective, OperationKey]],
) -> ObjectiveEvaluation:
    shared = None
    requests: dict[str, Value] = {}
    for name, (term, key) in selected.items():
        evaluation = term.evaluation(key)
        if evaluation is None:
            raise ValueError(
                f"objective term {name} must declare a shared evaluation hook"
            )
        if shared is not None and (
            shared.id != evaluation.id or shared.compute != evaluation.compute
        ):
            raise ValueError("objective terms require incompatible model evaluations")
        shared = evaluation
        _merge(requests, evaluation.requests, "evaluation requests")
    assert shared is not None
    return replace(shared, requests=requests)


def _outputs(
    selected: Mapping[str, tuple[ResolvedObjective, OperationKey]], *, forward: bool
) -> OutputSet:
    model_inputs: set[tuple[str, ...]] = set()
    loss_inputs: set[tuple[str, ...]] = set()
    controls: list[str] = []
    metrics: dict[str, Any] = {} if forward else {"loss_sum": "sum"}
    tensors = {}
    for name, (term, key) in selected.items():
        output = term.outputs[cast(str, key.outputs)].select(forward=forward)
        model_inputs.update(tuple(p) for p in output.model_inputs)
        loss_inputs.update(tuple(p) for p in output.loss_inputs)
        controls.extend(f"{name}.{control}" for control in output.controls)
        if not forward:
            controls.append(f"{name}.weight")
        metrics.update(
            {f"{name}.{metric}": value for metric, value in output.metrics.items()}
        )
        tensors.update(
            {f"{name}.{tensor}": value for tensor, value in output.tensors.items()}
        )
    return OutputSet(
        model_inputs=sorted(model_inputs),
        loss_inputs=sorted(loss_inputs),
        controls=controls,
        metrics=metrics,
        tensors=tensors,
    )


def _bind(terms: Mapping[str, ResolvedObjective], key: OperationKey) -> Computation:
    selected = _selection(terms, key)
    evaluation = _evaluation(selected)
    reducers = {name: term.reducer(child) for name, (term, child) in selected.items()}
    contracts = {
        name: term.outputs[cast(str, child.outputs)].select(
            forward=key.operation == "forward"
        )
        for name, (term, child) in selected.items()
    }

    def compute(model: Any, inputs: Any, loss_inputs: Any, loss_params: Any) -> Any:
        from shadowspill.pytorch.contracts import (
            ObjectiveResult,
            normalize_objective_result,
        )

        values = evaluation.compute(model, inputs, loss_inputs, evaluation.requests)
        total, results = None, {}
        for name, reducer in reducers.items():
            controls = {
                control: loss_params[f"{name}.{control}"]
                for control in contracts[name].controls
            }
            result = reducer(values, inputs, loss_inputs, controls)
            if key.operation == "forward" and isinstance(result, Mapping):
                metrics = result
            else:
                loss, metrics = normalize_objective_result(result, require_grad=False)
                if key.operation != "forward":
                    weighted = loss * loss_params[f"{name}.weight"]
                    total = weighted if total is None else total + weighted
            expected = set(contracts[name].metrics) | set(contracts[name].tensors)
            metrics = {} if metrics is None else metrics
            if not isinstance(metrics, Mapping) or set(metrics) != expected:
                raise ValueError(f"objective term {name} returned undeclared outputs")
            results.update(
                {f"{name}.{metric}": value for metric, value in metrics.items()}
            )
        if key.operation == "forward":
            return results
        assert total is not None
        results["loss_sum"] = total.detach()
        return ObjectiveResult(total, results)

    return compute


def _source(terms: Mapping[str, ResolvedObjective]) -> DefinitionSource:
    digest = sha256()
    for name in ("composition.py", "contracts.py", "registry.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    digest.update(
        json.dumps(
            {
                name: {"id": term.definition.id, "source": term.source.to_dict()}
                for name, term in terms.items()
            },
            sort_keys=True,
        ).encode()
    )
    return DefinitionSource("shadowspill", version("shadowspill"), digest.hexdigest())


def resolve_sum(
    registry: Registry,
    model: ResolvedModel,
    config: Mapping[str, Value] | None,
    defaults: Mapping[str, Scalar] | None,
    source: DefinitionSource | None,
) -> ResolvedObjective:
    if (
        config is None
        or set(config) != {"terms"}
        or not isinstance(config["terms"], Mapping)
    ):
        raise ValueError("an objective sum requires named terms")
    settings = {
        name: Objective.from_dict(plain(value))
        for name, value in config["terms"].items()
    }
    Objective.sum(settings)  # Same lightweight validation for remote requests.
    terms = {
        name: registry.resolve_objective(
            term.definition,
            model=model,
            config=term.config,
            defaults=cast(Mapping[str, Scalar], term.defaults),
        )
        for name, term in sorted(settings.items())
    }
    installed = _source(terms)
    if source is not None and source != installed:
        raise ValueError("objective definition source changed: shadowspill/sum@1")
    fields: dict[tuple[str, ...], TensorField] = {}
    capacities: dict[str, Capacity] = {}
    controls: dict[str, ScalarControl] = {}
    configured_defaults: dict[str, Scalar] = {}
    invariants: dict[str, str] = {}
    participation = next(iter(terms.values())).definition.participation
    for name, term in terms.items():
        if not (term.definition.padding_safe or term.definition.physical_batch_loss):
            raise ValueError(f"objective term {name} must declare padding safety")
        if term.definition.participation != participation:
            raise ValueError("objective terms must share the same participation field")
        if "weight" in term.controls.fields:
            raise ValueError("objective term control 'weight' is reserved by sums")
        _merge(fields, {tuple(f.path): f for f in term.inputs.fields}, "input schemas")
        _merge(capacities, {c.name: c for c in term.inputs.capacities}, "capacities")
        _merge(
            invariants, term.definition.allocation_invariants, "allocation invariants"
        )
        controls.update(
            {
                f"{name}.{control}": value
                for control, value in term.controls.fields.items()
            }
        )
        controls[f"{name}.weight"] = ScalarControl(1.0)
        configured_defaults.update(
            {f"{name}.{control}": value for control, value in term.defaults.items()}
        )
    schema = InputSchema(tuple(fields.values()), tuple(capacities.values()))
    control_schema = ControlSchema(controls)
    resolved_defaults = control_schema.resolve(
        defaults=configured_defaults, overrides=defaults
    )
    output_names = {"metrics"}
    for term in terms.values():
        output_names.update(set(term.outputs) - {term.definition.default_outputs})
    outputs = {}
    for output in sorted(output_names):
        selections = {
            operation: _selection(terms, OperationKey(operation, output))
            for operation in ("step", "forward_backward", "forward")
        }
        for selected in selections.values():
            _evaluation(selected)  # Reject incompatibility on CPU, before capture.
        _evaluation(_selection(terms, OperationKey("forward", output, "evaluation")))
        outputs[output] = replace(
            _outputs(selections["step"], forward=False),
            forward=_outputs(selections["forward"], forward=True),
        )

    def bind(configured: Any, key: OperationKey) -> Computation:
        return _bind(terms, key)

    objective = ObjectiveDefinition(
        "shadowspill/sum@1",
        _bind(terms, OperationKey("step", "metrics")),
        schema,
        controls=control_schema,
        outputs=outputs,
        bind=bind,
        participation=participation,
        requires=tuple(
            sorted({cap for term in terms.values() for cap in term.definition.requires})
        ),
        reduction="sum"
        if all(term.definition.reduction == "sum" for term in terms.values())
        else "custom",
        padding_safe=all(term.definition.padding_safe for term in terms.values()),
        physical_batch_loss=any(
            term.definition.physical_batch_loss for term in terms.values()
        ),
        allocation_invariants=invariants,
    )
    resolved_config = mapping(
        {
            "terms": {
                name: Objective(
                    term.definition.id, term.config, term.defaults
                ).to_dict()
                for name, term in terms.items()
            }
        },
        "objective sum",
    )
    return ResolvedObjective(
        objective, installed, resolved_config, schema, resolved_defaults
    )
