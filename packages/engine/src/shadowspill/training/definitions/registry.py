"""Explicit registration and CPU resolution; request strings never import code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from shadowspill_client._records import Value, mapping, plain
from shadowspill_client.protocol.controls import Scalar
from shadowspill_client.protocol.definitions import DefinitionSource
from shadowspill_client.protocol.inputs import InputSchema

from .contracts import (
    Computation,
    Configuration,
    ModelDefinition,
    ObjectiveDefinition,
    Schema,
)


def _configuration(
    factory: Configuration, values: Mapping[str, Value]
) -> tuple[Any, Mapping[str, Value]]:
    configured = factory(**plain(mapping(values, "definition configuration")))
    settings: Mapping[str, Any]
    if is_dataclass(configured) and not isinstance(configured, type):
        settings = asdict(configured)
    elif isinstance(configured, Mapping):
        settings = configured
    else:
        raise TypeError("configuration factory must return a dataclass or mapping")
    return configured, mapping(settings, "resolved configuration")


def _schema(schema: Schema, config: Any) -> InputSchema:
    resolved = schema(config) if callable(schema) else schema
    if not isinstance(resolved, InputSchema):
        raise TypeError("schema resolver must return InputSchema")
    return resolved


@dataclass(frozen=True)
class ResolvedModel:
    definition: ModelDefinition
    source: DefinitionSource
    config: Mapping[str, Value]
    inputs: InputSchema

    def create(self) -> Any:
        """Construct only when requested by preparation, never during inspection."""
        configured, _ = _configuration(self.definition.config, self.config)
        return self.definition.create(configured)


@dataclass(frozen=True)
class ResolvedObjective:
    definition: ObjectiveDefinition
    source: DefinitionSource
    config: Mapping[str, Value]
    inputs: InputSchema
    defaults: Mapping[str, Scalar]

    def computation(self, outputs: str | None = None) -> Computation:
        selected = self.definition.default_outputs if outputs is None else outputs
        if selected not in self.definition.outputs:
            raise ValueError(f"unknown output set: {selected}")
        if self.definition.bind is None:
            return self.definition.compute
        configured, _ = _configuration(self.definition.config, self.config)
        computation = self.definition.bind(configured, selected)
        if not callable(computation):
            raise TypeError("objective binding must return a callable")
        return computation


class Registry:
    """One explicit registry for supplied and external installed definitions.

    Startup code registers trusted callables with their immutable distribution
    provenance. Lookup uses exact IDs only, including on remote requests. No
    dynamic module names, uploaded Python or model-family branches are involved.
    """

    def __init__(self) -> None:
        self._models: dict[str, tuple[ModelDefinition, DefinitionSource]] = {}
        self._objectives: dict[str, tuple[ObjectiveDefinition, DefinitionSource]] = {}

    def add_model(
        self, definition: ModelDefinition, *, source: DefinitionSource
    ) -> None:
        if not isinstance(definition, ModelDefinition) or not isinstance(
            source, DefinitionSource
        ):
            raise TypeError("register a ModelDefinition with DefinitionSource")
        if definition.id in self._models:
            raise ValueError(f"model already registered: {definition.id}")
        self._models[definition.id] = (definition, source)

    def add_objective(
        self, definition: ObjectiveDefinition, *, source: DefinitionSource
    ) -> None:
        if not isinstance(definition, ObjectiveDefinition) or not isinstance(
            source, DefinitionSource
        ):
            raise TypeError("register an ObjectiveDefinition with DefinitionSource")
        if definition.id in self._objectives:
            raise ValueError(f"objective already registered: {definition.id}")
        self._objectives[definition.id] = (definition, source)

    def resolve_model(
        self,
        definition: str,
        *,
        config: Mapping[str, Value] | None = None,
        source: DefinitionSource | None = None,
    ) -> ResolvedModel:
        try:
            model, installed = self._models[definition]
        except KeyError as error:
            raise ValueError(
                f"model definition is not installed: {definition}"
            ) from error
        if source is not None and source != installed:
            raise ValueError(f"model definition source changed: {definition}")
        configured, settings = _configuration(
            model.config, {} if config is None else config
        )
        return ResolvedModel(
            model, installed, settings, _schema(model.inputs, configured)
        )

    def resolve_objective(
        self,
        definition: str,
        *,
        model: ResolvedModel,
        config: Mapping[str, Value] | None = None,
        defaults: Mapping[str, Scalar] | None = None,
        source: DefinitionSource | None = None,
    ) -> ResolvedObjective:
        try:
            objective, installed = self._objectives[definition]
        except KeyError as error:
            raise ValueError(
                f"objective definition is not installed: {definition}"
            ) from error
        if source is not None and source != installed:
            raise ValueError(f"objective definition source changed: {definition}")
        if missing := set(objective.requires) - set(model.definition.capabilities):
            raise ValueError(
                f"model lacks required objective capabilities: {sorted(missing)}"
            )
        configured, settings = _configuration(
            objective.config, {} if config is None else config
        )
        inputs = _schema(objective.inputs, configured)
        model_paths = {tuple(field.path) for field in model.inputs.fields}
        loss_fields = {tuple(field.path): field for field in inputs.fields}
        for name, output in objective.outputs.items():
            if (
                set(output.model_inputs) - model_paths
                or set(output.loss_inputs) - loss_fields.keys()
            ):
                raise ValueError(f"output {name} references an undeclared input field")
        if objective.participation.path:
            field = loss_fields.get(tuple(objective.participation.path))
            if field is None or field.dtype != "bool" or field.padding is not False:
                raise ValueError(
                    "participation requires a boolean loss field with False padding"
                )
        resolved_defaults = objective.controls.resolve(defaults=defaults)
        return ResolvedObjective(
            objective, installed, settings, inputs, resolved_defaults
        )
