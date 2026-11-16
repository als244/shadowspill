"""Explicit registration and CPU resolution; request strings never import code."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from shadowspill.parallelism import Parallelism
from shadowspill.resources.description import validate_resources
from shadowspill_client import Model, Optimizer
from shadowspill_client._records import Value, mapping, plain
from shadowspill_client.protocol.controls import ControlSchema, Scalar
from shadowspill_client.protocol.definitions import DefinitionSource, OutputSet
from shadowspill_client.protocol.inputs import InputSchema
from shadowspill_client.protocol.operations import OperationKey

from .build import ModelBuild, ModelBuildContext
from .contracts import (
    Computation,
    Configuration,
    ModelDefinition,
    ObjectiveDefinition,
    ObjectiveEvaluation,
    Schema,
)
from .initialization import InitializationContext, ModelInitialization
from .optimizer import OptimizerDefinition
from .policies import PolicyDefinition, PolicyKind, ResolvedPolicy
from .preparation import PreparationDefinition, ResolvedPreparationDefinition
from .references import ModelReferenceDefinition
from .settings import constructor_config

if TYPE_CHECKING:
    from shadowspill.training.inputs.adapters import InputBinding, InputLayout
    from shadowspill.training.inputs.validation import ValidatedBatch


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
    settings: Model | None = None

    def describe_initialization(
        self, context: InitializationContext
    ) -> ModelInitialization:
        if context.model != self.settings:
            raise ValueError("initialization context differs from the resolved model")
        if self.definition.initialization is None:
            raise ValueError(
                "model definition has no bounded initialization hook; "
                "install one or restore a compatible checkpoint"
            )
        configured, _ = _configuration(self.definition.config, self.config)
        result = self.definition.initialization(configured, context)
        if not isinstance(result, ModelInitialization):
            raise TypeError("initialization hook must return ModelInitialization")
        return result

    def group_inputs(self, batch: ValidatedBatch) -> tuple[tuple[int, ...], ...]:
        """Resolve installed atomic grouping once, before packing/preprocessing."""
        from shadowspill.training.inputs.packing import atomic_groups

        if self.definition.batching == "independent":
            return atomic_groups(len(batch.fields), None)
        if self.definition.batching != "grouped":
            raise ValueError("automatic packing requires independent or grouped inputs")
        assert self.definition.input_groups is not None
        configured, _ = _configuration(self.definition.config, self.config)
        groups = self.definition.input_groups(configured, batch)
        if (
            not isinstance(groups, (tuple, list))
            or len(groups) > len(batch.fields)
            or any(
                not isinstance(group, (tuple, list)) or len(group) > len(batch.fields)
                for group in groups
            )
            or sum(len(group) for group in groups) != len(batch.fields)
        ):
            raise ValueError("input_groups must return bounded Datum index lists")
        return atomic_groups(len(batch.fields), groups)

    def create(self, *, build: ModelBuild | None = None) -> Any:
        """Construct only when requested by preparation, never during inspection."""
        configured, _ = _configuration(
            self.definition.config, self.config if build is None else build.config
        )
        model = self.definition.create(configured)
        if self.settings is not None and self.settings.trainable is not None:
            from shadowspill.pytorch.trainable import select_trainable

            select_trainable(model, self.settings.trainable)
        return model

    def describe_build(self, context: ModelBuildContext) -> ModelBuild:
        """Resolve geometry/provider descriptions without constructing a model."""
        configured, _ = _configuration(self.definition.config, self.config)
        description = (
            ModelBuild(self.config)
            if self.definition.build is None
            else self.definition.build(configured, context)
        )
        if not isinstance(description, ModelBuild):
            raise TypeError("model build hook must return ModelBuild")
        _, normalized = _configuration(self.definition.config, description.config)
        validate_resources(description.resources, context.parallelism)
        return replace(description, config=normalized)

    def capture_context(self, build: ModelBuild) -> AbstractContextManager[Any]:
        """Open provider precision/dispatch settings only in a capture worker.

        The installed factory owns restoration, including failed captures. Its
        typed configuration and source are already part of the saved identity.
        Description, worker execution and client runtime never enter this hook.
        """
        if self.definition.capture_context is None:
            return nullcontext()
        configured, _ = _configuration(self.definition.config, build.config)
        return self.definition.capture_context(configured)

    def default_parallelism(self, schema: InputSchema) -> Parallelism:
        if self.definition.default_parallelism is None:
            return Parallelism()
        configured, _ = _configuration(self.definition.config, self.config)
        result = self.definition.default_parallelism(configured, schema)
        if not isinstance(result, Parallelism):
            raise TypeError("model default_parallelism must return generic Parallelism")
        return result

    def bind_inputs(self, layout: InputLayout) -> InputBinding:
        """Resolve fixed views/metadata without constructing the model or CUDA."""
        from shadowspill.training.inputs.adapters import InputAdapter, InputBinding

        declared = {
            ("inputs", *field.path): replace(
                field,
                path=("inputs", *field.path),
                validity=None
                if field.validity is None
                else ("inputs", *field.validity),
            )
            for field in self.inputs.fields
        }
        supplied = {
            tuple(field.path): field
            for field in layout.schema.fields
            if field.path[0] == "inputs"
        }
        if declared != supplied:
            raise ValueError(
                "adapter layout differs from the resolved model input schema"
            )
        if self.definition.input_adapter is None:
            fields = {
                tuple(field.path): field
                for field in layout.schema.fields
                if tuple(field.path) in layout.buffer.fields
            }
            if self.definition.batching != "independent" or any(
                field.batching == "pack" or any(isinstance(n, str) for n in field.shape)
                for field in fields.values()
            ):
                raise ValueError(
                    "packed, variable or grouped inputs require a model input adapter"
                )
            adapter = InputAdapter(padding_safe=True)
        else:
            configured, _ = _configuration(self.definition.config, self.config)
            adapter = self.definition.input_adapter(configured, layout)
        return InputBinding(layout, adapter)


@dataclass(frozen=True)
class ResolvedObjective:
    definition: ObjectiveDefinition
    source: DefinitionSource
    config: Mapping[str, Value]
    inputs: InputSchema
    defaults: Mapping[str, Scalar]

    @property
    def controls(self) -> ControlSchema:
        assert isinstance(self.definition.controls, ControlSchema)
        return self.definition.controls

    @property
    def outputs(self) -> Mapping[str, OutputSet]:
        assert not callable(self.definition.outputs)
        return self.definition.outputs

    def evaluation(self, key: OperationKey) -> ObjectiveEvaluation | None:
        if self.definition.evaluate is None:
            return None
        configured, _ = _configuration(self.definition.config, self.config)
        result = self.definition.evaluate(configured, key)
        if not isinstance(result, ObjectiveEvaluation):
            raise TypeError("evaluate must return ObjectiveEvaluation")
        return result

    def reducer(self, key: OperationKey) -> Computation:
        if key.outputs not in self.outputs:
            raise ValueError(f"unknown output set: {key.outputs}")
        if self.definition.bind is None:
            return self.definition.compute
        configured, _ = _configuration(self.definition.config, self.config)
        computation = self.definition.bind(configured, key)
        if not callable(computation):
            raise TypeError("objective binding must return a callable")
        return computation

    def computation(self, key: OperationKey | None = None) -> Computation:
        key = key or OperationKey("step", self.definition.default_outputs)
        reducer = self.reducer(key)
        evaluation = self.evaluation(key)
        if evaluation is None:
            return reducer

        def compute(model: Any, inputs: Any, loss_inputs: Any, loss_params: Any) -> Any:
            values = evaluation.compute(model, inputs, loss_inputs, evaluation.requests)
            return reducer(values, inputs, loss_inputs, loss_params)

        return compute


@dataclass(frozen=True)
class ResolvedOptimizer:
    definition: OptimizerDefinition
    source: DefinitionSource
    config: Mapping[str, Value]
    defaults: Mapping[str, Scalar]
    settings: Optimizer | None = None

    @property
    def master_dtype(self) -> str | None:
        assert not callable(self.definition.master_dtype)
        return self.definition.master_dtype

    def create(self, model: Any, controls: Mapping[str, Any]) -> Any:
        if controls.keys() != self.definition.controls.fields.keys():
            raise ValueError(
                "provide exactly the optimizer's declared runtime controls"
            )
        configured, _ = _configuration(self.definition.config, self.config)
        return self.definition.create(model, configured, controls)


class Registry:
    """One explicit registry for supplied and external installed definitions.

    Startup code registers trusted callables with their immutable distribution
    provenance. Lookup uses exact IDs only, including on remote requests. No
    dynamic module names, uploaded Python or model-family branches are involved.
    """

    def __init__(self) -> None:
        from .policies import builtin_policy
        from .stacked import builtin_preparation

        self._models: dict[str, tuple[ModelDefinition, DefinitionSource]] = {}
        self._objectives: dict[str, tuple[ObjectiveDefinition, DefinitionSource]] = {}
        self._optimizers: dict[str, tuple[OptimizerDefinition, DefinitionSource]] = {}
        self._references: dict[
            str, tuple[ModelReferenceDefinition, DefinitionSource]
        ] = {}
        default = builtin_preparation()
        self._preparations = {default.definition.id: default}
        policy, source = builtin_policy()
        self._policies = {policy.id: (policy, source)}

    def add_model_reference(
        self, definition: ModelReferenceDefinition, *, source: DefinitionSource
    ) -> None:
        if not isinstance(definition, ModelReferenceDefinition) or not isinstance(
            source, DefinitionSource
        ):
            raise TypeError("register a ModelReferenceDefinition with DefinitionSource")
        if definition.id in self._references:
            raise ValueError(f"model reference already registered: {definition.id}")
        self._references[definition.id] = (definition, source)

    def model_reference(
        self, definition: str, *, source: DefinitionSource | None = None
    ) -> tuple[ModelReferenceDefinition, DefinitionSource]:
        try:
            resolver, installed = self._references[definition]
        except KeyError as error:
            raise ValueError(
                f"model reference definition is not installed: {definition}"
            ) from error
        if source is not None and source != installed:
            raise ValueError(f"model reference definition source changed: {definition}")
        return resolver, installed

    def resolve_model_reference(
        self, requested: Model
    ) -> tuple[Model, str, DefinitionSource]:
        weights = requested.initialize
        if weights is None or weights.kind != "pretrained" or weights.reference is None:
            raise ValueError("metadata resolution requires pretrained weights")
        matches = [
            (definition, source)
            for definition, source in self._references.values()
            if definition.accepts(weights.reference)
        ]
        if len(matches) != 1:
            raise ValueError(
                "pretrained reference requires one installed metadata resolver; "
                f"found {len(matches)}"
            )
        definition, installed = matches[0]
        return definition.pin(requested), definition.id, installed

    def add_policy(
        self, definition: PolicyDefinition, *, source: DefinitionSource
    ) -> None:
        if not isinstance(definition, PolicyDefinition) or not isinstance(
            source, DefinitionSource
        ):
            raise TypeError("register a PolicyDefinition with DefinitionSource")
        if definition.id in self._policies:
            raise ValueError(f"policy already registered: {definition.id}")
        self._policies[definition.id] = (definition, source)

    def resolve_policy(
        self,
        definition: str,
        *,
        kind: PolicyKind,
        config: Mapping[str, Value] | None = None,
        source: DefinitionSource | None = None,
    ) -> ResolvedPolicy:
        try:
            policy, installed = self._policies[definition]
        except KeyError as error:
            raise ValueError(
                f"policy definition is not installed: {definition}"
            ) from error
        if policy.kind != kind:
            raise ValueError(f"policy {definition} cannot configure {kind}")
        if source is not None and source != installed:
            raise ValueError(f"policy definition source changed: {definition}")
        _, settings = _configuration(policy.config, {} if config is None else config)
        return ResolvedPolicy(policy, installed, settings)

    def add_preparation(
        self, definition: PreparationDefinition, *, source: DefinitionSource
    ) -> None:
        if not isinstance(definition, PreparationDefinition) or not isinstance(
            source, DefinitionSource
        ):
            raise TypeError("register a PreparationDefinition with DefinitionSource")
        if definition.id in self._preparations:
            raise ValueError(f"preparation already registered: {definition.id}")
        self._preparations[definition.id] = ResolvedPreparationDefinition(
            definition, source
        )

    def resolve_preparation(
        self, definition: str, *, source: DefinitionSource | None = None
    ) -> ResolvedPreparationDefinition:
        try:
            prepared = self._preparations[definition]
        except KeyError as error:
            raise ValueError(
                f"preparation definition is not installed: {definition}"
            ) from error
        if source is not None and prepared.source != source:
            raise ValueError(f"preparation definition source changed: {definition}")
        return prepared

    def add_optimizer(
        self, definition: OptimizerDefinition, *, source: DefinitionSource
    ) -> None:
        if not isinstance(definition, OptimizerDefinition) or not isinstance(
            source, DefinitionSource
        ):
            raise TypeError("register an OptimizerDefinition with DefinitionSource")
        if definition.id in self._optimizers:
            raise ValueError(f"optimizer already registered: {definition.id}")
        self._optimizers[definition.id] = (definition, source)

    def resolve_optimizer(
        self,
        definition: str,
        *,
        config: Mapping[str, Value] | None = None,
        defaults: Mapping[str, Scalar] | None = None,
        source: DefinitionSource | None = None,
        settings: Optimizer | None = None,
    ) -> ResolvedOptimizer:
        try:
            optimizer, installed = self._optimizers[definition]
        except KeyError as error:
            raise ValueError(
                f"optimizer definition is not installed: {definition}"
            ) from error
        if source is not None and source != installed:
            raise ValueError(f"optimizer definition source changed: {definition}")
        if settings is not None:
            if (
                settings.definition != definition
                or config is not None
                or defaults is not None
            ):
                raise ValueError(
                    "optimizer settings cannot overlap another configuration"
                )
            config = constructor_config(settings, optimizer.configure)
            defaults = cast(Mapping[str, Scalar], settings.defaults)
        configured, values = _configuration(
            optimizer.config, {} if config is None else config
        )
        optimizer = replace(
            optimizer,
            master_dtype=optimizer.master_dtype(configured)
            if callable(optimizer.master_dtype)
            else optimizer.master_dtype,
        )
        controls = optimizer.controls.resolve(
            defaults=optimizer.defaults, overrides=defaults
        )
        if settings is not None:
            settings = replace(settings, config=values, defaults=controls)
            _, checked = _configuration(
                optimizer.config, constructor_config(settings, optimizer.configure)
            )
            if checked != values:
                raise ValueError(
                    "optimizer settings hook is not stable with resolved defaults"
                )
        return ResolvedOptimizer(optimizer, installed, values, controls, settings)

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
        if definition.id == "shadowspill/sum@1" or definition.id in self._objectives:
            raise ValueError(f"objective already registered: {definition.id}")
        self._objectives[definition.id] = (definition, source)

    def resolve_model(
        self,
        definition: str,
        *,
        config: Mapping[str, Value] | None = None,
        source: DefinitionSource | None = None,
        settings: Model | None = None,
    ) -> ResolvedModel:
        try:
            model, installed = self._models[definition]
        except KeyError as error:
            raise ValueError(
                f"model definition is not installed: {definition}"
            ) from error
        if source is not None and source != installed:
            raise ValueError(f"model definition source changed: {definition}")
        if settings is not None:
            if settings.definition != definition or config is not None:
                raise ValueError("model settings cannot overlap another configuration")
            config = constructor_config(settings, model.configure)
        configured, values = _configuration(
            model.config, {} if config is None else config
        )
        if settings is not None:
            settings = replace(settings, config=values)
            _, checked = _configuration(
                model.config, constructor_config(settings, model.configure)
            )
            if checked != values:
                raise ValueError(
                    "model settings hook is not stable with resolved defaults"
                )
        return ResolvedModel(
            model, installed, values, _schema(model.inputs, configured), settings
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
        if definition == "shadowspill/sum@1":
            from .composition import resolve_sum

            return resolve_sum(self, model, config, defaults, source)
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
        requested = {} if config is None else config
        configured, settings = _configuration(
            objective.config,
            requested
            if objective.configure is None
            else objective.configure(requested, model),
        )
        if objective.configure is not None:
            _, checked = _configuration(
                objective.config, objective.configure(settings, model)
            )
            if checked != settings:
                raise ValueError(
                    "objective settings hook is not stable with resolved defaults"
                )
        objective = replace(
            objective,
            padding_safe=objective.padding_safe(configured)
            if callable(objective.padding_safe)
            else objective.padding_safe,
            physical_batch_loss=objective.physical_batch_loss(configured)
            if callable(objective.physical_batch_loss)
            else objective.physical_batch_loss,
            controls=objective.controls(configured)
            if callable(objective.controls)
            else objective.controls,
            outputs=objective.outputs(configured)
            if callable(objective.outputs)
            else objective.outputs,
        )
        assert isinstance(objective.controls, ControlSchema)
        assert not callable(objective.outputs)
        inputs = _schema(objective.inputs, configured)
        model_paths = {tuple(field.path) for field in model.inputs.fields}
        loss_fields = {tuple(field.path): field for field in inputs.fields}
        for name, output in objective.outputs.items():
            for concrete in (output, output.forward):
                if concrete is not None and (
                    set(concrete.model_inputs) - model_paths
                    or set(concrete.loss_inputs) - loss_fields.keys()
                ):
                    raise ValueError(
                        f"output {name} references an undeclared input field"
                    )
        if objective.participation.path:
            field = loss_fields.get(tuple(objective.participation.path))
            if (
                field is None
                or field.dtype != "bool"
                or field.padding is not False
                or field.validity is not None
            ):
                raise ValueError(
                    "participation requires an unmasked boolean loss field "
                    "with False padding"
                )
        resolved_defaults = objective.controls.resolve(defaults=defaults)
        return ResolvedObjective(
            objective, installed, settings, inputs, resolved_defaults
        )
