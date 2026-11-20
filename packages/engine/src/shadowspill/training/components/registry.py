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
from shadowspill_client.protocol.components import (
    ComponentSource,
    OutputSet,
    Participation,
)
from shadowspill_client.protocol.controls import ControlSchema, Scalar
from shadowspill_client.protocol.inputs import InputSchema
from shadowspill_client.protocol.operations import OperationKey

from .build import ModelBuild, ModelBuildContext
from .contracts import (
    Computation,
    Configuration,
    ModelComponent,
    ObjectiveComponent,
    ObjectiveEvaluation,
    Schema,
)
from .initialization import InitializationContext, ModelInitialization
from .optimizer import OptimizerComponent
from .policies import PolicyComponent, PolicyKind, ResolvedPolicy
from .preparation import PreparationComponent, ResolvedPreparationComponent
from .references import ModelReferenceComponent
from .settings import constructor_config

if TYPE_CHECKING:
    from shadowspill.pytorch.capture.gradients import GradientDestinationRule
    from shadowspill.training.inputs.adapters import InputBinding, InputLayout
    from shadowspill_client.inputs.validation import ValidatedBatch


def _lookup[Component](
    components: Mapping[str, tuple[Component, ComponentSource]],
    name: str,
    kind: str,
) -> tuple[Component, ComponentSource]:
    """Resolve a unique public name or an exact ID in the installed registry."""
    if name in components:
        return components[name]
    matches = sorted(
        identity
        for identity in components
        if identity.rpartition("@")[0].rsplit("/", 1)[-1] == name
    )
    if not matches:
        raise ValueError(f"{kind} component is not installed: {name}")
    if len(matches) != 1:
        raise ValueError(
            f"ambiguous {kind} name {name!r}; select an exact component ID: "
            + ", ".join(matches)
        )
    return components[matches[0]]


def _configuration(
    factory: Configuration, values: Mapping[str, Value]
) -> tuple[Any, Mapping[str, Value]]:
    configured = factory(**plain(mapping(values, "component configuration")))
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
    component: ModelComponent
    source: ComponentSource
    config: Mapping[str, Value]
    inputs: InputSchema
    settings: Model | None = None

    def describe_initialization(
        self, context: InitializationContext
    ) -> ModelInitialization:
        if context.model != self.settings:
            raise ValueError("initialization context differs from the resolved model")
        hook = (
            self.component.initialization
            if context.temporary is None
            else self.component.temporary_initialization
        )
        if hook is None:
            if context.temporary is not None:
                raise ValueError(
                    "model component has no bounded temporary initialization hook"
                )
            raise ValueError(
                "model component has no bounded initialization hook; "
                "install one or restore a compatible checkpoint"
            )
        configured, _ = _configuration(self.component.config, self.config)
        result = hook(configured, context)
        if not isinstance(result, ModelInitialization):
            raise TypeError("initialization hook must return ModelInitialization")
        return result

    def group_inputs(self, batch: ValidatedBatch) -> tuple[tuple[int, ...], ...]:
        """Resolve installed atomic grouping once, before packing/preprocessing."""
        from shadowspill_client.inputs.packing import atomic_groups

        if self.component.batching == "independent":
            return atomic_groups(len(batch.fields), None)
        if self.component.batching != "grouped":
            raise ValueError("automatic packing requires independent or grouped inputs")
        assert self.component.input_groups is not None
        configured, _ = _configuration(self.component.config, self.config)
        groups = self.component.input_groups(configured, batch)
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
            self.component.config, self.config if build is None else build.config
        )
        model = self.component.create(configured)
        overrides = (
            {}
            if self.settings is None
            else self.settings.precision.get("parameter_dtypes", {})
        )
        if self.component.parameter_dtypes or overrides:
            from shadowspill.pytorch.precision import apply_parameter_dtypes
            from shadowspill_client.config.precision import parameter_dtypes

            apply_parameter_dtypes(
                model,
                defaults=self.component.parameter_dtypes,
                overrides=parameter_dtypes(overrides),
            )
        if self.settings is not None and self.settings.trainable is not None:
            from shadowspill.pytorch.trainable import select_trainable

            select_trainable(model, self.settings.trainable)
        return model

    def describe_build(self, context: ModelBuildContext) -> ModelBuild:
        """Resolve geometry/provider descriptions without constructing a model."""
        configured, _ = _configuration(self.component.config, self.config)
        description = (
            ModelBuild(self.config)
            if self.component.build is None
            else self.component.build(configured, context)
        )
        if not isinstance(description, ModelBuild):
            raise TypeError("model build hook must return ModelBuild")
        _, normalized = _configuration(self.component.config, description.config)
        validate_resources(description.resources, context.parallelism)
        return replace(
            description,
            config=normalized,
            grad_dtype=description.grad_dtype
            if context.gradient_dtype is None
            else None
            if context.gradient_dtype == "parameter"
            else context.gradient_dtype,
        )

    def capture_context(self, build: ModelBuild) -> AbstractContextManager[Any]:
        """Open provider precision/dispatch settings only in a capture worker.

        The installed factory owns restoration, including failed captures. Its
        typed configuration and source are already part of the saved identity.
        Description, worker execution and client runtime never enter this hook.
        """
        if self.component.capture_context is None:
            return nullcontext()
        configured, _ = _configuration(self.component.config, build.config)
        return self.component.capture_context(configured)

    def gradient_destination_rules(
        self, build: ModelBuild
    ) -> tuple[GradientDestinationRule, ...]:
        """Resolve existing compiler contracts only in a training capture worker.

        Configuration and installed component source pin the factory. Compiled
        operations retain the resulting mutations, with no runtime callback.
        """
        if self.component.gradient_destination_rules is None:
            return ()
        from shadowspill.pytorch.capture.gradients.destinations import (
            normalize_gradient_destination_rules,
        )

        configured, _ = _configuration(self.component.config, build.config)
        return normalize_gradient_destination_rules(
            self.component.gradient_destination_rules(configured)
        )

    def default_parallelism(self, schema: InputSchema) -> Parallelism:
        if self.component.default_parallelism is None:
            return Parallelism()
        configured, _ = _configuration(self.component.config, self.config)
        result = self.component.default_parallelism(configured, schema)
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
        if self.component.input_adapter is None:
            fields = {
                tuple(field.path): field
                for field in layout.schema.fields
                if tuple(field.path) in layout.buffer.fields
            }
            if self.component.batching != "independent" or any(
                field.batching == "pack" or any(isinstance(n, str) for n in field.shape)
                for field in fields.values()
            ):
                raise ValueError(
                    "packed, variable or grouped inputs require a model input adapter"
                )
            adapter = InputAdapter(padding_safe=True)
        else:
            configured, _ = _configuration(self.component.config, self.config)
            adapter = self.component.input_adapter(configured, layout)
        return InputBinding(layout, adapter)


@dataclass(frozen=True)
class ResolvedObjective:
    component: ObjectiveComponent
    source: ComponentSource
    config: Mapping[str, Value]
    inputs: InputSchema
    defaults: Mapping[str, Scalar]

    @property
    def participation(self) -> Participation:
        assert isinstance(self.component.participation, Participation)
        return self.component.participation

    @property
    def controls(self) -> ControlSchema:
        assert isinstance(self.component.controls, ControlSchema)
        return self.component.controls

    @property
    def outputs(self) -> Mapping[str, OutputSet]:
        assert not callable(self.component.outputs)
        return self.component.outputs

    def evaluation(self, key: OperationKey) -> ObjectiveEvaluation | None:
        if self.component.evaluate is None:
            return None
        configured, _ = _configuration(self.component.config, self.config)
        result = self.component.evaluate(configured, key)
        if not isinstance(result, ObjectiveEvaluation):
            raise TypeError("evaluate must return ObjectiveEvaluation")
        return result

    def reducer(self, key: OperationKey) -> Computation:
        if key.outputs not in self.outputs:
            raise ValueError(f"unknown output set: {key.outputs}")
        if self.component.bind is None:
            return self.component.compute
        configured, _ = _configuration(self.component.config, self.config)
        computation = self.component.bind(configured, key)
        if not callable(computation):
            raise TypeError("objective binding must return a callable")
        return computation

    def computation(self, key: OperationKey | None = None) -> Computation:
        key = key or OperationKey("step", self.component.default_outputs)
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
    component: OptimizerComponent
    source: ComponentSource
    config: Mapping[str, Value]
    defaults: Mapping[str, Scalar]
    settings: Optimizer | None = None

    @property
    def master_dtype(self) -> str | None:
        assert not callable(self.component.master_dtype)
        return self.component.master_dtype

    @property
    def gradient_dtype(self) -> str | None:
        assert not callable(self.component.gradient_dtype)
        return self.component.gradient_dtype

    def create(self, model: Any, controls: Mapping[str, Any]) -> Any:
        if controls.keys() != self.component.controls.fields.keys():
            raise ValueError(
                "provide exactly the optimizer's declared runtime controls"
            )
        configured, _ = _configuration(self.component.config, self.config)
        return self.component.create(model, configured, controls)


class Registry:
    """One explicit registry for supplied and external installed components.

    Startup code registers trusted callables with their immutable distribution
    provenance. Model, objective and optimizer requests accept unique short names
    or exact IDs. Resolved settings retain exact IDs for saved Plans and state.
    No dynamic module names, uploaded Python or model-family branches are involved.
    """

    def __init__(self) -> None:
        from .policies import builtin_policy
        from .stacked import builtin_preparation

        self._models: dict[str, tuple[ModelComponent, ComponentSource]] = {}
        self._objectives: dict[str, tuple[ObjectiveComponent, ComponentSource]] = {}
        self._optimizers: dict[str, tuple[OptimizerComponent, ComponentSource]] = {}
        self._references: dict[
            str, tuple[ModelReferenceComponent, ComponentSource]
        ] = {}
        default = builtin_preparation()
        self._preparations = {default.component.id: default}
        policy, source = builtin_policy()
        self._policies = {policy.id: (policy, source)}

    def add_model_reference(
        self, component: ModelReferenceComponent, *, source: ComponentSource
    ) -> None:
        if not isinstance(component, ModelReferenceComponent) or not isinstance(
            source, ComponentSource
        ):
            raise TypeError("register a ModelReferenceComponent with ComponentSource")
        if component.id in self._references:
            raise ValueError(f"model reference already registered: {component.id}")
        self._references[component.id] = (component, source)

    def model_reference(
        self, name: str, *, source: ComponentSource | None = None
    ) -> tuple[ModelReferenceComponent, ComponentSource]:
        try:
            resolver, installed = self._references[name]
        except KeyError as error:
            raise ValueError(
                f"model reference component is not installed: {name}"
            ) from error
        if source is not None and source != installed:
            raise ValueError(f"model reference component source changed: {name}")
        return resolver, installed

    def resolve_model_reference(
        self, requested: Model
    ) -> tuple[Model, str, ComponentSource]:
        weights = requested.initialize
        if weights is None or weights.kind != "pretrained" or weights.reference is None:
            raise ValueError("metadata resolution requires pretrained weights")
        if requested.name is not None:
            model, _ = _lookup(self._models, requested.name, "model")
            requested = replace(requested, name=model.id)
        matches = [
            (component, source)
            for component, source in self._references.values()
            if component.accepts(weights.reference)
        ]
        if len(matches) != 1:
            raise ValueError(
                "pretrained reference requires one installed metadata resolver; "
                f"found {len(matches)}"
            )
        component, installed = matches[0]
        return component.pin(requested), component.id, installed

    def add_policy(
        self, component: PolicyComponent, *, source: ComponentSource
    ) -> None:
        if not isinstance(component, PolicyComponent) or not isinstance(
            source, ComponentSource
        ):
            raise TypeError("register a PolicyComponent with ComponentSource")
        if component.id in self._policies:
            raise ValueError(f"policy already registered: {component.id}")
        self._policies[component.id] = (component, source)

    def resolve_policy(
        self,
        name: str,
        *,
        kind: PolicyKind,
        config: Mapping[str, Value] | None = None,
        source: ComponentSource | None = None,
    ) -> ResolvedPolicy:
        try:
            policy, installed = self._policies[name]
        except KeyError as error:
            raise ValueError(f"policy component is not installed: {name}") from error
        if policy.kind != kind:
            raise ValueError(f"policy {name} cannot configure {kind}")
        if source is not None and source != installed:
            raise ValueError(f"policy component source changed: {name}")
        _, settings = _configuration(policy.config, {} if config is None else config)
        return ResolvedPolicy(policy, installed, settings)

    def add_preparation(
        self, component: PreparationComponent, *, source: ComponentSource
    ) -> None:
        if not isinstance(component, PreparationComponent) or not isinstance(
            source, ComponentSource
        ):
            raise TypeError("register a PreparationComponent with ComponentSource")
        if component.id in self._preparations:
            raise ValueError(f"preparation already registered: {component.id}")
        self._preparations[component.id] = ResolvedPreparationComponent(
            component, source
        )

    def resolve_preparation(
        self, name: str, *, source: ComponentSource | None = None
    ) -> ResolvedPreparationComponent:
        try:
            prepared = self._preparations[name]
        except KeyError as error:
            raise ValueError(
                f"preparation component is not installed: {name}"
            ) from error
        if source is not None and prepared.source != source:
            raise ValueError(f"preparation component source changed: {name}")
        return prepared

    def add_optimizer(
        self, component: OptimizerComponent, *, source: ComponentSource
    ) -> None:
        if not isinstance(component, OptimizerComponent) or not isinstance(
            source, ComponentSource
        ):
            raise TypeError("register an OptimizerComponent with ComponentSource")
        if component.id in self._optimizers:
            raise ValueError(f"optimizer already registered: {component.id}")
        self._optimizers[component.id] = (component, source)

    def resolve_optimizer(
        self,
        name: str,
        *,
        config: Mapping[str, Value] | None = None,
        defaults: Mapping[str, Scalar] | None = None,
        source: ComponentSource | None = None,
        settings: Optimizer | None = None,
    ) -> ResolvedOptimizer:
        optimizer, installed = _lookup(self._optimizers, name, "optimizer")
        if source is not None and source != installed:
            raise ValueError(f"optimizer component source changed: {name}")
        if settings is not None:
            if settings.name != name or config is not None or defaults is not None:
                raise ValueError(
                    "optimizer settings cannot overlap another configuration"
                )
            settings = replace(settings, name=optimizer.id)
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
            gradient_dtype=optimizer.gradient_dtype(configured)
            if callable(optimizer.gradient_dtype)
            else optimizer.gradient_dtype,
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

    def add_model(self, component: ModelComponent, *, source: ComponentSource) -> None:
        if not isinstance(component, ModelComponent) or not isinstance(
            source, ComponentSource
        ):
            raise TypeError("register a ModelComponent with ComponentSource")
        if component.id in self._models:
            raise ValueError(f"model already registered: {component.id}")
        self._models[component.id] = (component, source)

    def add_objective(
        self, component: ObjectiveComponent, *, source: ComponentSource
    ) -> None:
        if not isinstance(component, ObjectiveComponent) or not isinstance(
            source, ComponentSource
        ):
            raise TypeError("register an ObjectiveComponent with ComponentSource")
        if component.id == "shadowspill/sum@1" or component.id in self._objectives:
            raise ValueError(f"objective already registered: {component.id}")
        self._objectives[component.id] = (component, source)

    def resolve_model(
        self,
        name: str,
        *,
        config: Mapping[str, Value] | None = None,
        source: ComponentSource | None = None,
        settings: Model | None = None,
    ) -> ResolvedModel:
        model, installed = _lookup(self._models, name, "model")
        if source is not None and source != installed:
            raise ValueError(f"model component source changed: {name}")
        if settings is not None:
            if settings.name != name or config is not None:
                raise ValueError("model settings cannot overlap another configuration")
            settings = replace(settings, name=model.id)
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
        name: str,
        *,
        model: ResolvedModel,
        config: Mapping[str, Value] | None = None,
        defaults: Mapping[str, Scalar] | None = None,
        source: ComponentSource | None = None,
    ) -> ResolvedObjective:
        if name == "shadowspill/sum@1":
            from .composition import resolve_sum

            return resolve_sum(self, model, config, defaults, source)
        objective, installed = _lookup(self._objectives, name, "objective")
        if source is not None and source != installed:
            raise ValueError(f"objective component source changed: {name}")
        if missing := set(objective.requires) - set(model.component.capabilities):
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
            participation=objective.participation(configured)
            if callable(objective.participation)
            else objective.participation,
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
        if not isinstance(objective.participation, Participation) or any(
            type(flag) is not bool
            for flag in (objective.padding_safe, objective.physical_batch_loss)
        ):
            raise TypeError(
                "objective guarantees must resolve to participation and boolean flags"
            )
        inputs = _schema(objective.inputs, configured)
        model_paths = {tuple(field.path) for field in model.inputs.fields}
        loss_fields = {tuple(field.path): field for field in inputs.fields}
        for output_name, output in objective.outputs.items():
            for concrete in (output, output.forward):
                if concrete is not None and (
                    set(concrete.model_inputs) - model_paths
                    or set(concrete.loss_inputs) - loss_fields.keys()
                ):
                    raise ValueError(
                        f"output {output_name} references an undeclared input field"
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
