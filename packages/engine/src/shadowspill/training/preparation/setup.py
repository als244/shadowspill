"""One CPU setup resolver shared by optional inspection and training creation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

from shadowspill.parallelism import InputFieldPlacement, InputPlacement, Parallelism
from shadowspill.training.definitions.preparation import (
    PreparationContext,
    ResolvedPreparationDefinition,
)
from shadowspill.training.definitions.registry import (
    Registry,
    ResolvedModel,
    ResolvedObjective,
    ResolvedOptimizer,
)
from shadowspill.training.definitions.stacked import STACKED_PRESET
from shadowspill.training.inputs.staging import input_envelope
from shadowspill_client import (
    BatchConfig,
    Execution,
    Memory,
    Model,
    Objective,
    Operation,
    Optimizer,
    Preparation,
)
from shadowspill_client._records import plain
from shadowspill_client.inputs.validation import combined_schema
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.controls import Scalar
from shadowspill_client.protocol.definitions import DefinitionSource
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_client.protocol.preview import PreparationEstimate, TrainingPreview
from shadowspill_client.protocol.setup import TrainingSetup
from shadowspill_client.protocol.training import Origin, ResolvedTraining

from .expansion import ExpandedPreparation, expand_schedules
from .identity import setup_sources
from .operations import OperationContract, resolve_operations
from .policies import resolve_build_policies
from .resolution import _search, resolve_preparation


@dataclass(frozen=True)
class ResolvedSetup:
    config: ResolvedTraining
    context: PreparationContext
    objective: ResolvedObjective
    optimizer: ResolvedOptimizer | None
    expansions: Mapping[OperationKey, ExpandedPreparation]
    preset: ResolvedPreparationDefinition

    @property
    def model(self) -> ResolvedModel:
        return self.context.model

    def preview(self) -> TrainingPreview:
        return TrainingPreview(
            self.config,
            tuple(
                PreparationEstimate(
                    key, value.build_cases, len(value.candidates), len(value.rejected)
                )
                for key, value in self.expansions.items()
            ),
        )


def _assemble(
    registry: Registry,
    config: ResolvedTraining,
    model: ResolvedModel,
    objective: ResolvedObjective,
    optimizer: ResolvedOptimizer | None,
    parallelism: Parallelism,
) -> ResolvedSetup:
    schema = combined_schema(model.inputs, objective.inputs)
    context = PreparationContext(model, schema, tuple(config.batches), parallelism)
    if (
        schema != config.inputs
        or objective.controls != config.loss_controls
        or (None if optimizer is None else optimizer.definition.controls)
        != config.optimizer_controls
        or objective.participation != config.participation
        or {name: objective.outputs.get(name) for name in config.outputs}
        != config.outputs
    ):
        raise ValueError(
            "saved input/control/output contract differs from installed definitions"
        )
    selected: dict[str, list[Operation]] = {}
    for key in config.operations:
        if key.operation != "optim_step":
            assert key.outputs is not None
            selected.setdefault(key.operation, []).append(
                Operation(key.outputs, key.mode)
            )
    if (
        resolve_operations(objective, selected, optimizer=optimizer is not None)
        != config.operations
    ):
        raise ValueError("saved operation defaults/order or dependencies differ")
    preset = registry.resolve_preparation(
        config.preparation.preset, source=config.preparation.source
    )
    _search(config.preparation.planning.search)
    prepared, _ = resolve_build_policies(
        registry, config.preparation, sources=config.policy_sources
    )
    if prepared != config.preparation:
        raise ValueError("saved preparation settings are not fully resolved")
    # Bound the whole operation cross-product, not a separate allowance per
    # output set. Geometry alternatives remain distinct; no profiling or memory
    # admission is implied by these CPU descriptions.
    counts = config.preparation.planning.microbatch_counts
    assert counts is not None
    combinations = sum(key.operation != "optim_step" for key in config.operations)
    total = (
        combinations * len(config.batches) * len(config.preparation.cases) * len(counts)
    )
    if total > 4096:
        raise ValueError("operation/schedule expansion exceeds configured limit 4096")
    expansions = {}
    for key in config.operations:
        if key.operation == "optim_step":
            continue
        contract = OperationContract(key, objective)
        for batch in config.batches:
            input_envelope(schema, batch, contract.required_paths)
        expanded = expand_schedules(
            config.preparation, preset, context, contract.required_paths
        )
        if {item.batch for item in expanded.candidates} != set(
            range(len(config.batches))
        ):
            raise ValueError(
                f"operation {key.operation}/{key.outputs} has a batch option "
                "with no eligible scheduled capacities"
            )
        expansions[key] = expanded
    # A successful description must itself fit the bounded portable envelope.
    encode(config)
    return ResolvedSetup(
        config, context, objective, optimizer, MappingProxyType(expansions), preset
    )


def _describe(
    registry: Registry,
    *,
    model: Model,
    objective: Objective,
    batch: BatchConfig | Sequence[BatchConfig],
    memory: Memory,
    optimizer: Optimizer | None = None,
    parallelism: Parallelism | None = None,
    execution: Execution | None = None,
    operations: Mapping[str, Sequence[str | Operation]] | None = None,
    prepare: Preparation | None = None,
) -> ResolvedSetup:
    for name, value, cls in (
        ("model", model, Model),
        ("objective", objective, Objective),
        ("memory", memory, Memory),
    ):
        if not isinstance(value, cls):
            raise TypeError(f"{name} requires {cls.__name__}")
    for name, optional, optional_type in (
        ("optimizer", optimizer, Optimizer),
        ("execution", execution, Execution),
        ("parallelism", parallelism, Parallelism),
        ("prepare", prepare, Preparation),
    ):
        if optional is not None and not isinstance(optional, optional_type):
            raise TypeError(f"{name} requires {optional_type.__name__}")
    batches = (batch,) if isinstance(batch, BatchConfig) else tuple(batch)
    if not batches or any(not isinstance(item, BatchConfig) for item in batches):
        raise ValueError("batch requires BatchConfig or a nonempty list of options")
    origins: dict[str, Origin] = {
        "model": "explicit",
        "objective": "explicit",
        "batch": "explicit",
        "memory": "explicit",
        "optimizer": "explicit" if optimizer is not None else "engine_default",
        "execution": "explicit" if execution is not None else "engine_default",
        "parallelism": "explicit" if parallelism is not None else "engine_default",
        "operations": "explicit" if operations is not None else "definition",
        "model.initialize": "explicit"
        if model.initialize is not None
        else "engine_default",
    }
    execution = execution or Execution()
    requested_model = model
    for name in ("definition", "lora", "implementation", "trainable"):
        origins[f"model.{name}"] = (
            "explicit" if getattr(model, name) is not None else "definition"
        )
    sources: dict[str, DefinitionSource] = {}
    reference_definition = None
    if model.initialize is not None and model.initialize.kind == "pretrained":
        model, reference_definition, sources["reference"] = (
            registry.resolve_model_reference(model)
        )
    assert model.definition is not None
    resolved_model = registry.resolve_model(model.definition, settings=model)
    assert resolved_model.settings is not None
    sources["model"] = resolved_model.source
    for name in model.precision:
        origins[f"model.precision.{name}"] = (
            "explicit" if name in requested_model.precision else "definition"
        )
    resolved_objective = registry.resolve_objective(
        objective.definition,
        model=resolved_model,
        config=objective.config,
        defaults=cast(Mapping[str, Scalar], objective.defaults),
    )
    sources["objective"] = resolved_objective.source
    resolved_optimizer = (
        None
        if optimizer is None
        else registry.resolve_optimizer(optimizer.definition, settings=optimizer)
    )
    if resolved_optimizer is not None:
        sources["optimizer"] = resolved_optimizer.source
    for prefix, supplied, resolved in (
        ("model.config", requested_model.config, resolved_model.config),
        ("objective.config", objective.config, resolved_objective.config),
        ("objective.defaults", objective.defaults, resolved_objective.defaults),
    ):
        for name in resolved:
            origins[f"{prefix}.{name}"] = (
                "explicit" if name in supplied else "definition"
            )
    if optimizer is not None and resolved_optimizer is not None:
        for name in ("config", "defaults"):
            for key in getattr(resolved_optimizer, name):
                origins[f"optimizer.{name}.{key}"] = (
                    "explicit" if key in getattr(optimizer, name) else "definition"
                )
    schema = combined_schema(resolved_model.inputs, resolved_objective.inputs)
    if parallelism is None:
        parallelism = resolved_model.default_parallelism(schema)
        if resolved_model.definition.default_parallelism is not None:
            origins["parallelism"] = "definition"
    if parallelism.inputs is None:
        if parallelism.world_size != 1:
            raise ValueError(
                "multiple workers require explicit generic input placement"
            )
        parallelism = replace(
            parallelism,
            inputs=InputPlacement.independent(
                1, fields=[InputFieldPlacement(tuple(f.path)) for f in schema.fields]
            ),
        )
        origins["parallelism.inputs"] = "engine_default"
    else:
        origins["parallelism.inputs"] = origins["parallelism"]
        if not parallelism.inputs.fields:
            if any(len(group) != 1 for group in parallelism.inputs.partitions.groups):
                raise ValueError(
                    "cooperating input workers require explicit field placement"
                )
            parallelism = replace(
                parallelism,
                inputs=replace(
                    parallelism.inputs,
                    fields=[InputFieldPlacement(tuple(f.path)) for f in schema.fields],
                ),
            )
            origins["parallelism.inputs.fields"] = "engine_default"
    context = PreparationContext(resolved_model, schema, batches, parallelism)
    preset_name = (
        prepare.preset
        if prepare is not None and prepare.preset is not None
        else resolved_model.definition.preparation or STACKED_PRESET
    )
    preset = registry.resolve_preparation(preset_name)
    origins["preparation.preset"] = (
        "explicit"
        if prepare is not None and prepare.preset is not None
        else "definition"
        if resolved_model.definition.preparation is not None
        else "engine_default"
    )
    preparation, policies = resolve_build_policies(
        registry, resolve_preparation(preset, context, prepare)
    )
    origins.update(
        {f"preparation.{name}": origin for name, origin in preparation.origins.items()}
    )
    keys = resolve_operations(
        resolved_objective, operations, optimizer=optimizer is not None
    )
    engine_source, client_source = setup_sources()
    config = ResolvedTraining(
        model=resolved_model.settings,
        objective=replace(
            objective,
            definition=resolved_objective.definition.id,
            config=resolved_objective.config,
            defaults=resolved_objective.defaults,
        ),
        optimizer=None if resolved_optimizer is None else resolved_optimizer.settings,
        batches=batches,
        memory=memory,
        parallelism=parallelism.to_dict(),
        execution=execution,
        operations=keys,
        preparation=preparation,
        definition_sources=sources,
        policy_sources=policies,
        engine_source=engine_source,
        client_source=client_source,
        inputs=schema,
        loss_controls=resolved_objective.controls,
        optimizer_controls=None
        if resolved_optimizer is None
        else resolved_optimizer.definition.controls,
        outputs={
            key.outputs: resolved_objective.outputs[key.outputs]
            for key in keys
            if key.outputs is not None
        },
        participation=resolved_objective.participation,
        origins=origins,
        reference_definition=reference_definition,
    )
    return _assemble(
        registry,
        config,
        resolved_model,
        resolved_objective,
        resolved_optimizer,
        parallelism,
    )


def _recheck(registry: Registry, config: ResolvedTraining) -> ResolvedSetup:
    if (config.engine_source, config.client_source) != setup_sources():
        raise ValueError("saved setup engine/client source identity changed")
    sources = config.definition_sources
    if config.reference_definition is not None:
        definition, _ = registry.model_reference(
            config.reference_definition, source=sources["reference"]
        )
        if definition.pin(config.model) != config.model:
            raise ValueError("saved pretrained metadata/revision changed")
    assert config.model.definition is not None
    model = registry.resolve_model(
        config.model.definition, settings=config.model, source=sources["model"]
    )
    objective = registry.resolve_objective(
        config.objective.definition,
        model=model,
        config=config.objective.config,
        defaults=cast(Mapping[str, Scalar], config.objective.defaults),
        source=sources["objective"],
    )
    optimizer = (
        None
        if config.optimizer is None
        else registry.resolve_optimizer(
            config.optimizer.definition,
            settings=config.optimizer,
            source=sources["optimizer"],
        )
    )
    if (
        model.settings != config.model
        or objective.definition.id != config.objective.definition
        or objective.config != config.objective.config
        or objective.defaults != config.objective.defaults
        or (None if optimizer is None else optimizer.settings) != config.optimizer
    ):
        raise ValueError(
            "saved model/objective/optimizer defaults are not fully resolved"
        )
    parallelism = Parallelism.from_dict(plain(config.parallelism))
    if parallelism.to_dict() != plain(config.parallelism) or parallelism.inputs is None:
        raise ValueError("saved generic parallelism is not fully resolved")
    return _assemble(registry, config, model, objective, optimizer, parallelism)


def resolve_training(
    registry: Registry,
    settings: TrainingSetup | ResolvedTraining | None = None,
    **setup: Any,
) -> ResolvedSetup:
    """Describe keyword settings or recheck exactly one resolved snapshot.

    The same entry point is used by creation. No workers, model factories,
    runtime allocation, capture, compilation, profiling or search run here.
    A preview is neither a resource reservation nor a readiness certificate.
    """
    if settings is not None:
        if setup:
            raise ValueError("portable settings cannot overlap setup keywords")
        if isinstance(settings, ResolvedTraining):
            setup = {"config": settings}
        elif isinstance(settings, TrainingSetup):
            setup = {
                "model": settings.model,
                "objective": settings.objective,
                "batch": settings.batch,
                "memory": settings.memory,
                "optimizer": settings.optimizer,
                "execution": settings.execution,
                "operations": settings.operations,
                "prepare": settings.prepare,
                "parallelism": None
                if settings.parallelism is None
                else Parallelism.from_dict(plain(settings.parallelism)),
            }
        else:
            raise TypeError("settings requires TrainingSetup or ResolvedTraining")
    if "config" in setup:
        config = setup.pop("config")
        if setup:
            raise ValueError("config cannot overlap setup keywords")
        if not isinstance(config, ResolvedTraining):
            raise TypeError("config requires a ResolvedTraining snapshot")
        return _recheck(registry, config)
    return _describe(registry, **setup)
