"""Inspection freezes one portable setup; creation rechecks it without new defaults."""

import subprocess
import sys
from dataclasses import dataclass, replace

import pytest

from engine_tests.fixtures.definition_operations import definitions
from engine_tests.fixtures.external_definition import SOURCE
from shadowspill.parallelism import (
    GroupRef,
    InputFieldPlacement,
    InputPlacement,
    Parallelism,
)
from shadowspill.training.definitions import ModelReferenceDefinition, Registry
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    Execution,
    LoRA,
    Memory,
    Model,
    Objective,
    Operation,
    Optimizer,
    PlanningOptions,
    Preparation,
    Weights,
)
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.operations import OperationKey


def installed(*, model=None, objective=None, optimizer=None, model_source=SOURCE):
    m, o, p = definitions()
    registry = Registry()
    registry.add_model(model or m.definition, source=model_source)
    registry.add_objective(objective or o.definition, source=o.source)
    registry.add_optimizer(optimizer or p.definition, source=p.source)
    return registry


def arguments():
    return dict(
        model=Model("external/image_classifier@1", initialize=Weights.random(seed=17)),
        objective=Objective("external/classification_outputs@1"),
        optimizer=Optimizer("external/operation_sgd@1"),
        batch=BatchConfig(limits={"max_examples": 4}),
        memory=Memory(device_gib=2, spill_gib=1),
    )


def test_inspection_and_creation_use_one_snapshot_without_model_construction():
    m, _, p = definitions()

    def no_model(*args, **kwargs):
        pytest.fail("constructed a model/optimizer during description")

    registry = installed(
        model=replace(m.definition, create=no_model),
        optimizer=replace(p.definition, create=no_model),
    )
    requested = arguments()
    requested.update(
        operations={
            "forward_backward": ["metrics"],
            "forward": ["predictions", Operation("metrics", "training")],
        },
        execution=Execution(seed=31),
    )
    preview = resolve_training(registry, **requested)
    config = preview.config
    assert config.model.config == {"num_classes": 3}
    assert config.model.initialize == Weights.random(seed=17)
    assert config.optimizer.defaults == {"lr": 0.01}
    assert config.objective.defaults == {"temperature": 1.0}
    assert config.operations == (
        OperationKey("forward_backward", "metrics"),
        OperationKey("forward", "predictions", "evaluation"),
        OperationKey("forward", "metrics", "training"),
        OperationKey("optim_step", None),
    )
    assert config.origins["model.config.num_classes"] == "definition"
    assert config.origins["model.initialize"] == "explicit"
    assert config.origins["execution"] == "explicit"
    assert config.origins["operations"] == "explicit"
    assert len(preview.expansions) == 3
    for key, expanded in preview.expansions.items():
        assert len(expanded.candidates) == 5 and len(expanded.rejected) == 1
        assert expanded.coverage(preview.context)[0].max_datums == 4
        if key.outputs == "predictions":
            assert set(expanded.candidates[0].schedule.buffers[0].fields) == {
                ("inputs", "pixels")
            }
    requested["operations"]["forward"].clear()
    assert len(config.operations) == 4
    wire = encode(config)
    restored = resolve_training(registry, config=decode(wire))
    assert restored.config == config and encode(restored.config) == wire
    assert restored.expansions == preview.expansions
    with pytest.raises(TypeError):
        config.model.config["num_classes"] = 5
    with pytest.raises(ValueError, match="overlap"):
        resolve_training(registry, config=config, optimizer=None)


def test_changed_sources_missing_defaults_and_contracts_fail_recheck():
    registry = installed()
    config = resolve_training(registry, **arguments()).config
    with pytest.raises(ValueError, match="source changed"):
        resolve_training(
            installed(model_source=replace(SOURCE, digest="b" * 64)), config=config
        )
    with pytest.raises(ValueError, match="source identity changed"):
        resolve_training(
            registry,
            config=replace(
                config, engine_source=replace(config.engine_source, digest="c" * 64)
            ),
        )
    for changed in (
        replace(config, model=replace(config.model, config={})),
        replace(config, optimizer=replace(config.optimizer, defaults={})),
        replace(config, objective=replace(config.objective, defaults={})),
    ):
        with pytest.raises(ValueError, match="defaults are not fully resolved"):
            resolve_training(registry, config=changed)
    with pytest.raises(ValueError, match="missing policy source"):
        resolve_training(registry, config=replace(config, policy_sources={}))
    m, _, _ = definitions()
    schema = replace(m.inputs, fields=[replace(m.inputs.fields[0], dtype="float64")])
    with pytest.raises(ValueError, match="contract differs"):
        resolve_training(
            installed(model=replace(m.definition, inputs=schema)), config=config
        )
    parallelism = dict(config.parallelism)
    parallelism.pop("inputs")
    with pytest.raises(ValueError, match="parallelism is not fully resolved"):
        resolve_training(registry, config=replace(config, parallelism=parallelism))


def test_every_requested_batch_is_retained_and_multiworker_placement_is_explicit():
    args = arguments()
    args["batch"] = [BatchConfig(limits={"max_examples": 3}), args["batch"]]
    prepared = resolve_training(installed(), **args)
    assert [b.limits["max_examples"] for b in prepared.config.batches] == [3, 4]
    assert [
        p.max_datums
        for p in next(iter(prepared.expansions.values())).coverage(prepared.context)
    ] == [3, 4]
    with pytest.raises(ValueError, match="no eligible scheduled capacities"):
        resolve_training(
            installed(),
            **args,
            prepare=Preparation(
                candidates=BuildCandidates(microbatch_sizes=[4]),
                planning=PlanningOptions(microbatch_counts=[1]),
            ),
        )
    with pytest.raises(ValueError, match="explicit generic input placement"):
        resolve_training(
            installed(), **args, parallelism=Parallelism(ranks_per_replica=2)
        )
    fields = [InputFieldPlacement(tuple(f.path)) for f in prepared.config.inputs.fields]
    actual = Parallelism(
        ranks_per_replica=2, inputs=InputPlacement.independent(2, fields=fields)
    )
    result = resolve_training(installed(), **arguments(), parallelism=actual)
    assert result.context.parallelism == actual
    assert (
        resolve_training(installed(), config=result.config).context.parallelism
        == actual
    )
    with pytest.raises(ValueError, match="timing hints"):
        resolve_training(
            installed(),
            **{
                **arguments(),
                "batch": BatchConfig(limits={"max_examples": 4}, typical={"typo": 1}),
            },
        )


def test_pretrained_metadata_pins_revision_preserves_overrides_and_rechecks_that_pin():
    registry = installed()
    revisions = []

    def metadata(request):
        revisions.append(request.initialize.revision)
        return replace(
            request,
            definition="external/image_classifier@1",
            config={"num_classes": 3, **request.config},
            initialize=Weights.pretrained(
                request.initialize.reference,
                revision=request.initialize.revision or "b" * 40,
            ),
        )

    resolver = ModelReferenceDefinition(
        "external/metadata@1", lambda ref: ref.startswith("fixture://"), metadata
    )
    registry.add_model_reference(resolver, source=SOURCE)
    args = {
        **arguments(),
        "model": Model.from_pretrained(
            "fixture://classifier", config={"num_classes": 5}
        ),
    }
    prepared = resolve_training(registry, **args)
    assert prepared.config.model.config == {"num_classes": 5}
    assert prepared.config.model.initialize.revision == "b" * 40
    assert prepared.config.reference_definition == resolver.id
    assert resolve_training(registry, config=prepared.config).config == prepared.config
    assert revisions == [None, "b" * 40]
    missing = installed()
    with pytest.raises(ValueError, match="metadata resolver; found 0"):
        resolve_training(missing, **args)
    with pytest.raises(ValueError, match="reference definition is not installed"):
        resolve_training(missing, config=prepared.config)
    registry.add_model_reference(
        replace(resolver, id="external/ambiguous@1"), source=SOURCE
    )
    with pytest.raises(ValueError, match="metadata resolver; found 2"):
        resolve_training(registry, **args)
    bad = installed()
    bad.add_model_reference(
        replace(
            resolver, resolve=lambda m: replace(metadata(m), config={"num_classes": 3})
        ),
        source=SOURCE,
    )
    with pytest.raises(ValueError, match="changed explicit model settings"):
        resolve_training(bad, **args)
    defaults = replace(
        resolver,
        resolve=lambda m: replace(
            metadata(m), implementation="linear", precision={"compute": "float32"}
        ),
    )
    pinned = defaults.pin(Model.from_pretrained("fixture://classifier"))
    assert pinned.implementation == "linear" and pinned.precision == {
        "compute": "float32"
    }
    with pytest.raises(ValueError, match="changed explicit model settings"):
        defaults.pin(
            Model.from_pretrained(
                "fixture://classifier", precision={"compute": "float64"}
            )
        )


@dataclass(frozen=True)
class ConfiguredClassifier:
    num_classes: int = 3
    dtype: str = "float32"
    frozen: bool = False


def model_settings(request):
    if request.lora is not None or request.implementation not in (None, "linear"):
        raise ValueError("unsupported model options")
    if request.precision.keys() - {"compute"} or request.trainable not in (
        None,
        (),
        ("head",),
    ):
        raise ValueError("unsupported precision/trainable selection")
    return {
        **request.config,
        "dtype": request.precision.get("compute", "float32"),
        "frozen": request.trainable == (),
    }


def configured_classifier(config):
    import torch

    return torch.nn.Linear(
        4, config.num_classes, dtype=getattr(torch, config.dtype)
    ).requires_grad_(not config.frozen)


def test_installed_structural_settings_reach_the_ordinary_factory_and_are_stable():
    import torch

    m, _, p = definitions()
    definition = replace(
        m.definition,
        config=ConfiguredClassifier,
        configure=model_settings,
        create=configured_classifier,
    )
    registry = installed(model=definition)
    request = Model(
        definition.id,
        implementation="linear",
        precision={"compute": "float64"},
        trainable=[],
    )
    resolved = registry.resolve_model(definition.id, settings=request)
    actual = resolved.create()
    assert actual.weight.dtype == torch.float64 and not actual.weight.requires_grad
    assert (
        registry.resolve_model(definition.id, settings=resolved.settings).settings
        == resolved.settings
    )
    for unsupported in (
        replace(request, lora=LoRA(2)),
        replace(request, precision={"unknown": "float64"}),
    ):
        with pytest.raises(ValueError, match="unsupported"):
            registry.resolve_model(definition.id, settings=unsupported)
    with pytest.raises(ValueError, match="no settings hook"):
        installed().resolve_model(m.definition.id, settings=request)
    unstable = replace(
        definition,
        configure=lambda r: {
            **r.config,
            "num_classes": int(r.config.get("num_classes", 3)) + 1,
        },
    )
    with pytest.raises(ValueError, match="not stable"):
        installed(model=unstable).resolve_model(
            unstable.id, settings=Model(unstable.id)
        )

    def configure_optimizer(request):
        if request.groups.keys() - {"weight_decay"} or request.precision:
            raise ValueError("unsupported optimizer settings")
        return {**request.config, "weight_decay": request.groups.get("weight_decay", 0)}

    def optimizer_factory(model, config, controls):
        return torch.optim.SGD(
            model.parameters(),
            lr=controls["lr"],
            weight_decay=config["weight_decay"],
            foreach=False,
        )

    optimizer = replace(
        p.definition, configure=configure_optimizer, create=optimizer_factory
    )
    registry = installed(model=definition, optimizer=optimizer)
    setup = resolve_training(
        registry,
        **{
            **arguments(),
            "model": request,
            "optimizer": Optimizer(
                optimizer.id, groups={"weight_decay": 0.2}, defaults={"lr": 0.03}
            ),
        },
    )
    captured_factory = setup.optimizer.create(actual, {"lr": torch.tensor(0.03)})
    assert captured_factory.param_groups[0]["weight_decay"] == 0.2
    assert setup.config.optimizer.groups == {"weight_decay": 0.2}
    assert resolve_training(registry, config=setup.config).config == setup.config
    with pytest.raises(ValueError, match="no settings hook"):
        installed().resolve_optimizer(
            p.definition.id,
            settings=Optimizer(p.definition.id, precision={"state": "float64"}),
        )


def test_setup_resolution_and_snapshot_roundtrip_work_without_frameworks():
    code = """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads'):
    sys.modules[name] = None
from engine_tests.training.test_setup_resolution import installed, arguments
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client.protocol.codec import encode, decode
registry = installed()
setup = resolve_training(registry, **arguments())
restored = resolve_training(registry, config=decode(encode(setup.config)))
assert restored.config == setup.config
"""
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_definition_supplies_generic_parallelism_and_initialization_stays_explicit():
    model, _, _ = definitions()

    def default(config, schema):
        assert config.num_classes == 3
        return Parallelism(
            ranks_per_replica=2,
            inputs=InputPlacement(
                GroupRef("cooperating", ((0, 1),)),
                fields=[InputFieldPlacement(tuple(f.path)) for f in schema.fields],
            ),
        )

    registry = installed(model=replace(model.definition, default_parallelism=default))
    args = arguments()
    args["model"] = replace(args["model"], initialize=None)
    described = resolve_training(registry, **args)
    assert described.config.model.initialize is None
    assert described.config.origins["parallelism"] == "definition"
    assert described.context.parallelism.world_size == 2
    assert described.context.parallelism.inputs.partition_count == 1
    assert (
        resolve_training(registry, config=described.config).config == described.config
    )
    explicit = resolve_training(registry, **args, parallelism=Parallelism())
    assert explicit.context.parallelism.world_size == 1
    assert explicit.config.origins["parallelism"] == "explicit"


def test_generic_trainable_selection_preserves_ties_and_checks_all_names_first():
    import torch

    from shadowspill.pytorch.trainable import select_trainable

    model = torch.nn.Module()
    model.left = torch.nn.Linear(3, 3)
    model.right = torch.nn.Linear(3, 3)
    model.right.weight = model.left.weight
    identity = id(model.left.weight)
    select_trainable(model, ["right.weight"])
    assert model.left.weight.requires_grad and model.right.weight.requires_grad
    assert not model.left.bias.requires_grad and not model.right.bias.requires_grad
    with pytest.raises(ValueError, match="matches no registered tensor"):
        select_trainable(model, ["left.bias", "typo"])
    assert model.left.weight.requires_grad and not model.left.bias.requires_grad
    select_trainable(model, [])
    assert not any(p.requires_grad for p in model.parameters())
    assert id(model.right.weight) == identity
    resolved = installed().resolve_model(
        "external/image_classifier@1",
        settings=Model("external/image_classifier@1", trainable=["head.weight"]),
    )
    created = resolved.create()
    assert created.head.weight.requires_grad and not created.head.bias.requires_grad
