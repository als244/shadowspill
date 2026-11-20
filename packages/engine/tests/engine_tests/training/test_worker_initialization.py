"""Installed loaders declare their bound on CPU before filling owned chunks."""

import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.parallelism import InputPlacement, Parallelism
from shadowspill.training.components import (
    InitializationContext,
    InitializationRuntime,
    ModelInitialization,
)
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.worker.initialization import (
    initialize,
    open_initialization,
    prepare_initialization,
)
from shadowspill_client import Execution, Weights
from shadowspill_client.buffers import BufferBudget


def preparation(hook=None, **options):
    registry = installed()
    if hook is not None:
        model = registry.resolve_model(arguments()["model"].name)
        registry = installed(model=replace(model.component, initialization=hook))
    return resolve_training(registry, **{**arguments(), **options})


def runtime(tmp_path):
    return InitializationRuntime(
        tmp_path,
        lambda _: pytest.fail("random initialization opened an artifact"),
    )


def test_initialization_disk_bounds_are_validated_and_enter_startup_identity():
    setup = preparation()
    initial = prepare_initialization(setup)[0]
    assert initial.description.disk_bytes == 0
    changed = replace(
        initial, description=replace(initial.description, disk_bytes=1 << 20)
    )
    assert changed.signature() != initial.signature()
    assert changed.signature()["disk_bytes"] == 1 << 20
    for invalid in (-1, True, 0.5):
        with pytest.raises((ValueError, TypeError), match="initialization disk bytes"):
            replace(initial.description, disk_bytes=invalid)


def test_a_larger_loader_cannot_open_beyond_the_live_disk_allowance(tmp_path):
    initial = prepare_initialization(preparation())[0]
    initial = replace(
        initial,
        description=replace(
            initial.description,
            disk_bytes=1024,
            open=lambda _: pytest.fail("opened before checking disk allowance"),
        ),
    )
    with (
        pytest.raises(ValueError, match="admitted worker disk allowance"),
        open_initialization(initial, runtime(tmp_path)),
    ):
        pytest.fail("opened unadmitted loader")


def test_cpu_preflight_never_imports_framework_catalog_or_opens_loader():
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'triton', 'shadowspill_workloads'}:
            raise AssertionError('forbidden description import: ' + fullname)
sys.meta_path.insert(0, Block())
from engine_tests.fixtures.installed_component import install
from shadowspill.training.components import load_registry
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.worker.initialization import prepare_initialization
from shadowspill_client import Model, Weights, Objective, Optimizer, BatchConfig, Memory
from pathlib import Path
from tempfile import TemporaryDirectory
with TemporaryDirectory() as path:
    install(Path(path))
    sys.path.insert(0, path)
    setup = resolve_training(load_registry(['external-classifier-fixture']),
        model=Model('external/image_classifier@1', initialize=Weights.random(seed=17)),
        objective=Objective('external/classification_outputs@1'),
        optimizer=Optimizer('external/operation_sgd@1'),
        batch=BatchConfig(limits={'max_examples': 4}),
        memory=Memory(device_gib=2, spill_gib=1))
    prepared = prepare_initialization(setup)[0]
    assert prepared.signature()['weights'] == setup.config.model.initialize.to_dict()
    assert prepared.source == setup.model.source
    assert prepared.description.peak_bytes == 1 << 20
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_creation_requires_explicit_weights_and_a_bounded_installed_hook():
    registry = installed()
    model = registry.resolve_model(arguments()["model"].name)
    setup = resolve_training(
        registry,
        **{**arguments(), "model": replace(arguments()["model"], initialize=None)},
    )
    with pytest.raises(ValueError, match="explicit weights or a checkpoint"):
        prepare_initialization(setup)
    setup = resolve_training(
        installed(model=replace(model.component, initialization=None)), **arguments()
    )
    with pytest.raises(ValueError, match="no bounded initialization hook"):
        prepare_initialization(setup)
    with pytest.raises(TypeError, match="ModelInitialization"):
        prepare_initialization(preparation(lambda *_: object()))
    with pytest.raises(ValueError, match="host staging limit"):
        prepare_initialization(preparation(execution=Execution(max_staging_bytes=128)))
    with pytest.raises(TypeError, match="CPU description function"):
        replace(model.component, initialization=32)


def test_contexts_pin_rank_weights_revision_and_source():
    setup = preparation()
    plan = prepare_initialization(setup)[0]
    assert plan.context.model == setup.model.settings
    assert plan.signature()["model_source"] == setup.model.source.to_dict()
    with pytest.raises(ValueError, match="differ from the resolved"):
        replace(plan.context, weights=Weights.random(seed=18))
    with pytest.raises(ValueError, match="outside the prepared placement"):
        replace(plan.context, logical_rank=1)
    with pytest.raises((TypeError, ValueError), match="rank"):
        replace(plan.context, logical_rank=True)
    for weights, message in (
        (Weights.pretrained("fixture://classifier"), "pinned revision"),
        (Weights.artifact("/etc/weights"), "artifact ID"),
    ):
        with pytest.raises(ValueError, match=message):
            InitializationContext(
                replace(plan.context.model, initialize=weights),
                weights,
                plan.context.parallelism,
                0,
            )
    with pytest.raises(ValueError, match="differs from the resolved model"):
        setup.model.describe_initialization(
            replace(plan.context, model=replace(plan.context.model, config={}))
        )
    placement = Parallelism(ranks_per_replica=2, inputs=InputPlacement.independent(2))
    descriptions = prepare_initialization(preparation(parallelism=placement))
    assert list(descriptions) == [0, 1]
    assert descriptions[1].context.logical_rank == 1
    with pytest.raises(TypeError):
        descriptions[2] = descriptions[0]


@pytest.mark.parametrize("peak,chunk", [(0, 1), (8, 0), (8, 9), (True, 1)])
def test_invalid_bounds_are_rejected(peak, chunk):
    with pytest.raises((ValueError, TypeError)):
        ModelInitialization(peak, chunk, lambda _: None)


class OwnedChunks:
    logical_rank = 0

    def __init__(self, *, transform=lambda value: value):
        self.values = torch.empty(12)
        self.transform = transform

    def initialize(self, fill, *, chunk_bytes):
        self.chunk_bytes = chunk_bytes
        for offset, end in ((4, 8), (8, 12), (0, 4)):
            fill(
                TensorChunk("head.weight", (3, 4), torch.float32, offset, 12),
                self.transform(self.values[offset:end]),
            )


def test_seeded_initialization_is_independent_of_chunk_and_ownership(tmp_path):
    prepared = prepare_initialization(preparation())[0]
    arena = BufferBudget(prepared.description.peak_bytes)
    worker = OwnedChunks()
    initialize(worker, prepared, runtime(tmp_path), arena)
    with prepared.description.open(runtime(tmp_path)) as fill:
        expected = torch.empty(12)
        fill(TensorChunk("head.weight", (3, 4), torch.float32, 0, 12), expected)
    torch.testing.assert_close(worker.values, expected, rtol=0, atol=0)
    assert worker.chunk_bytes == 32 and arena.used == (0, 0)


@pytest.mark.parametrize("failure", ["none", "open", "fill", "wrong_type"])
def test_reservation_precedes_loader_and_cleanup_precedes_release(tmp_path, failure):
    arena = BufferBudget(1024)
    events = []

    @contextmanager
    def open_loader(runtime):
        assert arena.used == (1024, 1)
        events.append("open")
        try:
            if failure == "open":
                raise RuntimeError("open failed")

            def fill(request, destination):
                assert arena.used == (1024, 1)
                if failure == "fill":
                    raise RuntimeError("fill failed")
                destination.fill_(2)

            yield object() if failure == "wrong_type" else fill
        finally:
            assert arena.used == (1024, 1)
            events.append("close")

    prepared = prepare_initialization(
        preparation(lambda *_: ModelInitialization(1024, 16, open_loader))
    )[0]
    with pytest.raises(ValueError, match="staging bytes"):
        initialize(OwnedChunks(), prepared, runtime(tmp_path), BufferBudget(1000))
    assert events == []
    worker = OwnedChunks()
    worker.logical_rank = 1
    with pytest.raises(ValueError, match="another logical worker"):
        initialize(worker, prepared, runtime(tmp_path), arena)
    assert events == []
    if failure == "none":
        initialize(OwnedChunks(), prepared, runtime(tmp_path), arena)
    else:
        with pytest.raises((RuntimeError, TypeError)):
            initialize(OwnedChunks(), prepared, runtime(tmp_path), arena)
    assert events == ["open", "close"] and arena.used == (0, 0)


@pytest.mark.parametrize(
    "transform",
    [
        lambda value: torch.empty(9),  # Beyond the admitted destination chunk.
        lambda value: value.reshape(2, 2),
        lambda value: value.to(torch.float64),
        lambda value: value.as_strided((4,), (0,)),
    ],
)
def test_incompatible_destination_never_reaches_installed_loader(tmp_path, transform):
    prepared = prepare_initialization(preparation())[0]
    arena = BufferBudget(prepared.description.peak_bytes)
    with pytest.raises(ValueError, match="incompatible owned CPU chunk"):
        initialize(OwnedChunks(transform=transform), prepared, runtime(tmp_path), arena)
    assert arena.used == (0, 0)
