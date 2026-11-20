"""Temporary model state has its own bounded, source-free initialization hook."""

import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.offline.test_shared_plans import registered
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.training.components import InitializationRuntime, ModelInitialization
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.worker.initialization import (
    open_initialization,
    prepare_temporary_initialization,
    restore_staging,
)
from shadowspill_client import Weights


def prepared(*, hook=True, persistent_alias=False):
    registry = installed()
    component = registry.resolve_model(arguments()["model"].name).component

    def describe(config, context):
        assert context.weights.kind == "artifact"
        assert context.temporary == ("linear.weight",)

        @contextmanager
        def opened(runtime):
            yield lambda request, destination: destination.fill_(3)

        return ModelInitialization(peak_bytes=64, chunk_bytes=16, open=opened)

    component = replace(component, temporary_initialization=describe if hook else None)
    settings = arguments()
    settings["model"] = replace(
        settings["model"], initialize=Weights.artifact("weights")
    )
    setup = resolve_training(installed(model=component), **settings)

    def program(prefix):
        value = registered(prefix=prefix)
        return replace(
            value,
            bindings={
                **value.bindings,
                "registrations": [
                    {
                        **row,
                        "parameter": False,
                        "persistent": persistent_alias and index == 1,
                    }
                    for index, row in enumerate(value.bindings["registrations"])
                ],
            },
        )

    return SimpleNamespace(
        inventory=SimpleNamespace(layouts=SimpleNamespace(setup=setup)),
        plans={
            0: {
                name: SimpleNamespace(program=program(prefix))
                for name, prefix in (("a", ""), ("b", "evaluation-"))
            }
        },
    )


def test_temporary_description_resolves_one_shared_tied_buffer_and_counts_staging(
    tmp_path,
):
    initial = prepare_temporary_initialization(prepared(), 0)
    assert initial.context.temporary == ("linear.weight",)
    assert initial.signature()["temporary"] == ("linear.weight",)
    assert restore_staging(initial, 100) == (16, 96)
    with pytest.raises(ValueError, match="exceed admitted staging"):
        restore_staging(initial, 65)
    assert prepare_temporary_initialization(prepared(persistent_alias=True), 0) is None
    with pytest.raises(ValueError, match="temporary initialization hook"):
        prepare_temporary_initialization(prepared(hook=False), 0)

    import torch

    from shadowspill.artifacts.tensors import TensorChunk

    runtime = InitializationRuntime(
        tmp_path, lambda _: pytest.fail("weight source opened")
    )
    with open_initialization(initial, runtime) as fill:
        destination = torch.zeros(4)
        chunk = TensorChunk("linear.weight", (4, 4), torch.float32, 0, 16)
        fill(chunk, destination)
        assert destination.tolist() == [3] * 4
        with pytest.raises(ValueError, match="undeclared model state"):
            fill(replace(chunk, name="model.parameter"), destination)


def test_temporary_preflight_needs_no_framework_or_catalog_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torch', 'triton', 'shadowspill_workloads'}:
            raise AssertionError('unexpected framework import: ' + name)
sys.meta_path.insert(0, Block())
from engine_tests.training.test_worker_temporary import prepared
from shadowspill.training.worker.initialization import prepare_temporary_initialization
initial = prepare_temporary_initialization(prepared(), 0)
assert initial.context.temporary == ('linear.weight',)
assert initial.description.disk_bytes == 0
""",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
