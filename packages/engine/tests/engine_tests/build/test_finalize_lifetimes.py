"""Kernel finalization retains only tensors required by its current task."""

import weakref
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.program import TaskDefinition
from shadowspill.build import finalize
from shadowspill.profiling import session


@pytest.mark.parametrize("phase", ["finalize", "profile"])
def test_forward_only_roots_are_dead_when_backward_preparation_starts(
    monkeypatch, phase
):
    @dataclass(frozen=True)
    class Program:
        executables: dict = field(
            default_factory=lambda: {"forward": b"f", "backward": b"b"}
        )
        bindings: dict = field(default_factory=lambda: {"operator_dependencies": {}})
        task_definitions: tuple = (
            TaskDefinition("forward", "f" * 64),
            TaskDefinition("backward", "b" * 64),
        )
        operator_binaries: dict = field(default_factory=dict)

    references = {}
    cache_reclaims = []

    class Forward:
        def __call__(self, unused, needed):
            result, saved = unused.sin(), needed.cos()
            references.update(
                unused_input=weakref.ref(unused),
                needed_input=weakref.ref(needed),
                unused_output=weakref.ref(result),
                needed_output=weakref.ref(saved),
            )
            return result, saved

        def serialize(self):
            return b"f"

    class Backward:
        def __call__(self, needed, saved):
            assert references["unused_input"]() is None
            assert references["unused_output"]() is None
            assert references["needed_input"]() is needed
            assert references["needed_output"]() is saved
            return (needed + saved,)

        def serialize(self):
            return b"b"

    class Controls:
        def __init__(self, *args):
            pass

        def prepare(self, *args, **kwargs):
            return torch.randn(7), torch.randn(5)

    def slot(name, index):
        return {"object_id": name, "leaf_index": index}

    forward = {
        "profile_key": "forward",
        "inputs": (slot("unused", 0), slot("needed", 1)),
        "outputs": (slot("result", 0), slot("saved", 1)),
    }
    backward = {
        "profile_key": "backward",
        "inputs": (slot("needed", 0), slot("saved", 1)),
    }
    monkeypatch.setattr(
        finalize,
        "stage_jobs",
        lambda *args: ({"forward": forward, "backward": backward},),
    )
    monkeypatch.setattr(finalize, "ControlInputs", Controls)

    def prepare(*args, slots, values, **kwargs):
        return tuple(values[s["object_id"]] for s in slots)

    monkeypatch.setattr(finalize, "prepare_inputs", prepare)
    monkeypatch.setattr(
        finalize,
        "restore_executable",
        lambda payload, **kwargs: Forward() if payload == b"f" else Backward(),
    )
    monkeypatch.setattr(finalize, "capture_choices", lambda: nullcontext({}))
    monkeypatch.setattr(finalize, "capture_kernels", lambda: nullcontext({}))
    monkeypatch.setattr(finalize, "freeze_kernels", lambda *args: {})
    monkeypatch.setattr(
        finalize, "resolve_output_devices", lambda program, *args: program
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: cache_reclaims.append(bool(references))
    )
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kwargs: nullcontext())
    if phase == "finalize":
        built = finalize._finalize_tasks(Program(), torch.device("cpu"))
        for definition in built.task_definitions:
            identity = built.bindings["finalized_tasks"][definition.profile_key]
            assert digest(identity) == definition.compatibility_digest
            assert identity["task"] == definition.profile_key[0] * 64
            assert identity["inductor_kernels"] == {}
            assert identity["output_devices"] == (
                ("cpu", "cpu") if definition.profile_key == "forward" else ("cpu",)
            )
    else:
        monkeypatch.setattr(session, "prepare_inputs", prepare)
        monkeypatch.setattr(torch.cuda, "stream", lambda *args: nullcontext())
        functions = {"forward": Forward(), "backward": Backward()}
        instance = SimpleNamespace(
            program=Program(),
            device=torch.device("cpu"),
            profiles=SimpleNamespace(tasks={}),
            controls=Controls(),
            inputs={},
            streams={"forward": SimpleNamespace(synchronize=lambda: None)},
            function=lambda key: functions[key],
            measure=lambda key, arguments: functions[key](*arguments),
        )
        session.MeasurementSession.pair(
            instance, {"forward": forward, "backward": backward}
        )
    assert all(reference() is None for reference in references.values())
    assert cache_reclaims == [False]
