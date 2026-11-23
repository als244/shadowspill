"""Discard shared values without detaching bindings or touching other state."""

from types import SimpleNamespace

import pytest

from shadowspill.execution import executor as module
from shadowspill.execution.executor import Executor


def executor(monkeypatch, *, overlap=False, fail=False):
    released, events = [], []

    def release(alias):
        released.append(alias)
        if fail:
            raise OSError("spill release failed")

    def path(prefix, gradient):
        identities = {prefix + "grad": gradient, prefix + "weight": 10}
        objects = SimpleNamespace(
            requires_storage=lambda alias: True,
            runtime_object_id=identities.__getitem__,
            release_value=release,
        )
        bindings = tuple(
            SimpleNamespace(
                alias_group_id=prefix + alias, identity=SimpleNamespace(role=role)
            )
            for alias, role in (("grad", "gradient"), ("weight", "parameter"))
        )
        return SimpleNamespace(
            bridge=SimpleNamespace(objects=objects),
            plan=SimpleNamespace(program=SimpleNamespace(bindings=bindings)),
        )

    monkeypatch.setattr(module, "program_state", lambda program, rank: program)
    instance = Executor.__new__(Executor)
    instance._closed = instance._plan_released = False
    instance._failure = instance._distributed = None
    instance._initialized = True
    instance._gradients_held = True
    instance._pending = SimpleNamespace(result=lambda: events.append("waited"))
    instance._paths = (path("owner-", 11), path("accum-", 11))
    instance._named_paths = {"different-shape": path("shared-", 10 if overlap else 11)}
    return instance, released, events


def test_discard_synchronizes_and_releases_shared_gradient_once(monkeypatch):
    instance, released, events = executor(monkeypatch)
    instance.discard_gradients()
    assert events == ["waited"] and instance._pending is None
    assert released == ["owner-grad"]
    assert instance._initialized and instance._failure is None
    assert not instance._gradients_held


def test_discard_without_an_open_accumulation_releases_nothing(monkeypatch):
    instance, released, events = executor(monkeypatch)
    instance._gradients_held = False
    instance.discard_gradients()
    assert events == ["waited"] and not released
    assert instance._failure is None


def test_discard_refuses_weight_alias_before_releasing_any_value(monkeypatch):
    instance, released, _ = executor(monkeypatch, overlap=True)
    with pytest.raises(ValueError, match="overlaps other persistent state"):
        instance.discard_gradients()
    assert not released
    with pytest.raises(RuntimeError, match="unusable"):
        instance.synchronize()


def test_failed_release_cannot_leave_reusable_partial_state(monkeypatch):
    instance, released, _ = executor(monkeypatch, fail=True)
    with pytest.raises(OSError, match="spill release failed"):
        instance.discard_gradients()
    with pytest.raises(RuntimeError, match="unusable"):
        instance.discard_gradients()
    assert released == ["owner-grad"]
