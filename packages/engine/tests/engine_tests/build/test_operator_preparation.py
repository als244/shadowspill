"""Portable binary ownership and refusal without executing or importing a compiler."""

import copy
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import replace
from hashlib import sha256
from threading import Event, Thread

import pytest
import torch

from shadowspill import OperatorPreparation, Program
from shadowspill.artifacts._values import plain
from shadowspill.ir import DeviceSpec
from shadowspill.operator_preparation.lifecycle import (
    BuildPreparations,
    LoadedPreparations,
)

_events = []
_captured = ContextVar("test_prepared_capture", default=None)
_bound = ContextVar("test_prepared_binary", default=None)
_binary = b"provider-specific executable bytes"
_checksum = sha256(_binary).hexdigest()


def factory(*, phase, device, options):
    _events.append((phase, str(device), options))

    class Builder:
        @contextmanager
        def capture(self):
            value = {}
            token = _captured.set(value)
            try:
                yield value
            finally:
                _captured.reset(token)

        def export(self, value):
            assert value == {"selected": True}
            return {"binary": _checksum}, {_checksum: _binary}

        def close(self):
            _events.append("close-build")

    class Loader:
        def bind(self, manifest, read_blob):
            data = read_blob(manifest["binary"])
            if data != _binary:
                raise ValueError("corrupt test binary")
            _events.append("bind")

            class Binding:
                @contextmanager
                def activate(self):
                    token = _bound.set(data)
                    try:
                        yield
                    finally:
                        _bound.reset(token)

            return Binding()

        def close(self):
            _events.append("close-load")

    return Builder() if phase == "build" else Loader()


@pytest.fixture(autouse=True)
def clear_events():
    _events.clear()


def prepared_program():
    description = OperatorPreparation(__name__ + ":factory", {"variant": "a"})
    destination = torch.tensor(0)
    with closing(BuildPreparations((description,), torch.device("cpu"))) as build:
        for key in ("first", "continuing"):
            with build.capture(key):
                _captured.get()["selected"] = True
                destination.add_(1)
        program = Program(
            devices=(DeviceSpec("cuda_0"),),
            alias_groups=(),
            objects=(),
            tasks=(),
            task_definitions=(),
            executables={"first": b"first", "continuing": b"continuing"},
            bindings={"operator_preparation": build.metadata()},
            operator_binaries=build.blobs,
        )
    assert destination.item() == 2  # Capture never invokes a mutating task again.
    return program


def test_archive_deduplicates_binary_and_loads_only_requested_tasks(tmp_path):
    program = prepared_program()
    program.save(tmp_path / "program.ssp")
    restored = Program.load(tmp_path / "program.ssp")
    assert restored.digest == program.digest
    assert len(restored.operator_binaries) == 1
    assert len(restored._archive_payloads()) == 3
    with closing(LoadedPreparations(torch.device("cpu"))) as loaded:
        loaded.add(restored, ("first",))
        loaded.add(restored, ("first",))
        # Different Programs share the provider owner and existing task binding.
        loaded.add(replace(restored, name="second program"), ("continuing",))
        assert _events.count("bind") == 2
        assert sum(isinstance(row, tuple) and row[0] == "load" for row in _events) == 1
        assert loaded.wrap(restored, "first", lambda: _bound.get())() == _binary
        assert _bound.get() is None
    assert _events.count("close-load") == 1


def test_without_preparation_archive_shape_and_function_are_unchanged():
    program = Program(
        devices=(DeviceSpec("cuda_0"),),
        alias_groups=(),
        objects=(),
        tasks=(),
        task_definitions=(),
    )
    assert "operator_binaries" not in program.to_dict()
    restored = Program.from_dict(program.to_dict())
    assert restored.digest == program.digest

    def function():
        return None

    with closing(LoadedPreparations(torch.device("cpu"))) as loaded:
        loaded.add(program, ())
        assert loaded.wrap(program, "ignored", function) is function
    assert not _events


@pytest.mark.parametrize(
    "damage", ["source", "missing-task", "corrupt-binary", "missing-binary"]
)
def test_incompatible_preparation_refuses_before_task_invocation(damage):
    program = prepared_program()
    metadata = plain(program.bindings["operator_preparation"])
    if damage == "source":
        metadata["providers"][0]["source"] = "changed"
    elif damage == "missing-task":
        metadata["tasks"].pop("first")
    elif damage == "corrupt-binary":
        with pytest.raises(ValueError, match="binary digest"):
            replace(program, operator_binaries={_checksum: b"changed"})
        return
    else:
        program = replace(program, operator_binaries={})
    program = replace(program, bindings={"operator_preparation": metadata})
    with closing(LoadedPreparations(torch.device("cpu"))) as loaded:
        with pytest.raises((ValueError, KeyError)):
            loaded.add(program, ("first",))
        with pytest.raises(ValueError, match="not loaded"):
            loaded.wrap(program, "first", lambda: pytest.fail("executed"))


def test_active_task_prevents_close_and_exception_restores_scope():
    program = prepared_program()
    loaded = LoadedPreparations(torch.device("cpu"))
    loaded.add(program, ("first",))
    entered, release = Event(), Event()
    errors = []

    def task():
        assert _bound.get() == _binary
        entered.set()
        assert release.wait(10)
        raise ValueError("task error")

    wrapped = loaded.wrap(program, "first", task)

    def run():
        try:
            wrapped()
        except ValueError as error:
            errors.append(str(error))
        assert _bound.get() is None

    thread = Thread(target=run)
    thread.start()
    try:
        assert entered.wait(10)
        with pytest.raises(RuntimeError, match="active"):
            loaded.close()
        with pytest.raises(RuntimeError, match="during execution"):
            loaded.add(program, ("continuing",))
        assert "close-load" not in _events
    finally:
        release.set()
        thread.join(10)
        loaded.close()
    assert errors == ["task error"] and not thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        wrapped()


def test_missing_or_corrupt_binary_is_rejected_by_archive_restore():
    program = prepared_program()
    data = program.to_dict()
    payloads = dict(program._archive_payloads())
    payloads.pop(_checksum)
    with pytest.raises(ValueError, match="binary payload is missing"):
        Program.from_dict(data, payloads=payloads)
    payloads[_checksum] = b"corrupt"
    with pytest.raises(ValueError, match="binary digest"):
        Program.from_dict(data, payloads=payloads)


def test_policy_options_are_immutable_json_and_duplicate_builders_refuse():
    options = {"nested": [1, {"name": "a"}]}
    description = OperatorPreparation(__name__ + ":factory", options)
    original = copy.deepcopy(options)
    options["nested"].append(2)
    assert description.to_dict()["options"] == original
    with pytest.raises(ValueError, match="duplicate"):
        BuildPreparations((description, description), torch.device("cpu"))
    assert _events.count("close-build") == 1
    with pytest.raises(TypeError, match="mapping"):
        OperatorPreparation(__name__ + ":factory", [])
    with pytest.raises(ValueError, match="importable"):
        OperatorPreparation("__main__:factory")
