"""Calibration takes a runtime's host placement and never fails for want of it."""

import ctypes as ct
import os
from pathlib import Path

import pytest

from shadowspill.communication import placement
from shadowspill.communication.measure import measure_communication


class Function:
    def __init__(self, name, status, calls):
        self.name, self.status, self.calls = name, status, calls

    def __call__(self, *arguments):
        self.calls.append((self.name, arguments))
        return self.status


def library(monkeypatch, *, opened=0, acquired=0):
    calls = []

    class Library:
        shadowspill_pytorch_spill_probe_open = Function("open", opened, calls)
        shadowspill_pytorch_spill_probe_acquire = Function("acquire", acquired, calls)
        shadowspill_pytorch_spill_probe_destroy = Function("destroy", None, calls)

    monkeypatch.setattr(placement, "resolve_library", lambda name: Path("/lib") / name)
    monkeypatch.setattr(ct, "CDLL", lambda path: Library)
    return calls


def test_binding_creates_a_bound_probe_runtime_and_destroys_it(monkeypatch):
    calls = library(monkeypatch)
    assert placement.bind_to_device(3) is None
    assert [name for name, _ in calls] == ["open", "acquire", "destroy"]
    backend, device, extension, _ = calls[0][1]
    assert (backend, device, extension) == (
        b"/lib/libshadowspill_backend_cuda.so",
        3,
        None,
    )
    # Pinned-host kind, no configuration, minimal pools, and binding enabled.
    assert calls[1][1][1:] == (1, None, 4 << 20, 2 << 20, 0)


def test_an_unopened_probe_reports_its_status_and_owns_nothing(monkeypatch):
    calls = library(monkeypatch, opened=7)
    assert "status=7" in placement.bind_to_device(0)
    assert [name for name, _ in calls] == ["open"]


def test_a_failed_runtime_reports_its_status_and_is_destroyed(monkeypatch):
    calls = library(monkeypatch, acquired=5)
    assert "status=5" in placement.bind_to_device(0)
    assert [name for name, _ in calls] == ["open", "acquire", "destroy"]


def test_missing_libraries_and_symbols_are_reasons_not_errors(monkeypatch):
    monkeypatch.setattr(placement, "resolve_library", lambda name: None)
    assert "unavailable" in placement.bind_to_device(0)
    monkeypatch.setattr(placement, "resolve_library", lambda name: Path("/lib") / name)
    monkeypatch.setattr(ct, "CDLL", lambda path: object())
    assert "shadowspill_pytorch_spill_probe_open" in placement.bind_to_device(0)

    def unloadable(path):
        raise OSError("cannot open shared object file")

    monkeypatch.setattr(ct, "CDLL", unloadable)
    assert "cannot open" in placement.bind_to_device(0)


def test_placement_describes_this_threads_cpus_and_their_nodes():
    described = placement.host_placement()
    assert described["cpus"] == len(os.sched_getaffinity(0))
    assert described["numa_nodes"] == sorted(set(described["numa_nodes"]))
    assert described["numa_nodes"] or not placement.NODES.is_dir()


def test_a_narrowed_affinity_names_only_the_nodes_it_reaches(tmp_path, monkeypatch):
    for node, cpus in ((0, "0-3"), (1, "4-7,12"), (10, "16-19")):
        (tmp_path / f"node{node}").mkdir()
        (tmp_path / f"node{node}/cpulist").write_text(cpus + "\n")
    (tmp_path / "node2").mkdir()  # a node whose CPU list cannot be read
    monkeypatch.setattr(placement, "NODES", tmp_path)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {5, 12, 17})
    assert placement.host_placement() == {"cpus": 3, "numa_nodes": [1, 10]}


def test_the_binding_choice_is_a_boolean_checked_before_any_device_work():
    with pytest.raises(TypeError, match="numa_binding must be a bool"):
        measure_communication(numa_binding="auto")
