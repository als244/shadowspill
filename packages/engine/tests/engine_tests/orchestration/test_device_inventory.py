"""Device discovery preserves CUDA visibility and keeps the controller CPU-only."""

import ctypes
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shadowspill.orchestration.launch import device_inventory, device_probe
from shadowspill.orchestration.launch.device_inventory import (
    CODEC,
    DeviceInventory,
    DeviceObservation,
    probe_devices,
)
from shadowspill.profiling import pool_identity

GPU_A = "GPU-12345678-1111-2222-3333-444444444444"
GPU_B = "GPU-87654321-1111-2222-3333-444444444444"


def nvml(monkeypatch, visible=(GPU_B, GPU_A)):
    events = []

    class NotFound(Exception):
        pass

    def handle(identifier):
        events.append(identifier)
        return identifier

    api = SimpleNamespace(
        NVMLError_NotFound=NotFound,
        nvmlInit=lambda: events.append("init"),
        nvmlShutdown=lambda: events.append("shutdown"),
        nvmlDeviceGetHandleByUUID=handle,
        nvmlDeviceGetUUID=lambda handle: handle,
        nvmlDeviceGetName=lambda handle: "device " + handle,
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(total=8000, free=5000),
        nvmlSystemGetDriverVersion=lambda: "123.4",
    )
    monkeypatch.setitem(sys.modules, "pynvml", api)
    monkeypatch.setattr(device_probe, "visible_device_count", lambda: len(visible))
    monkeypatch.setattr(device_probe, "device_uuid", visible.__getitem__)
    return api, events


def test_cuda_ordinals_map_to_uuids_not_nvml_indexes(monkeypatch):
    _, events = nvml(monkeypatch)
    inventory = device_probe.observe()
    assert [d.uuid for d in inventory.devices] == [GPU_B, GPU_A]
    assert events == ["init", GPU_B, GPU_A, "shutdown"]
    assert [(d.total_bytes, d.free_bytes) for d in inventory.devices] == [
        (8000, 5000),
        (8000, 5000),
    ]
    assert CODEC.decode(CODEC.encode(inventory)) == inventory


def test_mig_instance_uses_its_own_uuid_not_the_parent(monkeypatch):
    api, events = nvml(monkeypatch, (GPU_A,))
    mig = "MIG-" + GPU_A.removeprefix("GPU-")

    def handle(identifier):
        events.append(identifier)
        if identifier == GPU_A:
            raise api.NVMLError_NotFound
        assert identifier == mig
        return identifier

    api.nvmlDeviceGetHandleByUUID = handle
    assert device_probe.observe().devices[0].uuid == mig
    assert events == ["init", GPU_A, mig, "shutdown"]


@pytest.mark.parametrize("failure", ["permission", "identity", "memory"])
def test_driver_failures_are_not_an_empty_or_different_inventory(monkeypatch, failure):
    api, events = nvml(monkeypatch, (GPU_A,))
    if failure == "permission":

        def denied(identifier):
            raise PermissionError("device not assigned")

        api.nvmlDeviceGetHandleByUUID = denied
    elif failure == "identity":
        api.nvmlDeviceGetUUID = lambda handle: GPU_B
    else:
        api.nvmlDeviceGetMemoryInfo = lambda handle: SimpleNamespace(total=10, free=11)
    with pytest.raises((ValueError, PermissionError)):
        device_probe.observe()
    assert events[-1] == "shutdown"


def test_empty_visibility_and_bounded_device_count(monkeypatch):
    _, events = nvml(monkeypatch, ())
    assert not device_probe.observe().devices
    assert events == ["init", "shutdown"]
    monkeypatch.setattr(device_probe, "visible_device_count", lambda: 1025)
    with pytest.raises(ValueError, match="count"):
        device_probe.observe()
    assert events == ["init", "shutdown"]


@pytest.mark.parametrize(
    "value", [GPU_A.replace("GPU-", ""), "0", GPU_A + "/1/2", GPU_A.replace("1", "A")]
)
def test_inventory_refuses_ambiguous_identity(value):
    with pytest.raises(ValueError, match="UUID"):
        DeviceObservation(value, "test", 10, 5)


def test_inventory_refuses_duplicates_and_invalid_memory():
    device = DeviceObservation(GPU_A, "test", 10, 5)
    with pytest.raises(ValueError, match="duplicate"):
        DeviceInventory("1", [device, device])
    for total, free in ((0, 0), (10, -1), (10, 11), (True, 1)):
        with pytest.raises(ValueError):
            DeviceObservation(GPU_A, "test", total, free)


@pytest.mark.parametrize("failure", [None, "status", "truncated", "timeout"])
def test_probe_owns_cleanup_and_bounded_output_before_return(
    tmp_path, monkeypatch, failure
):
    inventory = DeviceInventory("1", [DeviceObservation(GPU_A, "test", 10, 5)])
    children = []

    class Child:
        def __init__(self, command, **kwargs):
            children.append(self)
            assert command[:3] == [sys.executable, "-I", "-m"]
            assert command[3] == "shadowspill.orchestration.launch.device_probe"
            assert kwargs["environment"] == {"CUDA_VISIBLE_DEVICES": "2,0"}
            assert kwargs["log_limit_bytes"] == 1 << 20
            self.directory = kwargs["directory"]
            self.directory.mkdir()
            self.logs = [SimpleNamespace(truncated=failure == "truncated")]
            self.closed = False

        def wait(self, timeout):
            assert timeout == 2
            if failure == "timeout":
                raise subprocess.TimeoutExpired("probe", timeout)
            return 1 if failure == "status" else 0

        def close(self, *, grace_seconds):
            assert grace_seconds == 0
            self.closed = True
            # Logs are finalized before they may be decoded.
            (self.directory / "stdout.log").write_bytes(CODEC.encode(inventory))
            (self.directory / "stderr.log").write_text("discovery failed")

    monkeypatch.setattr(device_inventory, "OwnedProcess", Child)
    if failure:
        expected = {
            "status": RuntimeError,
            "truncated": ValueError,
            "timeout": subprocess.TimeoutExpired,
        }[failure]
        with pytest.raises(expected):
            probe_devices(
                environment={"CUDA_VISIBLE_DEVICES": "2,0"}, root=tmp_path, timeout=2
            )
    else:
        assert (
            probe_devices(
                environment={"CUDA_VISIBLE_DEVICES": "2,0"}, root=tmp_path, timeout=2
            )
            == inventory
        )
    assert children[0].closed and not list(tmp_path.iterdir())


@pytest.mark.parametrize("status,count", [(0, 2), (100, 0), (999, 0), (0, -1)])
def test_count_uses_driver_visibility_and_propagates_errors(monkeypatch, status, count):
    def init(flags):
        assert flags == 0
        return status

    def get_count(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = count
        return 0

    driver = SimpleNamespace(cuInit=init, cuDeviceGetCount=get_count)
    monkeypatch.setattr(pool_identity.ct, "CDLL", lambda name: driver)
    if status == 999 or count < 0:
        with pytest.raises(RuntimeError):
            pool_identity.visible_device_count()
    else:
        assert pool_identity.visible_device_count() == count
