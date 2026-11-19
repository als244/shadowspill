"""Bounded observations from an isolated device probe; no controller CUDA use."""

import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from shadowspill_client._records import Record, integer, number, record, text
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, RecordCodec

from .supervision import OwnedProcess

MAX_DEVICES = 1024


@dataclass(frozen=True)
class DeviceObservation(Record):
    uuid: str
    name: str
    total_bytes: int
    free_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.uuid, str)
            or re.fullmatch(
                r"(?:GPU|MIG)-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", self.uuid
            )
            is None
        ):
            raise ValueError("device observation requires a complete GPU or MIG UUID")
        if len(text(self.name, "device name")) > 256:
            raise ValueError("device name exceeds its bound")
        integer(self.total_bytes, "total device bytes", minimum=1)
        integer(self.free_bytes, "free device bytes")
        if self.free_bytes > self.total_bytes:
            raise ValueError("device free memory exceeds total memory")


@dataclass(frozen=True)
class DeviceInventory(Record):
    driver: str
    devices: Sequence[DeviceObservation]

    def __post_init__(self) -> None:
        if len(text(self.driver, "device driver version")) > 256:
            raise ValueError("device driver version exceeds its bound")
        if (
            not isinstance(self.devices, (tuple, list))
            or len(self.devices) > MAX_DEVICES
        ):
            raise ValueError("device inventory exceeds the bounded device count")
        devices = tuple(record(DeviceObservation, d, "device") for d in self.devices)
        if len({device.uuid for device in devices}) != len(devices):
            raise ValueError("device inventory contains a duplicate UUID")
        object.__setattr__(self, "devices", devices)


CODEC = RecordCodec({"node.devices": DeviceInventory})


def probe_devices(
    *, environment: Mapping[str, str], root: Path, timeout: float = 30
) -> DeviceInventory:
    """Observe only the supplied allocation environment in an owned probe child.

    CUDA resolves visibility and ordering; NVML indexes are never substituted.
    Results are snapshots, not scheduler assignments or device reservations.
    The trusted node daemon supplies its own environment, never request-supplied code.
    """
    if number(timeout, "device probe timeout", minimum=0) == 0:
        raise ValueError("device probe timeout must be positive")
    with TemporaryDirectory(prefix="devices-", dir=root) as temporary:
        directory = Path(temporary) / "process"
        child = OwnedProcess(
            [
                sys.executable,
                "-I",
                "-m",
                "shadowspill.orchestration.launch.device_probe",
            ],
            environment=environment,
            directory=directory,
            cwd=Path(temporary),
            log_limit_bytes=MAX_METADATA_BYTES,
        )
        try:
            status = child.wait(timeout)
        finally:
            child.close(grace_seconds=0)
        if status:
            with (directory / "stderr.log").open("rb") as stream:
                detail = stream.read(4096).decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"device discovery failed with status {status}: {detail}"
            )
        if child.logs[0].truncated:
            raise ValueError(
                "device inventory output was truncated or could not be read"
            )
        with (directory / "stdout.log").open("rb") as stream:
            data = stream.read(MAX_METADATA_BYTES + 1)
        value = CODEC.decode(data)
        assert isinstance(value, DeviceInventory)
        return value
