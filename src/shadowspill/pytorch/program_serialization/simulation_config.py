"""Serialization for simulator configuration."""

from __future__ import annotations

from dataclasses import asdict

from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.model import DeviceSimulationConfig

from .common import _integer, _list, _mapping, _string


def _simulation_config_to_dict(config: SimulationConfig) -> dict[str, object]:
    return {
        "devices": [asdict(item) for item in config.devices],
        "host_capacity_bytes": config.host_capacity_bytes,
    }


def _simulation_config_from_value(
    value: object,
    path: str,
) -> SimulationConfig:
    data = _mapping(value, path)
    devices = _list(data.get("devices"), f"{path}.devices")
    return SimulationConfig(
        tuple(
            DeviceSimulationConfig(
                device_id=_string(
                    item.get("device_id"), f"{path}.devices[{index}].device_id"
                ),
                capacity_bytes=_integer(
                    item.get("capacity_bytes"),
                    f"{path}.devices[{index}].capacity_bytes",
                ),
                fetch_bandwidth_bytes_per_second=_integer(
                    item.get("fetch_bandwidth_bytes_per_second"),
                    f"{path}.devices[{index}].fetch_bandwidth_bytes_per_second",
                ),
                evict_bandwidth_bytes_per_second=_integer(
                    item.get("evict_bandwidth_bytes_per_second"),
                    f"{path}.devices[{index}].evict_bandwidth_bytes_per_second",
                ),
                fetch_latency_ns=_integer(
                    item.get("fetch_latency_ns"),
                    f"{path}.devices[{index}].fetch_latency_ns",
                ),
                evict_latency_ns=_integer(
                    item.get("evict_latency_ns"),
                    f"{path}.devices[{index}].evict_latency_ns",
                ),
            )
            for index, raw in enumerate(devices)
            for item in (_mapping(raw, f"{path}.devices[{index}]"),)
        ),
        _integer(data.get("host_capacity_bytes"), f"{path}.host_capacity_bytes"),
    )
