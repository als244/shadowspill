"""Portable deployment selectors never enter the computation configuration."""

from dataclasses import replace

import pytest

from client_tests.test_training_config import resolved
from shadowspill_client import Resources
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.creation import TrainingCreation
from shadowspill_client.protocol.hardware import (
    HardwareDevice,
    HardwareInventory,
    HardwareNode,
)
from shadowspill_client.protocol.preview import TrainingDescription


def test_resources_snapshot_and_roundtrip_remain_separate_from_computation():
    devices = [0, "gpu-b"]
    resources = Resources(pool="lab", node="worker", devices=devices)
    devices.reverse()
    assert resources.devices == (0, "gpu-b")
    config = resolved()
    for record in (
        resources,
        TrainingDescription(config=config, resources=resources),
        TrainingCreation(config=config, resources=resources),
    ):
        assert decode(encode(record)) == record
    assert "resources" not in config.to_dict()


@pytest.mark.parametrize(
    "options",
    [
        {"pool": ""},
        {"gpu_type": "a" * 257},
        {"node": 4},
        {"devices": [0]},
        {"devices": []},
        {"devices": "gpu"},
        {"devices": ["gpu", "gpu"]},
        {"node": "worker", "devices": [True]},
        {"node": "worker", "devices": [-1]},
        {"devices": ["gpu" + str(i) for i in range(1025)]},
    ],
)
def test_ambiguous_or_unbounded_resource_requests_fail_locally(options):
    with pytest.raises((ValueError, TypeError)):
        Resources(**options)


def test_inventory_roundtrip_summary_and_identity_validation():
    device = HardwareDevice("gpu", 0, "GPU class", 24 << 30, False)
    node = HardwareNode("worker", "lab", [device], 64 << 30, 32 << 30, 8, "driver")
    inventory = HardwareInventory([node])
    assert decode(encode(inventory)) == inventory
    summary = inventory.summary()
    assert "GPU class" in summary and "Busy" in summary and "32.0 / 64.0" in summary
    assert "inspection reserves no hardware" in summary
    with pytest.raises(ValueError, match="ordered assigned indexes"):
        replace(node, devices=[replace(device, index=1)])
    with pytest.raises(ValueError, match="duplicate IDs"):
        HardwareInventory([node, node])
    with pytest.raises(ValueError, match="exceeds its assignment"):
        replace(node, host_available_bytes=65 << 30)


def test_description_requires_exactly_one_computation_source():
    with pytest.raises(ValueError, match="setup or resolved"):
        TrainingDescription(resources=Resources())
