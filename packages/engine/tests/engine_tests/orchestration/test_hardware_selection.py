"""Placement respects hardware, availability and explicit physical identities."""

from dataclasses import replace

import pytest

from shadowspill.orchestration.launch.admission import ResourcesUnavailable
from shadowspill.orchestration.nodes.selection import select_hardware
from shadowspill_client import Resources
from shadowspill_client.protocol.hardware import (
    HardwareDevice,
    HardwareInventory,
    HardwareNode,
)


def node(name, *, count=2, gpu="GPU-A", gib=24, pool="lab", busy=(), driver="driver"):
    return HardwareNode(
        name,
        pool,
        [
            HardwareDevice(f"{name}-gpu{i}", i, gpu, gib << 30, i not in busy)
            for i in range(count)
        ],
        128 << 30,
        100 << 30,
        32,
        driver,
    )


def choose(*nodes, ranks=2, resources=None, wait=False):
    return select_hardware(
        HardwareInventory(nodes),
        ranks=ranks,
        device_bytes=20 << 30,
        resources=resources,
        wait=wait,
    )


def test_automatic_selection_prefers_available_equivalent_devices_together():
    selected = choose(
        node("a", count=1),
        node("b", count=1),
        node("c"),
        node("d", gib=80),
        node("e", busy=(0, 1)),
    )
    assert [row.node for row in selected.workers] == ["c", "c"]
    assert selected.resources == Resources(pool="lab", gpu_type="GPU-A")
    assert [row.rank for row in selected.workers] == [0, 1]


def test_preview_selection_keeps_hardware_class_when_availability_changes():
    small = node("small", gpu="GPU-A")
    large = node("large", gpu="GPU-B", gib=80)
    preview = choose(small, large, wait=True)
    assert preview.resources.gpu_type == "GPU-A"
    busy = replace(small, devices=[replace(d, available=False) for d in small.devices])
    with pytest.raises(ResourcesUnavailable):
        choose(busy, large, resources=preview.resources)
    waiting = choose(busy, large, resources=preview.resources, wait=True)
    assert all(row.node == "small" for row in waiting.workers)


@pytest.mark.parametrize(
    "second",
    [
        node("b", count=1, gpu="GPU-B"),
        node("b", count=1, gib=48),
        node("b", count=1, driver="different"),
        node("b", count=1, pool="other"),
    ],
)
def test_automatic_selection_does_not_mix_unequal_classes_or_pools(second):
    with pytest.raises(ValueError, match="equivalent assigned"):
        choose(node("a", count=1), second)


def test_unknown_gpu_types_are_not_assumed_equivalent_across_nodes():
    a, b = node("a", count=1, gpu=None), node("b", count=1, gpu=None)
    with pytest.raises(ValueError, match="equivalent assigned"):
        choose(a, b)
    with pytest.raises(ValueError, match="mixed GPU types"):
        choose(a, b, resources=Resources(devices=["a-gpu0", "b-gpu0"]))


def test_exact_order_and_pool_constraints_survive_normalization():
    selected = choose(
        node("a"),
        node("b"),
        resources=Resources(pool="lab", node="b", devices=[1, "b-gpu0"]),
    )
    assert [row.device.id for row in selected.workers] == ["b-gpu1", "b-gpu0"]
    assert selected.resources == Resources(
        pool="lab", node="b", devices=["b-gpu1", "b-gpu0"], gpu_type="GPU-A"
    )
    with pytest.raises(ValueError, match="different compute pools"):
        choose(
            node("a"),
            node("b", pool="other"),
            resources=Resources(devices=["a-gpu0", "b-gpu0"]),
        )


@pytest.mark.parametrize(
    "resources,message",
    [
        (Resources(pool="missing"), "no assigned hardware"),
        (Resources(node="missing"), "no assigned hardware"),
        (Resources(gpu_type="other"), "equivalent assigned"),
        (Resources(devices=["a-gpu0"]), "device count"),
        (Resources(node="a", devices=[0, "a-gpu0"]), "same GPU twice"),
        (Resources(node="a", devices=[0, 2]), "unavailable or ambiguous"),
        (Resources(gpu_type="other", devices=["a-gpu0", "a-gpu1"]), "type/budget"),
    ],
)
def test_invalid_selection_never_silently_substitutes(resources, message):
    with pytest.raises(ValueError, match=message):
        choose(node("a"), resources=resources)


def test_undersized_and_explicitly_mixed_devices_are_rejected():
    with pytest.raises(ValueError, match="type/budget"):
        choose(node("a", gib=12), resources=Resources(node="a", devices=[0, 1]))
    with pytest.raises(ValueError, match="mixed GPU types"):
        choose(
            node("a"),
            node("b", gpu="GPU-B"),
            resources=Resources(devices=["a-gpu0", "b-gpu0"]),
        )
