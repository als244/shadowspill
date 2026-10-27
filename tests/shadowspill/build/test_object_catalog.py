"""Shared storage and off-device objects retain exact geometry."""

import torch

from shadowspill.build.lowering.objects import ObjectCatalog
from shadowspill.ir import ObjectRole, Persistence, SharedResidencyPolicy


def test_shared_residency_is_read_only_unless_emitted_tasks_write_it() -> None:
    catalog = ObjectCatalog(device_id="device_0")
    first = catalog.add(
        torch.ones(4),
        role=ObjectRole.INPUT,
        persistence=Persistence.STEP,
    )
    second = catalog.add(
        torch.ones(5),
        role=ObjectRole.INPUT,
        persistence=Persistence.STEP,
    )
    catalog.mark_shared_residency(
        first,
        SharedResidencyPolicy.SHARED_WRITABLE_CAUSAL,
        retain_spill_copy=False,
    )
    catalog.mark_shared_residency(
        second,
        SharedResidencyPolicy.SHARED_WRITABLE_UNORDERED,
        retain_spill_copy=False,
    )

    catalog.finalize_shared_writes((second,))

    policies = {
        item.alias_group_id: item.shared_residency for item in catalog.alias_groups()
    }
    assert policies[catalog.alias_id(first)] is (SharedResidencyPolicy.SHARED_READ_ONLY)
    assert policies[catalog.alias_id(second)] is (
        SharedResidencyPolicy.SHARED_WRITABLE_UNORDERED
    )


def test_cpu_aliases_have_no_gpu_extent_even_for_offset_views():
    catalog = ObjectCatalog(device_id="device_0")
    value = torch.arange(8)
    whole, sliced = (
        catalog.add(tensor, role=ObjectRole.INPUT, persistence=Persistence.STEP)
        for tensor in (value, value[2:5])
    )
    assert catalog.alias_id(whole) == catalog.alias_id(sliced)
    assert all(group.size_bytes == 0 for group in catalog.alias_groups())
    assert all(obj.size_bytes == obj.offset_bytes == 0 for obj in catalog.objects())
